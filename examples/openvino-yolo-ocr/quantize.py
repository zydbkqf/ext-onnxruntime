#!/usr/bin/env python3
"""Static int8 (QDQ) quantization for the three graphs in the YOLO + PP-OCR example.

    python quantize.py <fp32_dir> <int8_dir> [<calib_dir>]

Produces, next to the fp32 models:

    yolo11n_int8.onnx
    ch_PP-OCRv4_det_infer_int8.onnx
    ch_PP-OCRv4_rec_infer_int8.onnx

The calibration set is small on purpose (a dozen photos plus rendered text
lines): it is enough to exercise the activation ranges this pipeline actually
feeds, and it lets anyone reproduce the models with the same inputs. For a
production model, calibrate on a few hundred images from the real workload.

Requires: onnxruntime, onnx, pillow, numpy.
"""

import os
import random
import string
import sys

import numpy as np
import onnx
import onnx.version_converter
from PIL import Image, ImageDraw, ImageFont
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

YOLO_SIZE = 640
DET_LIMIT = 960
REC_HEIGHT = 48
REC_MAX_W = 320

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OPTS = dict(quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax)

# A detector's final Concat mixes box coordinates (0..640) with class
# probabilities (0..1); one MinMax range for both quantizes the probabilities
# to a couple of levels, and every score collapses to 0.
YOLO_OPTS = dict(OPTS, extra_options={
    "OpTypesToExcludeOutputQuantization": ["Concat", "Sigmoid"],
})


def constants_to_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """ORT's quantizer only looks at initializers; Paddle exports store weights
    as Constant nodes, which would otherwise be left in fp32."""
    graph = model.graph
    keep = []
    for node in graph.node:
        tensor = None
        if node.op_type == "Constant" and len(node.attribute) == 1:
            attr = node.attribute[0]
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor = attr.t
        if tensor is not None:
            tensor.name = node.output[0]
            graph.initializer.append(tensor)
        else:
            keep.append(node)
    del graph.node[:]
    graph.node.extend(keep)
    return model


def load_fp32(path: str) -> onnx.ModelProto:
    return constants_to_initializers(onnx.load(path))


class Reader(CalibrationDataReader):
    """Feeds pre-built NCHW float32 samples to the quantizer."""

    def __init__(self, name: str, samples: list[np.ndarray]):
        self._it = iter([{name: s} for s in samples])

    def get_next(self):
        return next(self._it, None)


def letterbox_nchw(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None]


def det_nchw(im: Image.Image) -> np.ndarray:
    """Long side <= 960, both sides multiples of 32, ImageNet normalisation."""
    w, h = im.size
    r = min(DET_LIMIT / w, DET_LIMIT / h)
    dw = max(32, round(w * r / 32) * 32)
    dh = max(32, round(h * r / 32) * 32)
    a = np.asarray(im.resize((dw, dh), Image.BILINEAR), dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return a.transpose(2, 0, 1)[None]


def rec_nchw(im: Image.Image) -> np.ndarray:
    w, h = im.size
    tw = min(REC_MAX_W, max(32, round(REC_HEIGHT * w / h)))
    a = np.asarray(im.resize((tw, REC_HEIGHT), Image.BILINEAR), dtype=np.float32)
    a = (a / 255.0 - 0.5) / 0.5
    return a.transpose(2, 0, 1)[None]


def rendered_lines(count: int = 32) -> list[Image.Image]:
    """Random one-line renders, light-on-dark and dark-on-light, for rec calibration."""
    fonts = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    fonts = [f for f in fonts if os.path.exists(f)]
    if not fonts:
        return []
    rnd = random.Random(7)
    alphabet = string.ascii_letters + string.digits + " .,:-/()"
    out = []
    for _ in range(count):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(4, 22)))
        size = rnd.randint(22, 44)
        font = ImageFont.truetype(rnd.choice(fonts), size)
        w = int(font.getlength(text)) + 16
        h = size + 14
        dark = rnd.random() < 0.5
        bg = (rnd.randint(0, 60),) * 3 if dark else (rnd.randint(200, 255),) * 3
        fg = (rnd.randint(200, 255),) * 3 if dark else (rnd.randint(0, 60),) * 3
        im = Image.new("RGB", (max(40, w), h), bg)
        ImageDraw.Draw(im).text((8, 4), text, font=font, fill=fg)
        out.append(im)
    return out


def quantize(src_path: str, dst_path: str, reader: Reader, opts=OPTS) -> None:
    model = load_fp32(src_path)
    if min(o.version for o in model.opset_import) < 13:
        # per-channel DequantizeLinear writes an `axis` attribute, which only
        # exists from opset 13 on; Paddle exports are opset 12.
        try:
            model = onnx.version_converter.convert_version(model, 13)
        except Exception as exc:                      # noqa: BLE001
            print(f"opset bump failed for {src_path}: {exc}; quantising per-tensor",
                  file=sys.stderr)
            opts = dict(opts, per_channel=False)
    tmp = dst_path + ".fp32.onnx"
    onnx.save(model, tmp)
    quantize_static(tmp, dst_path, reader, **opts)
    os.remove(tmp)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    calib = sys.argv[3] if len(sys.argv) > 3 else src
    os.makedirs(dst, exist_ok=True)

    images = sorted(
        os.path.join(calib, f) for f in os.listdir(calib)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        print(f"no calibration images in {calib}", file=sys.stderr)
        return 1

    # ---- YOLO: 640x640 letterboxed, 0..1 ------------------------------------
    yolo = Reader("images", [letterbox_nchw(p, YOLO_SIZE) for p in images])
    quantize(os.path.join(src, "yolo11n.onnx"),
             os.path.join(dst, "yolo11n_int8.onnx"), yolo, YOLO_OPTS)

    # ---- det: ROI-sized crops with ImageNet normalisation -------------------
    det_samples = []
    for p in images:
        im = Image.open(p).convert("RGB")
        det_samples.append(det_nchw(im))
        w, h = im.size
        # a tighter crop on purpose: the pipeline feeds ROI crops, not full frames
        det_samples.append(det_nchw(im.crop((w // 6, h // 3, w * 5 // 6, h * 4 // 5))))
    quantize(os.path.join(src, "ch_PP-OCRv4_det_infer.onnx"),
             os.path.join(dst, "ch_PP-OCRv4_det_infer_int8.onnx"),
             Reader("x", det_samples))

    # ---- rec: 48px text lines ----------------------------------------------
    rec_samples = [rec_nchw(im) for im in rendered_lines()]
    for p in images:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        # a wide band through the middle: text-like aspect ratios
        rec_samples.append(rec_nchw(im.crop((0, h // 3, w, h // 3 + h // 12))))
    quantize(os.path.join(src, "ch_PP-OCRv4_rec_infer.onnx"),
             os.path.join(dst, "ch_PP-OCRv4_rec_infer_int8.onnx"),
             Reader("x", rec_samples))

    for f in sorted(os.listdir(dst)):
        if f.endswith("_int8.onnx"):
            print(f"{f:42s} {os.path.getsize(os.path.join(dst, f)) / 1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
