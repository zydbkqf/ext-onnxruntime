# End-to-end example: YOLO11 + PP-OCR on Linux, CPU EP vs OpenVINO EP

A complete PHP pipeline whose only image dependency is GD:

```text
JPEG
 └─ GD decode + letterbox 640x640 ──► YOLO11n ──► bus box
      └─ GD crop ROI, 2x ──► PP-OCRv4 det ──► text-line boxes
            └─ GD crop 48px tall ──► PP-OCRv4 rec ──► CTC text
```

The runnable script is [`examples/openvino-yolo-ocr/pipeline.php`](../examples/openvino-yolo-ocr/pipeline.php).
It prints one row per stage with `t=` (milliseconds since process start) and the
stage cost, both from `hrtime(true)`, so the rows can be matched against an
external timeline or a profiler. The same script runs on the built-in CPU EP
(`cpu`) and on the OpenVINO EP (`openvino`); every number below is Linux-only
and was measured on one machine:

```text
Host        Intel Xeon Platinum 8488C (Sapphire Rapids), 2 vCPU @ 2.4 GHz, 7 GiB
            flags: avx512f, avx512_vnni, avx512_bf16, amx_int8, amx_bf16, avx_vnni
OS          Debian 13 (trixie), kernel 6.12, docker 29.10.1
PHP         8.5.13 CLI NTS inside php:8.5-cli, no opcache, GD with freetype+jpeg
ORT         1.32.0 CPU package (official NuGet build)
OV plugin   1.8.1, CPU device, INFERENCE_NUM_THREADS=2 from load_config
```

## What you need installed

| Requirement | Detail |
| --- | --- |
| PHP 8.5 | NTS is fine. The example was run on `php:8.5-cli` (PHP 8.5.13). |
| GD | `--with-freetype --with-jpeg`. Only `imagecreatefromjpeg`, `imagecopyresampled`, `imagebmp` are used. |
| Extension | `ext-onnxruntime` 0.4.14+, built against **ORT 1.32.0** (see [EP-OPENVINO.md](EP-OPENVINO.md) for the plugin/ORT version rule). |
| OpenVINO plugin | `Intel.ML.OnnxRuntime.EP.OpenVINO` **1.8.1**, unpacked to `runtimes/linux-x64/native/`. |
| Models | YOLO11n and PP-OCRv4 det/rec, plus the rec dictionary — see below. All three have dynamic H/W inputs. |
| Image | Any JPEG. The doc uses a 1280x1601 street photo. |

The ORT package usually ships `libonnxruntime.so` without a version suffix, while
the extension links against `libonnxruntime.so.1`; add the symlink or the module
fails to load with `libonnxruntime.so.1: cannot open shared object file`.

## Model directory

```text
models/
  yolo11n.onnx                    10.9 MB  fp32; dynamic batch/H/W input
  ch_PP-OCRv4_det_infer.onnx       4.7 MB  fp32; dynamic H/W, output 1x1xHxW map
  ch_PP-OCRv4_rec_infer.onnx      10.9 MB  fp32; input 1x3x48xW, output 1xTx6625
  ppocr_keys_v1.txt                 26 KB  6622 entries, index 0 = CTC blank
  # optional, written by quantize.py, used when INT8=1:
  yolo11n_int8.onnx                3.5 MB
  ch_PP-OCRv4_det_infer_int8.onnx  1.55 MB
  ch_PP-OCRv4_rec_infer_int8.onnx  3.19 MB
```

```bash
mkdir -p models && cd models

# YOLO11n, already exported to ONNX in the ultralytics asset release
curl -L -o yolo11n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx

# PP-OCRv4 det/rec: the rapidocr wheel ships exactly these two ONNX files
pip download rapidocr_onnxruntime==1.4.4 --no-deps -d /tmp/rr
unzip -j /tmp/rr/rapidocr_onnxruntime-*.whl 'rapidocr_onnxruntime/models/*' -d .
# -> ch_PP-OCRv4_det_infer.onnx, ch_PP-OCRv4_rec_infer.onnx

# rec dictionary (mirror of ppocr/utils/ppocr_keys_v1.txt)
curl -L -o ppocr_keys_v1.txt \
  https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.7/ppocr/utils/ppocr_keys_v1.txt
```

All three graphs are plain fp32: `yolo11n.onnx` holds 184 FLOAT initializers
(10.47 MB) plus INT64 shape constants and no `QuantizeLinear` / `QLinearConv` /
`MatMulInteger` node, and the two PP-OCRv4 files keep their weights in fp32
`Constant` tensors.

## All three models ship fp32 — int8 is self-built

Neither upstream project publishes an int8 ONNX file for this trio:

- The ultralytics release that carries `yolo11n.onnx` (v8.3.0, and v8.2.0 /
  v8.1.0) has no `*int8*.onnx` asset at all — the `-int8` variants exist only as
  engine-specific exports produced locally by `model.export(int8=True)`.
- `rapidocr_onnxruntime` 1.4.4 ships the two PP-OCRv4 files in fp32, and the
  newer `rapidocr` 3.9.2 wheel ships PP-OCRv6 small det/rec, also fp32.
- Paddle's own published OCR weights are `.pdmodel`/`.pdiparams` archives; they
  need `paddle2onnx` plus a `paddlepaddle` install before ORT sees them, and the
  result is still fp32.

So the example carries a quantizer instead,
[`examples/openvino-yolo-ocr/quantize.py`](../examples/openvino-yolo-ocr/quantize.py):
static QDQ, per-channel int8 weights, QUInt8 activations, MinMax calibration.

```bash
python3 quantize.py models models_int8 calib   # calib = a dozen photos
# ch_PP-OCRv4_det_infer_int8.onnx   1.55 MB  (from 4.70)
# ch_PP-OCRv4_rec_infer_int8.onnx   3.19 MB  (from 10.86)
# yolo11n_int8.onnx                 3.52 MB  (from 10.93)
```

Two things had to be worked around, both easy to miss:

- **Paddle exports are opset 12.** ORT's per-channel `DequantizeLinear` writes an
  `axis` attribute, which the standard only allows from opset 13; the model then
  fails to load with `Unrecognized attribute: axis for operator DequantizeLinear`.
  `quantize.py` bumps the model to opset 13 first and falls back to per-tensor
  weights if the conversion fails.
- **A detector's last `Concat` mixes box coordinates with class scores.** One
  MinMax range over `[0, 640]` boxes and `[0, 1]` probabilities leaves the scores
  with a step of ~2.5 units: every anchor quantizes to 0 and the model returns
  **zero detections** with correct boxes. Excluding `Concat`/`Sigmoid` outputs
  from quantisation keeps `max score 0.851` against `0.865` fp32 and the same
  50 anchors above 0.35.

Accuracy holds at the level this pipeline cares about: the bus is still the top
object (0.86 fp32 -> 0.79 int8), the destination sign still reads `156LASALLE`
on both EPs, and the det box moves by a dozen pixels on the 512-px canvas.

## Run it

The example ran inside `php:8.5-cli` with everything under a single mount:

```bash
docker run -d --name ort-ov \
  -v /tmp/ort:/host --entrypoint sleep php:8.5-cli infinity

docker exec ort-ov bash -c '
  apt-get update -qq &&
  apt-get install -y -qq libpng-dev libjpeg-dev libfreetype6-dev &&
  docker-php-ext-configure gd --with-freetype --with-jpeg &&
  docker-php-ext-install -j2 gd'

# both the ORT library and the plugin directory must be on LD_LIBRARY_PATH
docker exec \
  -e LD_LIBRARY_PATH=/host/ort/lib:/host/openvino-ep/runtimes/linux-x64/native \
  -e OV_PLUGIN=/host/openvino-ep/runtimes/linux-x64/native/libonnxruntime_providers_openvino_plugin.so \
  ort-ov php -d extension=/host/src/modules/onnxruntime.so \
  /host/src/examples/openvino-yolo-ocr/pipeline.php \
  /host/demo.jpg /host/models openvino
```

The script takes `<image> <model_dir> <cpu|openvino>` and reads `OV_PLUGIN` from
the environment. Environment switches, all optional:

| Variable | Effect |
| --- | --- |
| `INT8=1` | use the `*_int8.onnx` files written by `quantize.py` |
| `DET_SCALE=2.0` | det canvas = ROI long side × this, default 2.0 (see below) |
| `DET_LIMIT=960` | hard cap for the det canvas long side |
| `THREADS=2` | intra-op threads per session (`load_config` for OpenVINO) |
| `REPS=N` | add steady-state medians for the three sessions |

Expected tail of the output:

```text
objects    bus@0.86, car@0.85, car@0.74, car@0.72, car@0.62
text
   "156LASALLE"      roi 102,42-252,69      src 358,1009-432,1023  33 steps  150x27->267x48
   ...
```

`156LASALLE` is the bus destination sign; the `roi`/`src` columns are the box on
the det canvas and in the original image. The remaining lines are roof signs and
plates — a det model on a street scene finds text everywhere, which is why the
pipeline runs det before rec rather than rec over the whole frame.

## Cold start: where the first image goes

Single-shot timeline, one row per stage (CPU EP fp32 / OpenVINO EP fp32):

| # | Stage | CPU EP | OpenVINO EP |
| --- | --- | --- | --- |
| 1 | GD `imagecreatefromjpeg` 1280x1601 | 26.4 ms | 26.2 |
| 2 | GD letterbox -> 640x640 | 38.0 | 43.7 |
| 3 | GD `imagebmp` -> RGB bytes (1.2 MB) | 37.3 | 36.6 |
| 4 | `Tensor::fromBuffer` (hwc, 1/255) | 4.4 | 4.4 |
| 5 | YOLO `Session` create | 57.4 | **385.7** |
| 6 | YOLO run (first) | 75.7 | 166.3 |
| 7 | YOLO decode + NMS | 29.0 | 21.4 |
| 8 | GD crop ROI 250x181 -> 512x352 | 9.5 | 7.4 |
| 9 | det `fromBuffer` + `Session` create | 75.1 | **250.3** |
| 10 | det run (first) | 90.2 | 69.4 |
| 11 | det `toBuffer` + DB components | 11.4 | 11.8 |
| 12 | rec `Session` create | 120.0 | **403.4** |
| 13/14 | 2 lines: crop+`fromBuffer` / rec | 5.9 / 50.4 | 3.6 / 85.8 |
| | **total** | **~590 ms** | **~1500 ms** |

OpenVINO loses the single-image race by ~2.5x purely on graph compilation:
385 + 250 + 403 ms of `Session` create plus a fat first inference. The CPU EP
pays 57 + 75 + 120 ms for the same three sessions.

That cost is one-time per process. Which is why the next section is the
important one.

## Steady state: fp32 vs int8, CPU EP vs OpenVINO EP

Warm medians (`REPS=9`, same sessions, same input tensors, `THREADS=2`,
det canvas 512x352):

| Model | CPU EP fp32 | CPU EP int8 | OV EP fp32 | OV EP int8 |
| --- | --- | --- | --- | --- |
| YOLO11n 640x640 | 106 ms | **73 ms** | **30 ms** | 66 ms |
| OCR det 512x352 | 49 ms | 44 ms | **15 ms** | 48 ms |
| OCR rec 200x48 | 26 ms | 31 ms | **4 ms** | 26 ms |
| models per image | 181 ms | 148 ms | **49 ms** | 140 ms |
| session create (3) | 252 ms | 505 ms | 1039 ms | **5.4 s** |
| cold total | 589 ms | 812 ms | 1499 ms | 6219 ms |

Two conclusions, and they contradict each other in a useful way:

- **fp32 OpenVINO is the fastest configuration on this box** — 3.7x faster than
  fp32 CPU EP on the models alone. Sapphire Rapids exposes AMX, and the 1.8.1
  CPU plugin's oneDNN backend reaches for it; the built-in ORT CPU EP does not
  get anywhere near the same convolution throughput with two threads.
- **int8 wins only on the CPU EP, and only clearly on YOLO** (106 -> 73 ms).
  Under OpenVINO, int8 is *slower* than fp32 on every graph, and its session
  create cost explodes from 1.0 s to 5.4 s (the QDQ graph has to be compiled and
  fused, and 1.8.1 does not end up with AMX int8 kernels for these shapes).

With the measured PHP side (~150 ms: 103 ms GD + 29 ms NMS + 18 ms DB and crops):

| Per image | CPU EP fp32 | CPU EP int8 | OV EP fp32 | OV EP int8 |
| --- | --- | --- | --- | --- |
| wall clock | 331 ms | 297 ms | **199 ms** | 290 ms |
| single-stream rate | 3.0 img/s | 3.4 img/s | **5.0 img/s** | 3.4 img/s |

So: pick OpenVINO + fp32 here, and spend the next effort on the PHP side, which
is 75% of the remaining per-image time.

## Why int8 is sometimes slower: VNNI, AMX, bf16

int8 inference only pays when the CPU has instructions that multiply 8-bit
matrices directly. On x86 that is one of:

| Extension | Requirement | What it gives |
| --- | --- | --- |
| `avx512_vnni` | Cascade Lake / Ice Lake+ | `VPDPBUSD`, int8 dot products in AVX-512 registers |
| `avx_vnni` | Alder Lake+ without AVX-512 | the same idea on 256-bit registers |
| `amx_int8` | Sapphire Rapids+ | tile-based int8 matmul, several times VNNI |
| `avx512_bf16` / `amx_bf16` | Cooper Lake / Sapphire Rapids | bf16, a *different* path from int8 |

If none of them is present, an int8 graph still runs — every `DequantizeLinear`
/ `QuantizeLinear` pair forces a convert back to fp32 around the convolution,
so you pay dequantisation on top of the same fp32 math. Measured on a Xeon VM
that had `avx512f` but **no** `avx512_vnni`, int8 changed nothing on the CPU EP
(YOLO 52.8 -> 48.0 ms) and made OpenVINO dramatically worse (46 -> 64 ms warm and
8.2 s cold, all compilation). On the AMX machine the same files give a real
1.45x on the CPU EP's YOLO.

Checking a host takes one command:

```bash
grep -Eo 'avx512_vnni|avx_vnni|amx_int8|avx512_bf16|amx_bf16' /proc/cpuinfo | sort -u
# ARM: grep -Eo 'i8mm|dotprod' /proc/cpuinfo | sort -u
```

Two caveats that trip people up:

- **Instructions are not enough; the kernels have to use them.** This box has
  `amx_int8` and int8 still loses under OpenVINO 1.8.1, because the plugin's
  int8 handling for QDQ graphs does not beat its own fp32 (bf16/AMX) path for
  these layer shapes. The same machine shows int8 winning on the ORT CPU EP and
  losing on OpenVINO — same hardware, different kernels.
- **bf16 is not int8.** `avx512_bf16` lets a *fp32* graph run faster by rounding
  internally; it says nothing about whether quantized weights will be fast.

That is why the example keeps fp32 as the default and int8 behind `INT8=1`: the
right choice depends on the EP and the CPU generation, not on "int8 is always
faster". The README's recommendation to quantize assumes a CPU with VNNI/AMX
and an EP that maps them; both halves of that assumption have to hold.

## Tuning: what actually moves the number

Measured on the same box, one change at a time:

- **Size the det canvas to the ROI, not to a constant.** The ROI here is
  250x181, so a 960-px canvas is a 4x upscale that adds no information. At
  512x352 (2x) det is 15 ms under OpenVINO and still reads `156LASALLE`
  (it was 55 ms at 960); at 320 (native-ish) it is 6 ms but loses most of the
  smaller lines. `DET_SCALE=2.0` is the default, `DET_LIMIT` caps it for large ROIs.
- **Keep sessions resident.** Per-image session creation is 252 ms (CPU EP) or
  1.0 s (OpenVINO); a Swoole/php-fpm pool that builds the three sessions once
  removes that outright. This single change is worth more than any EP choice
  for low-traffic servers.
- **Threads.** `THREADS=2` matches the 2 vCPU here; 4 threads on a 2-core box
  oversubscribes (measured 873 -> 933 ms cold on the older VM). OpenVINO 1.8.1
  ignores `setIntraOpNumThreads` entirely — its count only comes from
  `load_config`, which is why the script writes it there too.
- **Batch the rec calls.** The 2-5 det lines could go through one `[N,3,48,W]`
  batch (or a bucket of two widths) instead of N calls; rec is only 4-26 ms so
  this is a few percent, not a win worth code churn.
- **Move NMS/DB out of PHP.** 29 ms NMS over 8400 anchors plus 11-18 ms of
  8-connected components is pure PHP looping; in C (or behind an extension
  decode API) that is ~40 ms back per image on this box.
- **int8 last.** It is a 1.45x on YOLO for the CPU EP here, and a regression on
  OpenVINO. Only reach for it after the session and canvas decisions are made.

## The parts that matter

**GD has no raw-pixel API in PHP.** `imagebmp($im, null, false)` with
compression off is the cheapest bridge to bytes: an uncompressed 24-bit BMP is
a header plus BGR rows stored bottom-up with a 4-byte stride. Reversing each
3-byte pixel (`strrev`) and walking rows backwards gives top-down RGB, at ~37 ms
for 640x640 versus ~48 ms for 409,600 `imagecolorat()` calls:

```php
ob_start();
imagebmp($im, null, false);
$bmp = ob_get_clean();

$off    = unpack('V', substr($bmp, 10, 4))[1];
$stride = intdiv($imagesx * 3 + 3, 4) * 4;
for ($y = 0; $y < $imagesy; $y++) {
    $row  = substr($bmp, $off + ($imagesy - 1 - $y) * $stride, $imagesx * 3);
    $rgb .= implode('', array_map('strrev', str_split($row, 3)));
}
```

**One `fromBuffer` call covers layout, cast and normalisation.** YOLO wants
`[1,3,640,640]` float32 in 0..1 from HWC bytes; the OCR models want ImageNet
mean/std (det) or 0.5/0.5 (rec):

```php
$t = OnnxRuntime\Tensor::fromBuffer($rgb, [1, 3, 640, 640], 'uint8', [
    'layout' => 'hwc',
    'cast'   => 'float32',
    'scale'  => 1 / 255,                    // v = (v * scale - mean) / std
]);
```

**Appending the plugin device.** The plugin is registered once; every session
then gets the same `EpDevice` plus its own thread config:

```php
$runtime = OnnxRuntime\Runtime::default();
$runtime->registerExecutionProviderLibrary('openvino', $plugin);
foreach ($runtime->getEpDevices() as $d) {
    if ($d->epName() === 'OpenVINOExecutionProvider') { $device = $d; }
}
$so->appendEpDevice($device, [
    'load_config' => '{"CPU":{"INFERENCE_NUM_THREADS":"2"}}',
]);
```

Registering the same library twice throws `execution provider library is
already registered: openvino`, so the example keeps the device in a `static`.

**YOLO output.** `[1,84,8400]`: four box channels, then 80 class scores per
anchor. `toBuffer()` + `unpack('g*')` reads the 705,600 floats without building
nested arrays, then a plain NMS pass runs in PHP (29 ms including the parse).

**DB postprocess, simplified.** The det output is a probability map; real DB
does contour tracing, box scoring and unclip. The example thresholds at 0.3,
finds 8-connected components (one per text line), and crops from the **det
canvas**, not from the source image — the ROI was already upscaled once, so a
12 px tall sign becomes a 48 px tall rec input instead of a 4x upscale of blur.
That single change is what turns the output from `56LASALL` into `156LASALLE`.

## Caveats

- One image, one code path. No batching, no queuing, no async; for that see the
  `Session::runAsync` material in the README.
- The DB step is a demo-grade simplification: components are boxes, not
  polygons, and rotated text is not handled. A full implementation needs
  `unclip` on the polygon.
- The class list is the 80-entry COCO set; a different YOLO export needs its
  own names.
- `yolo11n.onnx` has a dynamic input (`[N,3,H,W]`), so `YOLO_SIZE` can change
  without re-exporting — keep it a multiple of 32 and update the letterbox.
- If OpenVINO's CPU plugin refuses the host CPU, the sample does not fall back
  silently: `openvino_device()` throws when no device is exposed. Pass `cpu` as
  the third argument to run the same pipeline on the built-in EP.
- Numbers on this box vary by up to ~20% between runs (2 vCPU, shared cloud
  host); all tables are medians of at least three runs.
- The calibration set in `quantize.py` is a dozen images. A shipped model
  deserves a few hundred samples from the real workload.

## See also

- [EP-OPENVINO.md](EP-OPENVINO.md) — plugin packaging, device names, option table
- [EP-CPU.md](EP-CPU.md) — built-in CPU EP and thread settings
- [benchmarks/README.md](../benchmarks/README.md) — OpenVINO vs CPU EP on BGE-small
