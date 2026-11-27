import json, os
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
lines = [("ONNX Runtime PHP", 96, 80, 120), ("Letterbox and Resize Test", 72, 80, 300), ("2026-11-26 OCR 12345", 64, 80, 470)]
for text, size, x, y in lines:
    font = ImageFont.truetype("DejaVuSans.ttf", size)
    d.text((x, y), text, font=font, fill=(20, 20, 20))
img.save("/tmp/ort-img/work/test.jpg", quality=85)
img.save("/tmp/ort-img/work/test.png")

d2 = Image.new("RGB", (1200, 900), (255, 255, 255))
ImageDraw.Draw(d2).text((40, 40), "letterbox", font=ImageFont.truetype("DejaVuSans.ttf", 80), fill=(0,0,0))
d2.save("/tmp/ort-img/work/tall.png")

img = Image.open("/tmp/ort-img/work/test.jpg").convert("RGB")
img.resize((960, 544), Image.BILINEAR).save("/tmp/ort-img/work/det_target.png")
open("/tmp/ort-img/work/det_target.raw","wb").write(img.resize((960,544), Image.BILINEAR).tobytes())
open("/tmp/ort-img/work/src.raw","wb").write(img.tobytes())

# crop line 1 for rec
crop = img.crop((70, 110, 1150, 240))
cw, ch = crop.size
rh = 48
rw = int(round(cw * rh / ch))
crop.resize((rw, rh), Image.BILINEAR).save("/tmp/ort-img/work/rec_target.png")
open("/tmp/ort-img/work/rec_target.raw","wb").write(crop.resize((rw, rh), Image.BILINEAR).tobytes())
open("/tmp/ort-img/work/rec_src.raw","wb").write(crop.tobytes())
meta = {"image": [W,H], "det": [960,544], "rec_crop": [cw,ch], "rec_target": [rw,rh]}
open("/tmp/ort-img/work/meta.json","w").write(json.dumps(meta, indent=2))
print(json.dumps(meta))
print("jpg bytes", os.path.getsize("/tmp/ort-img/work/test.jpg"))
