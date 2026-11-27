<?php
declare(strict_types=1);

/*
 * End-to-end detection + OCR on the OpenVINO EP, with GD doing the image work.
 *
 *   JPEG -> GD letterbox -> YOLO11n (OpenVINO EP) -> bus box
 *        -> GD crop -> PP-OCRv4 det -> GD crop -> PP-OCRv4 rec -> text
 *
 * ---- machine used for the numbers in docs/EXAMPLE-OPENVINO-YOLO-OCR.md --------
 * Host          : Intel Xeon Platinum 8488C (Sapphire Rapids), 2 vCPU @ 2.4 GHz,
 *                 7 GiB RAM, Debian 13, kernel 6.12, docker 29.10.1
 *                 flags: avx512f avx512_vnni avx512_bf16 amx_int8 amx_bf16 avx_vnni
 * PHP           : 8.5.13 CLI NTS inside php:8.5-cli, no opcache, GD built with
 *                 --with-freetype --with-jpeg
 * ORT           : 1.32.0 CPU package (official NuGet build)
 * OpenVINO EP   : Intel.ML.OnnxRuntime.EP.OpenVINO 1.8.1, CPU device
 * Threads       : 2 for both backends. OpenVINO is pinned with the device-scoped
 *                 load_config {"CPU":{"INFERENCE_NUM_THREADS":"2"}}; 1.8.1 ignores
 *                 setIntraOpNumThreads entirely (logs "num_of_threads is ignored").
 * Container     : docker run -d --name ort-ov -v /tmp/ort:/host --entrypoint sleep
 *                 php:8.5-cli infinity  (no --cpus, no --memory limits)
 * -------------------------------------------------------------------------------
 *
 * Every stage prints `t=<ms since process start>` and `<ms spent>`, both read
 * from hrtime(true), so a row can be matched against an external timeline.
 *
 * Usage:
 *   php pipeline.php <image.jpg> <model_dir> [cpu|openvino]
 *   model_dir contains yolo11n.onnx, ch_PP-OCRv4_det_infer.onnx,
 *   ch_PP-OCRv4_rec_infer.onnx and ppocr_keys_v1.txt. INT8=1 switches to the
 *   *_int8.onnx builds that quantize.py writes next to them.
 *
 *   Environment:
 *     INT8=1        use the *_int8.onnx files from quantize.py
 *     DET_LIMIT=N   cap for the det canvas long side, default 960
 *     DET_SCALE=X   det canvas = ROI long side * X, default 2.0
 *     THREADS=N     intra-op threads per session, default 2
 *     REPS=N        add steady-state medians for the three sessions
 */

const YOLO_SIZE = 640;          // fixed by the exported graph
const YOLO_CONF = 0.35;
const YOLO_IOU  = 0.45;
const DET_LIMIT = 960;          // cap for the det canvas long side
const DET_SCALE = 2.0;          // det canvas = ROI long side * DET_SCALE, capped by DET_LIMIT
const THREADS   = 2;            // intra-op threads per session; 4 logical cores here
const DET_THRESH = 0.30;
const REC_HEIGHT = 48;
const REC_MAX_W  = 320;

$imagePath = $argv[1] ?? __DIR__ . '/demo.jpg';
$modelDir  = rtrim($argv[2] ?? getenv('MODEL_DIR') ?: __DIR__, '/');
$backend   = $argv[3] ?? 'openvino';
// Overridable so throughput can be traded against small-text recall without
// editing the file: DET_LIMIT=640 php pipeline.php ... / THREADS=4 php ...
$detLimit  = (int) (getenv('DET_LIMIT') ?: DET_LIMIT);
$detScale  = (float) (getenv('DET_SCALE') ?: DET_SCALE);
$threads   = (int) (getenv('THREADS') ?: THREADS);

$T0 = hrtime(true);
$last = $T0;

/** Print one stage row: absolute ms since start, plus the stage cost. */
function stage(string $name, int &$last, int $t0, string $note = ''): void
{
    $now = hrtime(true);
    printf("t=%9.2f ms   %8.2f ms   %-34s%s\n",
        ($now - $t0) / 1e6, ($now - $last) / 1e6, $name, $note === '' ? '' : "  " . $note);
    $last = $now;
}

/**
 * GD image -> interleaved RGB bytes, row-major, top-down.
 *
 * GD has no raw-pixel API in PHP, so the cheapest bridge is imagebmp() with
 * compression off: an uncompressed 24-bit BMP is a header plus BGR rows that
 * are stored bottom-up and padded to a 4-byte stride. Reversing each 3-byte
 * pixel with strrev() turns BGR into RGB, and walking the rows backwards turns
 * bottom-up into top-down. That is one str_split + strrev pass per row (24 ms
 * for 640x640 on the machine above) instead of 409,600 imagecolorat() calls
 * (48 ms) and it never touches the PHP heap per pixel.
 */
function gd_to_rgb(GdImage $im): string
{
    $w = imagesx($im);
    $h = imagesy($im);
    ob_start();
    imagebmp($im, null, false);
    $bmp = ob_get_clean();

    $off    = unpack('V', substr($bmp, 10, 4))[1];
    $bmpW   = unpack('V', substr($bmp, 18, 4))[1];
    $bmpH   = unpack('V', substr($bmp, 22, 4))[1];
    $bpp    = unpack('v', substr($bmp, 28, 2))[1];
    if ($bmpW !== $w || $bmpH !== $h || $bpp !== 24) {
        throw new RuntimeException("Unexpected BMP: {$bmpW}x{$bmpH} {$bpp}bpp (GD writes 24bpp for opaque truecolor)");
    }
    $stride = intdiv($w * 3 + 3, 4) * 4;

    $rgb = '';
    for ($y = 0; $y < $h; $y++) {
        $row = substr($bmp, $off + ($h - 1 - $y) * $stride, $w * 3);
        $rgb .= implode('', array_map('strrev', str_split($row, 3)));
    }
    return $rgb;
}

/** Letterbox the source into a size×size square, padding with 114 grey. */
function letterbox(GdImage $src, int $size): array
{
    $w = imagesx($src);
    $h = imagesy($src);
    $scale = min($size / $w, $size / $h);
    $nw = (int) round($w * $scale);
    $nh = (int) round($h * $scale);
    $padX = intdiv($size - $nw, 2);
    $padY = intdiv($size - $nh, 2);

    $dst = imagecreatetruecolor($size, $size);
    imagefilledrectangle($dst, 0, 0, $size, $size, imagecolorallocate($dst, 114, 114, 114));
    imagecopyresampled($dst, $src, $padX, $padY, 0, 0, $nw, $nh, $w, $h);
    return [$dst, $scale, $padX, $padY];
}

/** Crop + resize in one GD call, preserving aspect ratio into a padded box. */
function crop_resize(GdImage $src, int $x1, int $y1, int $x2, int $y2, int $tw, int $th): GdImage
{
    $dst = imagecreatetruecolor($tw, $th);
    imagefilledrectangle($dst, 0, 0, $tw, $th, imagecolorallocate($dst, 0, 0, 0));
    imagecopyresampled($dst, $src, 0, 0, $x1, $y1, $tw, $th, max(1, $x2 - $x1), max(1, $y2 - $y1));
    return $dst;
}

/**
 * Resolve a model file. fp32 is the default; INT8=1 switches to the quantized
 * build produced by quantize.py (and fails loudly when it was never built).
 */
function model_file(string $dir, string $stem): string
{
    if (getenv('INT8') !== '1') {
        return $dir . '/' . $stem . '.onnx';
    }
    $int8 = $dir . '/' . $stem . '_int8.onnx';
    if (!is_file($int8)) {
        throw new RuntimeException("INT8=1 but no quantized model: {$int8}");
    }
    return $int8;
}

/** Register the OpenVINO plugin once and return its CPU device. */
function openvino_device(): OnnxRuntime\EpDevice
{
    static $device = null;
    if ($device !== null) {
        return $device;
    }
    $plugin = getenv('OV_PLUGIN') ?: '/opt/openvino-ep/runtimes/linux-x64/native/libonnxruntime_providers_openvino_plugin.so';
    OnnxRuntime\Runtime::default()->registerExecutionProviderLibrary('openvino', $plugin);
    foreach (OnnxRuntime\Runtime::default()->getEpDevices() as $d) {
        if ($d->epName() === 'OpenVINOExecutionProvider') {
            return $device = $d;
        }
    }
    throw new RuntimeException('OpenVINO plugin registered but exposed no CPU device');
}

/** SessionOptions with the requested backend appended. */
function make_options(string $backend, int $threads): OnnxRuntime\SessionOptions
{
    $so = new OnnxRuntime\SessionOptions();
    $so->setIntraOpNumThreads($threads);

    if ($backend === 'openvino') {
        $device = openvino_device();
        // Device-scoped JSON is the only knob the plugin honours; a plain
        // num_of_threads option is logged and ignored by 1.8.1.
        $so->appendEpDevice($device, [
            'load_config' => '{"CPU":{"INFERENCE_NUM_THREADS":"' . $threads . '"}}',
        ]);
    }
    return $so;
}

/** Greedy NMS over [x1,y1,x2,y2,score,class] rows. */
function nms(array $boxes, float $iou): array
{
    usort($boxes, fn($a, $b) => $b[4] <=> $a[4]);
    $keep = [];
    foreach ($boxes as $b) {
        foreach ($keep as $k) {
            $ix1 = max($b[0], $k[0]); $iy1 = max($b[1], $k[1]);
            $ix2 = min($b[2], $k[2]); $iy2 = min($b[3], $k[3]);
            $iw = max(0.0, $ix2 - $ix1); $ih = max(0.0, $iy2 - $iy1);
            $inter = $iw * $ih;
            $union = ($b[2] - $b[0]) * ($b[3] - $b[1]) + ($k[2] - $k[0]) * ($k[3] - $k[1]) - $inter;
            if ($union > 0 && $inter / $union > $iou) {
                continue 2;
            }
        }
        $keep[] = $b;
    }
    return $keep;
}

/** CTC greedy decode from the raw [1,T,C] float32 payload. */
function ctc_decode(string $bytes, int $steps, int $classes, array $dict): string
{
    $f = unpack('g*', $bytes);           // 1-based, float32 little-endian
    $text = '';
    $prev = -1;
    for ($t = 0; $t < $steps; $t++) {
        $base = $t * $classes;
        $best = 0;
        $bestV = $f[$base + 1];
        for ($c = 1; $c < $classes; $c++) {
            $v = $f[$base + $c + 1];
            if ($v > $bestV) { $bestV = $v; $best = $c; }
        }
        if ($best !== $prev && $best !== 0) {
            $text .= $dict[$best - 1] ?? '';
        }
        $prev = $best;
    }
    return $text;
}

// ---------------------------------------------------------------- stage 1: decode
$im = imagecreatefromjpeg($imagePath);
if ($im === false) {
    throw new RuntimeException("Cannot decode {$imagePath}");
}
$w = imagesx($im);
$h = imagesy($im);
stage("1. GD imagecreatefromjpeg {$w}x{$h}", $last, $T0);

// ---------------------------------------------------------------- stage 2: letterbox
[$lb, $scale, $padX, $padY] = letterbox($im, YOLO_SIZE);
stage("2. GD letterbox -> " . YOLO_SIZE . "x" . YOLO_SIZE, $last, $T0, "scale=" . round($scale, 4));

// ---------------------------------------------------------------- stage 3: GD -> bytes
$rgb = gd_to_rgb($lb);
stage("3. GD imagebmp -> RGB bytes", $last, $T0, strlen($rgb) . " B");

// ---------------------------------------------------------------- stage 4: fromBuffer
$yoloT = OnnxRuntime\Tensor::fromBuffer($rgb, [1, 3, YOLO_SIZE, YOLO_SIZE], 'uint8', [
    'layout' => 'hwc',
    'cast'   => 'float32',
    'scale'  => 1 / 255,
]);
stage("4. Tensor::fromBuffer (hwc, 1/255)", $last, $T0, json_encode($yoloT->shape()));

// ---------------------------------------------------------------- stage 5: YOLO session
$yoloPath = model_file($modelDir, 'yolo11n');
$yolo = new OnnxRuntime\Session($yoloPath, make_options($backend, $threads));
stage("5. YOLO Session create ({$backend})", $last, $T0, basename($yoloPath));

// ---------------------------------------------------------------- stage 6: YOLO run
$yoloIn = $yolo->inputNames()[0];
$t1 = hrtime(true);
$out = $yolo->run([$yoloIn => $yoloT]);
$last2 = $t1;
stage("6. YOLO run", $last2, $T0);
$last = $last2;

// ---------------------------------------------------------------- stage 7: postprocess
$y = is_array($out) ? reset($out) : $out;   // [1,84,8400]
$anchors = 8400;
$f = unpack('g*', $y->toBuffer());
$boxes = [];
for ($a = 0; $a < $anchors; $a++) {
    $cx = $f[0 * $anchors + $a + 1];
    $cy = $f[1 * $anchors + $a + 1];
    $bw = $f[2 * $anchors + $a + 1];
    $bh = $f[3 * $anchors + $a + 1];
    $bestV = 0.0; $bestC = -1;
    for ($c = 4; $c < 84; $c++) {
        $v = $f[$c * $anchors + $a + 1];
        if ($v > $bestV) { $bestV = $v; $bestC = $c - 4; }
    }
    if ($bestV < YOLO_CONF) {
        continue;
    }
    $x1 = ($cx - $bw / 2 - $padX) / $scale;
    $y1 = ($cy - $bh / 2 - $padY) / $scale;
    $x2 = ($cx + $bw / 2 - $padX) / $scale;
    $y2 = ($cy + $bh / 2 - $padY) / $scale;
    $boxes[] = [max(0.0, $x1), max(0.0, $y1), min((float) $w, $x2), min((float) $h, $y2), $bestV, $bestC];
}
$kept = nms($boxes, YOLO_IOU);
stage("7. YOLO decode + NMS", $last, $T0, count($boxes) . " raw -> " . count($kept) . " kept");

$coco = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
    'bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
    'carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
foreach ($kept as $i => $b) {
    printf("   det[%d] %-8s conf=%.2f  box=(%d,%d)-(%d,%d)\n",
        $i, $coco[$b[5]] ?? (string) $b[5], $b[4], $b[0], $b[1], $b[2], $b[3]);
    if ($i >= 5) { break; }
}

// ---------------------------------------------------------------- stage 8: ROI crop
$roi = null;
foreach ($kept as $b) {
    if ($b[5] === 5) {          // bus: the demo image has a destination sign
        $roi = $b;
        break;
    }
}
$roi ??= $kept[0] ?? [0, 0, $w, $h, 0.0, -1];
$x1 = (int) max(0, $roi[0] - 6);
$y1 = (int) max(0, $roi[1] - 6);
$x2 = (int) min($w, $roi[2] + 6);
$y2 = (int) min($h, $roi[3] + 6);
$cw = $x2 - $x1;
$chh = $y2 - $y1;
// Upscaling past ~2x adds no information for the rec model, it only costs
// det time quadratically: a 250x181 ROI on a 960 canvas is a 4x upscale.
$r = min($detLimit / $cw, $detLimit / $chh, $detScale);
$dw = max(32, (int) round($cw * $r / 32) * 32);
$dh = max(32, (int) round($chh * $r / 32) * 32);
$roiIm = crop_resize($im, $x1, $y1, $x2, $y2, $dw, $dh);
stage("8. GD crop ROI + resize", $last, $T0, "{$cw}x{$chh} -> {$dw}x{$dh}");

// ---------------------------------------------------------------- stage 9: det
$detBytes = gd_to_rgb($roiIm);
$detT = OnnxRuntime\Tensor::fromBuffer($detBytes, [1, 3, $dh, $dw], 'uint8', [
    'layout' => 'hwc',
    'cast'   => 'float32',
    'scale'  => 1 / 255,
    'mean'   => [0.485, 0.456, 0.406],
    'std'    => [0.229, 0.224, 0.225],
]);
$detPath = model_file($modelDir, 'ch_PP-OCRv4_det_infer');
$det = new OnnxRuntime\Session($detPath, make_options($backend, $threads));
stage("9. det fromBuffer + Session", $last, $T0, basename($detPath));

$t = hrtime(true);
$detIn = $det->inputNames()[0];
$detOut = $det->run([$detIn => $detT]);
$last2 = $t;
stage("10. det run", $last2, $T0);
$last = $last2;

// ---------------------------------------------------------------- stage 11: DB components
$do = is_array($detOut) ? reset($detOut) : $detOut;   // [1,1,H,W]
$ds = $do->shape();
$oh = $ds[2] ?? 0;
$ow = $ds[3] ?? 0;
$db = unpack('g*', $do->toBuffer());

// Threshold the probability map into one string per row, then walk 8-connected
// components. A real DB postprocess also does contour tracing, box scoring and
// unclip; for a demo picture one component per text line is enough and the
// whole pass stays under 150 ms in PHP.
$rows = [];
for ($yy = 0; $yy < $oh; $yy++) {
    $base = $yy * $ow;
    $r = '';
    for ($xx = 0; $xx < $ow; $xx++) {
        $r .= $db[$base + $xx + 1] >= DET_THRESH ? '1' : '0';
    }
    $rows[] = $r;
}
$comps = [];
for ($yy = 0; $yy < $oh; $yy++) {
    for ($xx = 0; $xx < $ow; $xx++) {
        if ($rows[$yy][$xx] !== '1') {
            continue;
        }
        $rows[$yy][$xx] = '0';
        $queue = [$yy * $ow + $xx];
        $qhead = 0;
        $area = 0;
        $minX = $xx; $maxX = $xx; $minY = $yy; $maxY = $yy;
        while ($qhead < count($queue)) {
            $code = $queue[$qhead++];
            $cy = intdiv($code, $ow);
            $cx = $code - $cy * $ow;
            $area++;
            if ($cx < $minX) { $minX = $cx; }
            if ($cx > $maxX) { $maxX = $cx; }
            if ($cy < $minY) { $minY = $cy; }
            if ($cy > $maxY) { $maxY = $cy; }
            for ($dy = -1; $dy <= 1; $dy++) {
                $ny = $cy + $dy;
                if ($ny < 0 || $ny >= $oh) { continue; }
                $nrow = &$rows[$ny];
                for ($dx = -1; $dx <= 1; $dx++) {
                    $nx = $cx + $dx;
                    if ($nx < 0 || $nx >= $ow) { continue; }
                    if ($nrow[$nx] === '1') {
                        $nrow[$nx] = '0';
                        $queue[] = $ny * $ow + $nx;
                    }
                }
                unset($nrow);
            }
        }
        if ($area >= 30 && $maxX - $minX >= 10 && $maxY - $minY >= 6) {
            $comps[] = [$minX, $minY, $maxX, $maxY, $area];
        }
    }
}
usort($comps, fn($a, $b) => $b[4] <=> $a[4]);
$comps = array_slice($comps, 0, 5);
if ($comps === []) {
    $comps[] = [0, 0, $ow - 1, $oh - 1, $ow * $oh];   // no text: try the whole ROI
}
stage("11. det toBuffer + DB components", $last, $T0, count($comps) . " text line(s)");

// ---------------------------------------------------------------- stage 12: rec session
$recPath = model_file($modelDir, 'ch_PP-OCRv4_rec_infer');
$rec = new OnnxRuntime\Session($recPath, make_options($backend, $threads));
stage("12. rec Session create ({$backend})", $last, $T0, basename($recPath));

// ---------------------------------------------------------------- stage 13/14: crop + rec
$sx = $cw / $ow;
$sy = $chh / $oh;
$dict = file($modelDir . '/ppocr_keys_v1.txt', FILE_IGNORE_NEW_LINES);
$lines = [];
$bufMs = 0.0;
$runMs = 0.0;
foreach ($comps as $c) {
    // Crop from the det-resolution image, not from the source: the ROI has
    // already been upscaled once, so the rec input is close to native size
    // instead of a 4x upscale of a 12px-tall text line.
    $padX = (int) max(4, round(0.06 * ($c[2] - $c[0])));
    $padY = (int) max(3, round(0.55 * ($c[3] - $c[1])));
    $tx1 = (int) max(0, $c[0] - $padX);
    $ty1 = (int) max(0, $c[1] - $padY);
    $tx2 = (int) min($dw, $c[2] + $padX);
    $ty2 = (int) min($dh, $c[3] + $padY);

    $t = hrtime(true);
    $tw = max(1, $tx2 - $tx1);
    $th_ = max(1, $ty2 - $ty1);
    $recW = (int) min(REC_MAX_W, max(32, round(REC_HEIGHT * $tw / $th_)));
    $recIm = crop_resize($roiIm, $tx1, $ty1, $tx2, $ty2, $recW, REC_HEIGHT);
    $recT = OnnxRuntime\Tensor::fromBuffer(gd_to_rgb($recIm), [1, 3, REC_HEIGHT, $recW], 'uint8', [
        'layout' => 'hwc',
        'cast'   => 'float32',
        'scale'  => 1 / 255,
        'mean'   => 0.5,
        'std'    => 0.5,
    ]);
    $bufMs += (hrtime(true) - $t) / 1e6;

    $t = hrtime(true);
    $recIn = $rec->inputNames()[0];
    $recOut = $rec->run([$recIn => $recT]);
    $firstRecT = $firstRecT ?? $recT;
    $runMs += (hrtime(true) - $t) / 1e6;

    $ro = is_array($recOut) ? reset($recOut) : $recOut;   // [1,T,C]
    $shape = $ro->shape();
    $steps = $shape[1];
    $recClasses = $shape[2];
    $text = ctc_decode($ro->toBuffer(), $steps, $recClasses, $dict);
    if ($text === '') {
        continue;
    }
    $lines[] = [
        'text'  => $text,
        'box'   => "{$tx1},{$ty1}-{$tx2},{$ty2}",
        'src'   => sprintf('%d,%d-%d,%d', $x1 + (int) ($tx1 * $sx), $y1 + (int) ($ty1 * $sy), $x1 + (int) ($tx2 * $sx), $y1 + (int) ($ty2 * $sy)),
        'in'    => "{$tw}x{$th_}->{$recW}x" . REC_HEIGHT,
        'steps' => $steps,
    ];
}
$now = hrtime(true);
printf("t=%9.2f ms   %8.2f ms   %-34s  %d line(s): crop+buffer %.2f ms, rec %.2f ms\n",
    ($now - $T0) / 1e6, ($now - $last) / 1e6, "13/14. per-line crop + rec", count($lines), $bufMs, $runMs);

printf("\nobjects    %s\n", implode(', ', array_map(
    fn($b) => ($coco[$b[5]] ?? $b[5]) . '@' . round($b[4], 2), array_slice($kept, 0, 5))));
printf("text\n");
foreach ($lines as $l) {
    printf("   %-16s  roi %-18s src %-18s %2d steps  %s\n",
        '"' . $l['text'] . '"', $l['box'], $l['src'], $l['steps'], $l['in']);
}
printf("\ntotal %.2f ms\n", (hrtime(true) - $T0) / 1e6);

// ---------------------------------------------------------------- warm runs (optional)
// REPS=5 gives steady-state numbers to compare backends with; the timeline
// above is single-shot by design, so it includes compile and first-run costs.
$reps = (int) (getenv('REPS') ?: 0);
if ($reps > 0 && isset($firstRecT)) {
    $median = function (callable $fn) use ($reps): float {
        $times = [];
        for ($i = 0; $i < $reps; $i++) {
            $t = hrtime(true);
            $fn();
            $times[] = (hrtime(true) - $t) / 1e6;
        }
        sort($times);
        return $times[intdiv($reps, 2)];
    };
    printf("\nwarm median over %d runs (same sessions, same inputs):\n", $reps);
    printf("   YOLO11n   %7.2f ms\n", $median(fn() => $yolo->run([$yoloIn => $yoloT])));
    printf("   OCR det   %7.2f ms\n", $median(fn() => $det->run([$detIn => $detT])));
    printf("   OCR rec   %7.2f ms\n", $median(fn() => $rec->run([$recIn => $firstRecT])));
}
