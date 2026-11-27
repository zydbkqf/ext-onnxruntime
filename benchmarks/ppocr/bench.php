<?php
declare(strict_types=1);
ini_set("memory_limit", "-1");
$work = getenv("W") ?: "/tmp/ort-img/work";
$meta = json_decode(file_get_contents($work . "/meta.json"), true);
$res = [];
function n(): int { return hrtime(true); }
function rep(string $k, int $ns): void { $v = $ns / 1e6; printf("%-40s %9.2f ms\n", $k, $v); $GLOBALS["res"][$k] = round($v, 2); }

// --- Stage 1: file read ---
$t = n(); $raw = file_get_contents($work . "/det_target.raw"); rep("1. file_get_contents 960x544 RGB", ($e = n()) - $t);

// --- Stage 2: unpack to array ---
$t = n(); $bytes = unpack("C*", $raw); rep("2. unpack(C*) 1.57M bytes", ($e = n()) - $t);

// --- Stage 3: build nested CHW float array in pure PHP ---
[$dw, $dh] = $meta["det"];
$t = n();
$c0 = []; $c1 = []; $c2 = [];
for ($y = 0; $y < $dh; $y++) {
    $r0 = []; $r1 = []; $r2 = [];
    $base = $y * $dw * 3;
    for ($x = 0; $x < $dw; $x++) {
        $i = $base + $x * 3 + 1;
        $r0[] = ($bytes[$i] / 255.0 - 0.5) / 0.5;
        $r1[] = ($bytes[$i + 1] / 255.0 - 0.5) / 0.5;
        $r2[] = ($bytes[$i + 2] / 255.0 - 0.5) / 0.5;
    }
    $c0[] = $r0; $c1[] = $r1; $c2[] = $r2;
}
rep("3. PHP nested CHW build (det 1.57M f32)", ($e = n()) - $t);

$t = n(); $nested = [[$c0, $c1, $c2]]; rep("4. wrap into [1][3][H][W]", ($e = n()) - $t);

// --- Stage 5: fromArray ---
$t = n(); $tensor = OnnxRuntime\Tensor::fromArray($nested, "float32"); rep("5. Tensor::fromArray (det)", ($e = n()) - $t);
printf("   tensor shape: %s dtype=%s\n", json_encode($tensor->shape()), $tensor->dtype());

// --- Stage 6: det inference ---
$t = n(); $so = new OnnxRuntime\SessionOptions(); $so->setIntraOpNumThreads(4);
$sess = new OnnxRuntime\Session(getenv("ORT_ROOT") ?: "/tmp/ort-img" , "/models/ch_PP-OCRv4_det_infer.onnx", $so);
rep("6. det session create", ($e = n()) - $t);
$names = $sess->inputNames();
printf("   det inputs: %s\n", json_encode(array_map(fn($x) => [$x, $sess->inputShapes()[$x] ?? null], $names)));
$inputs = [$names[0] => $tensor];
$t = n(); $out = $sess->run($inputs); rep("7. det run #1 (cold)", ($e = n()) - $t);
$runs = [];
for ($i = 0; $i < 5; $i++) { $t = n(); $sess->run($inputs); $runs[] = ($e = n()) - $t; }
rep("8. det run warm avg (5x)", (int)(array_sum($runs) / count($runs)));
$o = is_array($out) ? reset($out) : $out;
$arr = $o->toArray();
printf("   det out shape: %s\n", json_encode($o->shape()));

// --- Stage 9: rec ---
[$cw, $ch2] = $meta["rec_crop"]; [$rw, $rh] = $meta["rec_target"];
$raw2 = file_get_contents($work . "/rec_target.raw");
$t = n(); $b2 = unpack("C*", $raw2); rep("9. rec unpack (57k bytes)", ($e = n()) - $t);
$t = n();
$p0 = [[]]; $p1 = [[]]; $p2 = [[]];
for ($y = 0; $y < $rh; $y++) { $r0 = []; $r1 = []; $r2 = [];
    $base = $y * $rw * 3;
    for ($x = 0; $x < $rw; $x++) { $i = $base + $x * 3 + 1;
        $r0[] = ($b2[$i] / 255.0 - 0.5) / 0.5;
        $r1[] = ($b2[$i + 1] / 255.0 - 0.5) / 0.5;
        $r2[] = ($b2[$i + 2] / 255.0 - 0.5) / 0.5;
    }
    $p0[0][] = $r0; $p1[0][] = $r1; $p2[0][] = $r2;
}
$nested2 = [[$p0[0], $p1[0], $p2[0]]];
rep("10. rec nested CHW build (57k f32)", ($e = n()) - $t);
$t = n(); $tensor2 = OnnxRuntime\Tensor::fromArray($nested2, "float32"); rep("11. rec fromArray", ($e = n()) - $t);
$t = n(); $sess2 = new OnnxRuntime\Session(getenv("ORT_ROOT") ?: "/tmp/ort-img" , "/models/ch_PP-OCRv4_rec_infer.onnx", $so);
rep("12. rec session create", ($e = n()) - $t);
$n2 = $sess2->inputNames(); $in2 = [$n2[0] => $tensor2];
$t = n(); $out2 = $sess2->run($in2); rep("13. rec run #1 (cold)", ($e = n()) - $t);
$runs2 = [];
for ($i = 0; $i < 10; $i++) { $t = n(); $sess2->run($in2); $runs2[] = ($e = n()) - $t; }
rep("14. rec run warm avg (10x)", (int)(array_sum($runs2) / count($runs2)));

// --- Stage 15: rec decode (toArray + CTC) ---
$t = n();
$o2 = is_array($out2) ? reset($out2) : $out2;
$logits = $o2->toArray();
$dict = file($work . "/../ppocr_keys_v1.txt", FILE_IGNORE_NEW_LINES);
$blank = 0; $prev = -1; $text = "";
foreach ($logits[0] as $step) {
    $best = 0; $bv = $step[0];
    $n = count($step);
    for ($k = 1; $k < $n; $k++) { if ($step[$k] > $bv) { $bv = $step[$k]; $best = $k; } }
    if ($best !== $prev && $best !== $blank) {
        $text .= ($best - 1 < count($dict)) ? $dict[$best - 1] : " ";
    }
    $prev = $best;
}
rep("15. rec toArray + CTC decode", ($e = n()) - $t);
printf("   rec out shape: %s\n   OCR text: \"%s\"\n", json_encode($o2->shape()), $text);

file_put_contents($work . "/bench.json", json_encode(["metrics" => $res, "text" => $text, "det_shape" => $o->shape(), "rec_shape" => $o2->shape()], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
