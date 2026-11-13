--TEST--
Large arrays and repeated conversions do not accumulate PHP memory
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$rows = [];
for ($i = 0; $i < 500; $i++) {
    $row = [];
    for ($j = 0; $j < 500; $j++) {
        $row[] = $i * 500 + $j;
    }
    $rows[] = $row;
}
$t = OnnxRuntime\Tensor::fromArray($rows);
$sum = 0;
foreach ($t->toArray() as $row) {
    $sum += array_sum($row);
}
echo $t->dtype(), ' ', json_encode($t->shape()), ' ', $sum, "\n";
unset($t, $rows, $row, $sum);

// A larger string tensor exercises the per-element string path too; the
// numeric payload test above does not cover std::string allocation.
$strings = [];
for ($i = 0; $i < 2000; $i++) {
    $strings[] = str_repeat(chr(65 + $i % 26), 127) . "\0";
}
$st = OnnxRuntime\Tensor::fromArray($strings, 'string');
$roundTrip = $st->toArray();
if ($roundTrip !== $strings) {
    echo "string-roundtrip-failed\n";
    exit(1);
}
echo $st->dtype(), ' ', json_encode($st->shape()), ' ',
     strlen($roundTrip[0]), ' ', strlen($roundTrip[1999]), "\n";
unset($st, $roundTrip, $strings);

$base = memory_get_usage();
for ($i = 0; $i < 3000; $i++) {
    $tensor = OnnxRuntime\Tensor::fromArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
    $clone = clone $tensor;
    unset($tensor, $clone);
}
for ($i = 0; $i < 1000; $i++) {
    try {
        OnnxRuntime\Tensor::fromArray([[1, 2], [3, 4, 5]]);
    } catch (OnnxRuntime\ConversionException $e) {
    }
}
gc_collect_cycles();
$delta = memory_get_usage() - $base;
// memory_get_usage() covers the Zend/PHP allocator only.  This assertion is
// a PHP-layer object lifecycle smoke test; native heap leaks are covered by
// tests/run-sanitizers.sh (ASan/LSan/Valgrind), not by this PHPT.
echo $delta < 1048576 ? "php-memory-stable\n" : "php-memory-growth:$delta\n";
?>
--EXPECT--
int64 [500,500] 31249875000
string [2000] 128 128
php-memory-stable
