--TEST--
Session rejects missing, unknown, non-Tensor, wrong-dtype and wrong-shape inputs
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$good = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');

$cases = [
    'missing' => [],
    'unknown' => ['Nope' => $good],
    'non-tensor' => ['X' => 123],
    'numeric-key' => [$good],
    'wrong-dtype' => ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'int64')],
    'wrong-shape' => ['X' => OnnxRuntime\Tensor::fromArray([1, 2, 3], 'float32')],
];
foreach ($cases as $label => $inputs) {
    try {
        $s->run($inputs);
        echo $label, ':no-error', "\n";
    } catch (Throwable $e) {
        echo $label, ':', get_class($e), "\n";
    }
}

try {
    $s->run('not-an-array');
    echo "arg:no-error\n";
} catch (TypeError $e) {
    echo "arg:TypeError\n";
}

// The same Session remains usable after rejected calls.
echo 'recovery=', json_encode($s->run(['X' => $good])['Y']->toArray()), "\n";
?>
--EXPECT--
missing:OnnxRuntime\ConversionException
unknown:OnnxRuntime\ConversionException
non-tensor:OnnxRuntime\ConversionException
numeric-key:OnnxRuntime\ConversionException
wrong-dtype:OnnxRuntime\ConversionException
wrong-shape:OnnxRuntime\ConversionException
arg:TypeError
recovery=[[1,2,3],[4,5,6]]
