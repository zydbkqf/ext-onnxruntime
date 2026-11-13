--TEST--
Session validates input dtype/rank/static shape before Run and run() arginfo is accurate
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$good = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');

$cases = [
    'dtype' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'int64'),
    'rank' => OnnxRuntime\Tensor::fromArray([1, 2, 3], 'float32'),
    'shape' => OnnxRuntime\Tensor::fromArray([[1, 2], [3, 4]], 'float32'),
];

foreach ($cases as $label => $tensor) {
    try {
        $s->run(['X' => $tensor]);
        echo $label, ':no-error', "\n";
    } catch (Throwable $e) {
        $marker = str_contains($e->getMessage(), 'dtype mismatch') ? 'dtype mismatch'
            : (str_contains($e->getMessage(), 'rank mismatch') ? 'rank mismatch'
            : (str_contains($e->getMessage(), 'shape mismatch') ? 'shape mismatch'
            : 'unexpected-message'));
        echo $label, ':', get_class($e), ':', $marker, "\n";
    }
}

$reflection = new ReflectionMethod('OnnxRuntime\\Session', 'run');
echo 'run-required:', $reflection->getNumberOfRequiredParameters(), "\n";

// Dynamic model dimensions must remain free, while static dimensions still
// have to match.  The dynamic fixture has shape [? , 3].
$d = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
echo 'dynamic:', json_encode($d->run(['X' => $good])['Y']->toArray()), "\n";
?>
--EXPECT--
dtype:OnnxRuntime\ConversionException:dtype mismatch
rank:OnnxRuntime\ConversionException:rank mismatch
shape:OnnxRuntime\ConversionException:shape mismatch
run-required:1
dynamic:[[1,2,3],[4,5,6]]
