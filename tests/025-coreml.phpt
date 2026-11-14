--TEST--
SessionOptions appends the CoreML execution provider and runs a model
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
if (PHP_OS_FAMILY !== 'Darwin') die('skip CoreML is macOS-only');
$r = OnnxRuntime\Runtime::default();
if (!in_array('CoreMLExecutionProvider', $r->getAvailableProviders(), true)) {
    die('skip CoreML execution provider unavailable');
}
?>
--FILE--
<?php
$o = new OnnxRuntime\SessionOptions();
$o->appendExecutionProvider('CoreML');

echo 'providers=', json_encode($o->toArray()['executionProviders']), "\n";

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/multi_io.onnx', $o);
$a = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
$b = OnnxRuntime\Tensor::fromArray([[6, 5, 4], [3, 2, 1]], 'float32');
$out = $s->run(['A' => $a, 'B' => $b]);

echo 'sum=', json_encode($out['sum']->toArray()), "\n";
echo 'diff=', json_encode($out['diff']->toArray()), "\n";
?>
--EXPECT--
providers=[{"name":"CoreML","options":[]}]
sum=[[7,7,7],[7,7,7]]
diff=[[-5,-3,-1],[1,3,5]]
