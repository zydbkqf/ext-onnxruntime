--TEST--
Session preserves multiple input and output names and order
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/multi_io.onnx');
echo 'inputs=', json_encode($s->inputNames()), "\n";
echo 'outputs=', json_encode($s->outputNames()), "\n";
echo 'inputTypes=', json_encode($s->inputTypes()), "\n";
echo 'outputTypes=', json_encode($s->outputTypes()), "\n";

$a = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
$b = OnnxRuntime\Tensor::fromArray([[10, 20, 30], [40, 50, 60]], 'float32');
$result = $s->run(['B' => $b, 'A' => $a]);

echo 'keys=', json_encode(array_keys($result)), "\n";
echo 'sum=', json_encode($result['sum']->toArray()), "\n";
echo 'diff=', json_encode($result['diff']->toArray()), "\n";
echo 'sumShape=', json_encode($result['sum']->shape()), "\n";
echo 'diffShape=', json_encode($result['diff']->shape()), "\n";
?>
--EXPECT--
inputs=["A","B"]
outputs=["sum","diff"]
inputTypes={"A":"float32","B":"float32"}
outputTypes={"sum":"float32","diff":"float32"}
keys=["sum","diff"]
sum=[[11,22,33],[44,55,66]]
diff=[[-9,-18,-27],[-36,-45,-54]]
sumShape=[2,3]
diffShape=[2,3]
