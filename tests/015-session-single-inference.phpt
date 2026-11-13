--TEST--
Session runs a real single-input model and returns an owned PHP Tensor
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], 'float32');

echo 'keys=', json_encode(array_keys($result = $s->run(['X' => $t]))), "\n";
echo 'dtype=', $result['Y']->dtype(), "\n";
echo 'shape=', json_encode($result['Y']->shape()), "\n";
echo 'output=', json_encode($result['Y']->toArray()), "\n";
echo 'input-unchanged=', json_encode($t->toArray()), "\n";
?>
--EXPECT--
keys=["Y"]
dtype=float32
shape=[2,3]
output=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
input-unchanged=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
