--TEST--
Session accepts zero-element Tensors and lets ORT validate the model shape
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$one = new OnnxRuntime\Session(__DIR__ . '/fixtures/zero_1d.onnx');
$t1 = OnnxRuntime\Tensor::fromArray([], 'float32');
var_dump($t1->shape());
$o1 = $one->run(['X' => $t1]);
var_dump($o1['Y']->shape(), $o1['Y']->toArray());

$two = new OnnxRuntime\Session(__DIR__ . '/fixtures/zero_2d.onnx');
$t2 = OnnxRuntime\Tensor::fromArray([[], []], 'float32');
var_dump($t2->shape());
$o2 = $two->run(['X' => $t2]);
var_dump($o2['Y']->shape(), $o2['Y']->toArray());
echo "zero-ok\n";
?>
--EXPECT--
array(1) {
  [0]=>
  int(0)
}
array(1) {
  [0]=>
  int(0)
}
array(0) {
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(0)
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(0)
}
array(2) {
  [0]=>
  array(0) {
  }
  [1]=>
  array(0) {
  }
}
zero-ok
