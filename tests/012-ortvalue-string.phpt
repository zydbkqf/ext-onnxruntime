--TEST--
String Tensor <-> OrtValue: empty, normal, embedded NUL, scalar, multi-dimensional, empty tensor
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$t = OnnxRuntime\Tensor::fromArray(["a\0b", '', 'c'], 'string');
$r = $t->toOrtValue()->toTensor();
echo $r->dtype(), ' ', json_encode($r->shape()), "\n";
foreach ($r->toArray() as $s) {
    echo strlen($s), ':', bin2hex($s), "\n";
}

$t = OnnxRuntime\Tensor::fromArray([['a', 'b'], ['c', 'd']], 'string');
$r = $t->toOrtValue()->toTensor();
echo $r->dtype(), ' ', json_encode($r->shape()), ' ',
    json_encode($r->toArray()), "\n";

// rank-0 string
$r = OnnxRuntime\Tensor::fromArray('x', 'string')->toOrtValue()->toTensor();
echo $r->dtype(), ' ', json_encode($r->shape()), ' ', json_encode($r->toArray()), "\n";

// empty string tensor
$r = OnnxRuntime\Tensor::fromArray([], 'string')->toOrtValue()->toTensor();
echo $r->dtype(), ' ', json_encode($r->shape()), ' ', json_encode($r->toArray()), "\n";
?>
--EXPECT--
string [3]
3:610062
0:
1:63
string [2,2] [["a","b"],["c","d"]]
string [] "x"
string [0] []
