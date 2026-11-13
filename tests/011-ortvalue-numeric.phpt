--TEST--
Tensor <-> OrtValue round-trip for all numeric dtypes, scalar and zero-element tensors
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ',
        json_encode($t->toArray()), "\n";
}

$cases = [
    ['float32', [[1.5, 2.5], [3.5, 4.5]]],
    ['float64', [[1.5, 2.5], [3.5, 4.5]]],
    ['int64', [[1, 2], [3, 4]]],
    ['int32', [[-1, 2], [3, 4]]],
    ['int16', [[-1, 2], [3, 4]]],
    ['int8', [[-1, 2], [3, 4]]],
    ['uint8', [[0, 2], [3, 255]]],
    ['bool', [[true, false], [false, true]]],
];
foreach ($cases as [$dtype, $data]) {
    $t = OnnxRuntime\Tensor::fromArray($data, $dtype);
    show($t->toOrtValue()->toTensor());
}

// Scalar (rank 0)
show(OnnxRuntime\Tensor::fromArray(42, 'int64')->toOrtValue()->toTensor());

// Zero-element tensors
show(OnnxRuntime\Tensor::fromArray([], 'float32')->toOrtValue()->toTensor());
show(OnnxRuntime\Tensor::fromArray([[], []], 'float32')->toOrtValue()->toTensor());

// Debug info of an OrtValue
$t = OnnxRuntime\Tensor::fromArray([[1, 2]], 'int32');
var_dump($t->toOrtValue()->__debugInfo());
?>
--EXPECT--
float32 [2,2] [[1.5,2.5],[3.5,4.5]]
float64 [2,2] [[1.5,2.5],[3.5,4.5]]
int64 [2,2] [[1,2],[3,4]]
int32 [2,2] [[-1,2],[3,4]]
int16 [2,2] [[-1,2],[3,4]]
int8 [2,2] [[-1,2],[3,4]]
uint8 [2,2] [[0,2],[3,255]]
bool [2,2] [[true,false],[false,true]]
int64 [] 42
float32 [0] []
float32 [2,0] [[],[]]
array(3) {
  ["dtype"]=>
  string(5) "int32"
  ["rank"]=>
  int(2)
  ["shape"]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
}
