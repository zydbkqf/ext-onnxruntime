--TEST--
Basic shape inference and conversion
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ',
        json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), ' ',
        $t->rank(), "\n";
}

show(OnnxRuntime\Tensor::fromArray([1, 2, 3]));
show(OnnxRuntime\Tensor::fromArray([1.5, 2.5]));
show(OnnxRuntime\Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
show(OnnxRuntime\Tensor::fromArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]));
show(OnnxRuntime\Tensor::fromArray(42));
show(OnnxRuntime\Tensor::fromArray([]));
show(OnnxRuntime\Tensor::fromArray([[1], [2], [3]]));
?>
--EXPECT--
int64 [3] [1,2,3] 1
float64 [2] [1.5,2.5] 1
float64 [2,3] [[1.0,2.0,3.0],[4.0,5.0,6.0]] 2
int64 [2,2,2] [[[1,2],[3,4]],[[5,6],[7,8]]] 3
int64 [] 42 0
float32 [0] [] 1
int64 [3,1] [[1],[2],[3]] 2
