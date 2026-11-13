--TEST--
Empty tensors and integer boundary round-trips
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ', $t->rank(), ' ',
        json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";
}

// zero-element tensors keep dtype and shape, with an empty payload
show(OnnxRuntime\Tensor::fromArray([], 'int64'));
show(OnnxRuntime\Tensor::fromArray([], 'bool'));
show(OnnxRuntime\Tensor::fromArray([[], []], 'int64'));
show(OnnxRuntime\Tensor::fromArray([[[]]], 'float32'));

// rank-0 and signed integer boundaries survive the round-trip
show(OnnxRuntime\Tensor::fromArray(9223372036854775807));
show(OnnxRuntime\Tensor::fromArray(-9223372036854775807 - 1));
show(OnnxRuntime\Tensor::fromArray([-128, 127], 'int8'));
show(OnnxRuntime\Tensor::fromArray([0, 255], 'uint8'));
show(OnnxRuntime\Tensor::fromArray([-32768, 32767], 'int16'));
show(OnnxRuntime\Tensor::fromArray([-2147483648, 2147483647], 'int32'));

// 3D string tensor and an empty 3D dimension
show(OnnxRuntime\Tensor::fromArray([[['a', 'b']]], 'string'));
show(OnnxRuntime\Tensor::fromArray([[[], []]], 'string'));
show(OnnxRuntime\Tensor::fromArray('x', 'string'));
?>
--EXPECT--
int64 [0] 1 []
bool [0] 1 []
int64 [2,0] 2 [[],[]]
float32 [1,1,0] 3 [[[]]]
int64 [] 0 9223372036854775807
int64 [] 0 -9223372036854775808
int8 [2] 1 [-128,127]
uint8 [2] 1 [0,255]
int16 [2] 1 [-32768,32767]
int32 [2] 1 [-2147483648,2147483647]
string [1,1,2] 3 [[["a","b"]]]
string [1,2,0] 3 [[[],[]]]
string [] 0 "x"
