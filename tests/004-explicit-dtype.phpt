--TEST--
Explicit dtype conversion and dtype name resolution
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ',
        json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";
}

show(OnnxRuntime\Tensor::fromArray([1, 2, 3], 'float32'));
show(OnnxRuntime\Tensor::fromArray([1.0, 2.0], 'int8'));
show(OnnxRuntime\Tensor::fromArray([0, 1], 'bool'));
show(OnnxRuntime\Tensor::fromArray([1.0, 2.0], 'uint8'));
show(OnnxRuntime\Tensor::fromArray([300], 'int16'));
show(OnnxRuntime\Tensor::fromArray([9223372036854775807], 'int64'));

// dtype names are case-insensitive and accept aliases
show(OnnxRuntime\Tensor::fromArray([1], 'FLOAT'));
show(OnnxRuntime\Tensor::fromArray([1.5], 'double'));

// named arguments exercise the arginfo
show(OnnxRuntime\Tensor::fromArray(value: [1.5, 2.5], dtype: 'float32'));

// strings cannot be coerced to numbers
try {
    OnnxRuntime\Tensor::fromArray(['1', '2'], 'int64');
    echo "no-error\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo $e->getMessage(), "\n";
}

// numbers cannot be silently stringified
try {
    OnnxRuntime\Tensor::fromArray([1], 'string');
    echo "no-error\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo $e->getMessage(), "\n";
}

// unknown dtype
try {
    OnnxRuntime\Tensor::fromArray([1], 'float16');
    echo "no-error\n";
} catch (ValueError $e) {
    echo get_class($e), ': ', $e->getMessage(), "\n";
}
?>
--EXPECT--
float32 [3] [1.0,2.0,3.0]
int8 [2] [1,2]
bool [2] [false,true]
uint8 [2] [1,2]
int16 [1] [300]
int64 [1] [9223372036854775807]
float32 [1] [1.0]
float64 [1] [1.5]
float32 [2] [1.5,2.5]
value[0]: cannot convert string to an integer tensor element
value[0]: cannot convert int to a string tensor element; values must already be PHP strings
ValueError: Unsupported dtype "float16"; supported dtypes are: float32, float64, int64, int32, int16, int8, uint8, bool, string
