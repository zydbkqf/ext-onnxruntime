--TEST--
Float tensors allow precision loss but not overflow
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
// float32 cannot represent 16777217 exactly; this is accepted by policy.
$t = OnnxRuntime\Tensor::fromArray([16777217], 'float32');
echo $t->dtype(), ' ', json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

// int64 -> double beyond 2^53 also loses precision; accepted.
$t = OnnxRuntime\Tensor::fromArray([9007199254740993], 'float64');
echo $t->dtype(), ' ', json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

// Integer tensors do not accept precision loss.
try {
    OnnxRuntime\Tensor::fromArray([1.5], 'int32');
    echo "no-error\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo "int-rejected\n";
}

// Overflow is still an error, not a silent infinity.
try {
    OnnxRuntime\Tensor::fromArray([1e39], 'float32');
    echo "no-error\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo "overflow-rejected\n";
}
?>
--EXPECT--
float32 [16777216.0]
float64 [9007199254740992.0]
int-rejected
overflow-rejected
