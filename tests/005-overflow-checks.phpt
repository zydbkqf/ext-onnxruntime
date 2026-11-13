--TEST--
Overflow and precision-loss checks for explicit dtypes
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function fail(callable $callback): void {
    try {
        $callback();
        echo "no-error\n";
    } catch (OnnxRuntime\ConversionException $e) {
        echo $e->getMessage(), "\n";
    }
}

fail(fn() => OnnxRuntime\Tensor::fromArray([128], 'int8'));
fail(fn() => OnnxRuntime\Tensor::fromArray([-1], 'uint8'));
fail(fn() => OnnxRuntime\Tensor::fromArray([300], 'uint8'));
fail(fn() => OnnxRuntime\Tensor::fromArray([32768], 'int16'));
fail(fn() => OnnxRuntime\Tensor::fromArray([2147483648], 'int32'));
fail(fn() => OnnxRuntime\Tensor::fromArray([1.5], 'int32'));
fail(fn() => OnnxRuntime\Tensor::fromArray([1e39], 'float32'));
fail(fn() => OnnxRuntime\Tensor::fromArray([NAN], 'int64'));
fail(fn() => OnnxRuntime\Tensor::fromArray([INF], 'int8'));
fail(fn() => OnnxRuntime\Tensor::fromArray([1e20], 'int64'));

// boundaries are accepted
$t = OnnxRuntime\Tensor::fromArray([128 - 1], 'int8');
echo $t->toArray()[0], "\n";
$t = OnnxRuntime\Tensor::fromArray([255], 'uint8');
echo $t->toArray()[0], "\n";
$t = OnnxRuntime\Tensor::fromArray([-9223372036854775807 - 1], 'int64');
echo $t->toArray()[0], "\n";

// float32 special values are preserved
$t = OnnxRuntime\Tensor::fromArray([NAN, INF, -INF], 'float32');
var_dump($t->toArray());
?>
--EXPECTF--
value[0]: value 128 is out of range for int8
value[0]: value -1 is out of range for uint8
value[0]: value 300 is out of range for uint8
value[0]: value 32768 is out of range for int16
value[0]: value 2147483648 is out of range for int32
value[0]: value 1.5 is not an integer and would lose precision when converted to an integer tensor
value[0]: value %s overflows float32
value[0]: non-finite float cannot be converted to an integer tensor
value[0]: non-finite float cannot be converted to an integer tensor
value[0]: value %s is out of int64 range
127
255
-9223372036854775808
array(3) {
  [0]=>
  float(NAN)
  [1]=>
  float(INF)
  [2]=>
  float(-INF)
}
