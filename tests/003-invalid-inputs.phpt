--TEST--
Invalid inputs are rejected with explicit ConversionException messages
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

// irregular sub-arrays are never padded
fail(fn() => OnnxRuntime\Tensor::fromArray([[1, 2], [3, 4, 5]]));

// scalar mixed into a nested array
fail(fn() => OnnxRuntime\Tensor::fromArray([1, [2]]));

// mixed bool and numeric values need an explicit dtype
fail(fn() => OnnxRuntime\Tensor::fromArray([true, 1]));

// string mixed with numeric values cannot be inferred
fail(fn() => OnnxRuntime\Tensor::fromArray(['a', 1]));

// null has no ONNX representation here
fail(fn() => OnnxRuntime\Tensor::fromArray([1, null]));

// associative keys
fail(fn() => OnnxRuntime\Tensor::fromArray(['a' => 1]));

// sparse numeric keys
fail(fn() => OnnxRuntime\Tensor::fromArray([0 => 1, 2 => 3]));

// unsupported PHP object
fail(fn() => OnnxRuntime\Tensor::fromArray([1, new stdClass()]));

// recursion guard
$nested = 1;
for ($i = 0; $i < 70; $i++) {
    $nested = [$nested];
}
try {
    OnnxRuntime\Tensor::fromArray($nested);
    echo "no-error\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo (strpos($e->getMessage(), 'array nesting exceeds 64 levels') !== false)
        ? "depth-guard\n" : "wrong-depth-message\n";
}
?>
--EXPECT--
value[1]: irregular nested array; expected shape [2], got [3]
value[1]: nested array mixes sub-arrays and scalar values
cannot infer a single tensor dtype: array mixes bool and numeric values; pass an explicit dtype
cannot infer a single tensor dtype: array mixes string and numeric/bool values; pass an explicit dtype
value[1]: unsupported PHP type "null" in tensor data (expected int, float, bool, string or array)
value: expected a list with sequential integer keys starting at 0; associative or sparse arrays are not supported
value: expected a list with sequential integer keys starting at 0; associative or sparse arrays are not supported
value[1]: unsupported PHP type "stdClass" in tensor data (expected int, float, bool, string or array)
depth-guard
