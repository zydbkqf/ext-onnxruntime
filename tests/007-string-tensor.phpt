--TEST--
String tensors keep their own payload representation
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ',
        json_encode($t->toArray()), "\n";
}

show(OnnxRuntime\Tensor::fromArray(['hello', 'world']));
show(OnnxRuntime\Tensor::fromArray([['a', 'bb'], ['ccc', 'dddd']]));
show(OnnxRuntime\Tensor::fromArray(['', 'x']));
show(OnnxRuntime\Tensor::fromArray([], 'string'));

// embedded NUL bytes survive the round-trip
$t = OnnxRuntime\Tensor::fromArray(["a\0b", "c"], 'string');
$out = $t->toArray();
echo bin2hex($out[0]), ' ', strlen($out[0]), ' ', bin2hex($out[1]), "\n";

// explicit string dtype with nested strings
show(OnnxRuntime\Tensor::fromArray([['x']], 'string'));
?>
--EXPECT--
string [2] ["hello","world"]
string [2,2] [["a","bb"],["ccc","dddd"]]
string [2] ["","x"]
string [0] []
610062 3 63
string [1,1] [["x"]]
