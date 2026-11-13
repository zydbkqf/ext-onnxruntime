--TEST--
Session loads a real ONNX model, exposes tensor metadata and rejects bad model files
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
echo 'inputs=', json_encode($s->inputNames()), "\n";
echo 'outputs=', json_encode($s->outputNames()), "\n";
echo 'inputTypes=', json_encode($s->inputTypes()), "\n";
echo 'outputTypes=', json_encode($s->outputTypes()), "\n";
echo 'inputShapes=', json_encode($s->inputShapes()), "\n";
echo 'outputShapes=', json_encode($s->outputShapes()), "\n";

$d = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
echo 'dynamicInput=', json_encode($d->inputShapes()), "\n";
echo 'dynamicOutput=', json_encode($d->outputShapes()), "\n";

$reflection = new ReflectionClass(OnnxRuntime\Session::class);
echo 'final=', $reflection->isFinal() ? 'yes' : 'no', "\n";

try {
    clone $s;
    echo "clone:no-error\n";
} catch (Error $e) {
    echo "clone:rejected\n";
}

try {
    new OnnxRuntime\Session(__DIR__ . '/fixtures/does-not-exist.onnx');
    echo "path:no-error\n";
} catch (OnnxRuntime\Exception $e) {
    echo "path:exception\n";
}

try {
    new OnnxRuntime\Session(__DIR__ . '/fixtures/corrupt.onnx');
    echo "corrupt:no-error\n";
} catch (OnnxRuntime\Exception $e) {
    echo "corrupt:exception\n";
}

try {
    new OnnxRuntime\Session();
    echo "args:no-error\n";
} catch (ArgumentCountError $e) {
    echo "args:rejected\n";
}
?>
--EXPECT--
inputs=["X"]
outputs=["Y"]
inputTypes={"X":"float32"}
outputTypes={"Y":"float32"}
inputShapes={"X":[2,3]}
outputShapes={"Y":[2,3]}
dynamicInput={"X":[null,3]}
dynamicOutput={"Y":[null,3]}
final=yes
clone:rejected
path:exception
corrupt:exception
args:rejected
