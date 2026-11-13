--TEST--
Session IoBinding supports CPU input/output binding and repeated runs
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$b = $s->createIoBinding();
$t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
                                   'float32');
$b->bindInput('X', $t);
$b->bindOutput('Y');

// The binding owns a deep copy of the input; the PHP Tensor may die now.
unset($t);

$r = $s->runWithBinding($b);
echo 'keys=', json_encode(array_keys($r)), "\n";
echo 'out=', json_encode($r['Y']->toArray()), "\n";
echo 'get=', json_encode($b->getOutputs()['Y']->toArray()), "\n";

$r2 = $s->runWithBinding($b);
echo 'out2=', json_encode($r2['Y']->toArray()), "\n";

$s2 = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
try {
    $s2->runWithBinding($b);
    echo "cross-session:none\n";
} catch (Throwable $e) {
    echo 'cross-session=', get_class($e), ':', $e->getMessage(), "\n";
}

try {
    new OnnxRuntime\IoBinding();
    echo "new:none\n";
} catch (Throwable $e) {
    echo 'new=', get_class($e), ':', $e->getMessage(), "\n";
}

try {
    clone $b;
    echo "clone:none\n";
} catch (Throwable $e) {
    echo 'clone=', get_class($e), ':', $e->getMessage(), "\n";
}
?>
--EXPECT--
keys=["Y"]
out=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
get=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
out2=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
cross-session=OnnxRuntime\ConversionException:I/O binding belongs to a different Session
new=Error:OnnxRuntime\IoBinding is created by Session::createIoBinding and cannot be constructed
clone=Error:OnnxRuntime\IoBinding cannot be cloned
