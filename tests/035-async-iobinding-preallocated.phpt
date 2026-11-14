--TEST--
Phase 7F async I/O binding with a preallocated output buffer owned by the Task
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$b = $s->createIoBinding();
$b->bindInput('X', OnnxRuntime\Tensor::fromArray(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'float32'));

$buffer = OnnxRuntime\Tensor::fromArray(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 'float32');
$b->bindOutputBuffer('Y', $buffer);
unset($buffer);

$task = $s->runWithBindingAsync($b);
echo 'wait=', $task->wait(5000) ? '1' : '0', "\n";
echo 'out=', json_encode($task->getResult()['Y']->toArray(),
                           JSON_PRESERVE_ZERO_FRACTION), "\n";

// The binding is Task-owned; synchronous output access is rejected.
try {
    $b->getOutputs();
    echo "get=none\n";
} catch (Throwable $e) {
    echo 'get=', get_class($e), "\n";
}

// Queued/running binding tasks must not share buffers or Session state.
$tasks = [];
for ($i = 0; $i < 4; ++$i) {
    $bi = $s->createIoBinding();
    $v = (float) ($i + 1);
    $bi->bindInput('X', OnnxRuntime\Tensor::fromArray(
        [[$v, $v, $v], [$v, $v, $v]], 'float32'));
    $bi->bindOutput('Y');
    $tasks[] = $s->runWithBindingAsync($bi);
}
foreach ($tasks as $i => $t) {
    $t->wait(5000);
    $v = (float) ($i + 1);
    echo 't', $i, '=', json_encode($t->getResult()['Y']->toArray(),
                                   JSON_PRESERVE_ZERO_FRACTION), "\n";
}
?>
--EXPECT--
wait=1
out=[[1.0,2.0,3.0],[4.0,5.0,6.0]]
get=OnnxRuntime\ConversionException
t0=[[1.0,1.0,1.0],[1.0,1.0,1.0]]
t1=[[2.0,2.0,2.0],[2.0,2.0,2.0]]
t2=[[3.0,3.0,3.0],[3.0,3.0,3.0]]
t3=[[4.0,4.0,4.0],[4.0,4.0,4.0]]
