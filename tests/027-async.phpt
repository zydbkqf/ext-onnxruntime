--TEST--
Native async inference: runAsync, Task state/result, input and Session lifetime
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$r = OnnxRuntime\Runtime::default();
$r->setAsyncThreads(2);
$r->setAsyncQueueCapacity(8);
$info = $r->asyncInfo();
echo 'workers=', $info['workers'], " capacity=", $info['queueCapacity'], "\n";

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$input = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
$task = $s->runAsync(['X' => $input]);
unset($input); // worker owns a deep copy

echo 'class=', get_class($task), "\n";
echo 'state0=', in_array($task->getState(), ['queued', 'running'], true) ? 'pending' : 'bad', "\n";
echo 'wait=', $task->wait(5.0) ? 'true' : 'false', "\n";
echo 'ready=', $task->isReady() ? 'true' : 'false', "\n";
echo 'state1=', $task->getState(), "\n";
echo 'result=', json_encode($task->getResult()['Y']->toArray()), "\n";

$s2 = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$t1 = $s2->runAsync(['X' => OnnxRuntime\Tensor::fromArray([[7, 8, 9], [1, 1, 1]], 'float32')]);
$t2 = $s2->runAsync(['X' => OnnxRuntime\Tensor::fromArray([[2, 2, 2], [3, 3, 3]], 'float32')]);
$t1->wait(5.0);
$t2->wait(5.0);
echo 'multi=', json_encode($t1->getResult()['Y']->toArray()), '|', json_encode($t2->getResult()['Y']->toArray()), "\n";

// The Session PHP object may disappear while the Task holds native state.
$t3 = $s2->runAsync(['X' => OnnxRuntime\Tensor::fromArray([[4, 4, 4], [5, 5, 5]], 'float32')]);
unset($s2);
$t3->wait(5.0);
echo 'detached-session=', json_encode($t3->getResult()['Y']->toArray()), "\n";

echo 'cancel-completed=', $t3->cancel() ? 'true' : 'false', "\n";
echo 'error-null=', $t3->getError() === null ? 'true' : 'false', "\n";

try {
    new OnnxRuntime\Task();
    echo 'task-new:no-error', "\n";
} catch (Error $e) {
    echo 'task-new:Error', "\n";
}

try {
    clone $t3;
    echo 'task-clone:no-error', "\n";
} catch (Error $e) {
    echo 'task-clone:Error', "\n";
}

try {
    $s2 = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
    $s2->runAsync(['X' => OnnxRuntime\Tensor::fromArray([[1]], 'float32')]);
    echo 'bad-input:no-error', "\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo 'bad-input:ConversionException', "\n";
}
?>
--EXPECT--
workers=2 capacity=8
class=OnnxRuntime\Task
state0=pending
wait=true
ready=true
state1=completed
result=[[1,2,3],[4,5,6]]
multi=[[7,8,9],[1,1,1]]|[[2,2,2],[3,3,3]]
detached-session=[[4,4,4],[5,5,5]]
cancel-completed=false
error-null=true
task-new:Error
task-clone:Error
bad-input:ConversionException
