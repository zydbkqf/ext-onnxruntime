--TEST--
Phase 7F async I/O binding: Task owns binding, buffers stay alive, results are built on the PHP thread
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
unset($t);

$task = $s->runWithBindingAsync($b);
echo 'class=', get_class($task), "\n";

try {
    $b->bindInput('X', OnnxRuntime\Tensor::fromArray([[9.0]], 'float32'));
    echo "bind-after-submit=none\n";
} catch (Throwable $e) {
    echo 'bind-after-submit=', get_class($e), ':', $e->getMessage(), "\n";
}

try {
    $b->getOutputs();
    echo "get-after-submit=none\n";
} catch (Throwable $e) {
    echo 'get-after-submit=', get_class($e), ':', $e->getMessage(), "\n";
}

try {
    $s->runWithBinding($b);
    echo "sync-after-submit=none\n";
} catch (Throwable $e) {
    echo 'sync-after-submit=', get_class($e), ':', $e->getMessage(), "\n";
}

echo 'wait=', $task->wait(5000) ? '1' : '0', "\n";
echo 'state=', $task->getState(), "\n";
$r = $task->getResult();
echo 'out=', json_encode($r['Y']->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

// The Task retains the binding after the PHP IoBinding object is destroyed.
$b2 = $s->createIoBinding();
$b2->bindInput('X', OnnxRuntime\Tensor::fromArray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], 'float32'));
$b2->bindOutput('Y');
$task2 = $s->runWithBindingAsync($b2);
unset($b2);
echo 'wait2=', $task2->wait(5000) ? '1' : '0', "\n";
echo 'out2=', json_encode($task2->getResult()['Y']->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

// A binding cannot be submitted twice.
$b3 = $s->createIoBinding();
$b3->bindInput('X', OnnxRuntime\Tensor::fromArray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], 'float32'));
$b3->bindOutput('Y');
$task3 = $s->runWithBindingAsync($b3);
try {
    $s->runWithBindingAsync($b3);
    echo "resubmit=none\n";
} catch (Throwable $e) {
    echo 'resubmit=', get_class($e), ':', $e->getMessage(), "\n";
}
$task3->wait(5000);

// Cross-session binding is rejected before submission.
$b4 = $s->createIoBinding();
$b4->bindInput('X', OnnxRuntime\Tensor::fromArray([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], 'float32'));
$b4->bindOutput('Y');
$s2 = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
try {
    $s2->runWithBindingAsync($b4);
    echo "cross-session=none\n";
} catch (Throwable $e) {
    echo 'cross-session=', get_class($e), ':', $e->getMessage(), "\n";
}

// A worker failure is reported through the Task, not through the PHP thread.
$b5 = $s->createIoBinding();
$b5->bindInput('X', OnnxRuntime\Tensor::fromArray([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], 'float32'));
$task5 = $s->runWithBindingAsync($b5);
$task5->wait(5000);
try {
    $task5->getResult();
    echo "failed-task=none\n";
} catch (Throwable $e) {
    echo 'failed-task=', get_class($e), ':', $e->getMessage(), "\n";
}
?>
--EXPECT--
class=OnnxRuntime\Task
bind-after-submit=OnnxRuntime\ConversionException:I/O binding is owned by an async Task; read outputs with Task::getResult() and create a new binding for another run
get-after-submit=OnnxRuntime\ConversionException:I/O binding outputs are owned by an async Task; use Task::getResult() after Task::wait()
sync-after-submit=OnnxRuntime\ConversionException:I/O binding is owned by an async Task; read outputs with Task::getResult() and create a new binding for another run
wait=1
state=completed
out=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
wait2=1
out2=[[7.0,8.0,9.0],[10.0,11.0,12.0]]
resubmit=OnnxRuntime\ConversionException:I/O binding has already been submitted to an async Task
cross-session=OnnxRuntime\ConversionException:I/O binding belongs to a different Session
failed-task=OnnxRuntime\ConversionException:output "Y" is not bound
