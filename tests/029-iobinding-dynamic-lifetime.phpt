--TEST--
Session IoBinding handles dynamic and zero-element outputs, multiple I/O, errors and lifetime
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
$b = $s->createIoBinding();
$b->bindOutput('Y');
foreach ([
    [[1.5, 2.5, 3.5]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
] as $rows) {
    $b->bindInput('X', OnnxRuntime\Tensor::fromArray($rows, 'float32'));
    $out = $s->runWithBinding($b)['Y'];
    echo json_encode($out->shape()), ' ', json_encode($out->toArray()), "\n";
}

$z = new OnnxRuntime\Session(__DIR__ . '/fixtures/zero_1d.onnx');
$zb = $z->createIoBinding();
$zb->bindInput('X', OnnxRuntime\Tensor::fromArray([], 'float32'));
$zb->bindOutput('Y');
$zero = $z->runWithBinding($zb)['Y'];
echo 'zero-shape=', json_encode($zero->shape()), "\n";

$m = new OnnxRuntime\Session(__DIR__ . '/fixtures/multi_io.onnx');
$mb = $m->createIoBinding();
$mb->bindInput('A', OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                                  'float32'));
$mb->bindInput('B', OnnxRuntime\Tensor::fromArray([[10, 20, 30], [40, 50, 60]],
                                                  'float32'));
$mb->bindOutput('diff');
$mb->bindOutput('sum');
$mr = $m->runWithBinding($mb);
echo 'multi-keys=', json_encode(array_keys($mr)), "\n";
echo 'multi-sum=', json_encode($mr['sum']->toArray()), "\n";
echo 'multi-diff=', json_encode($mr['diff']->toArray()), "\n";

$e = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$eb = $e->createIoBinding();
try {
    $eb->getOutputs();
    echo "empty-get:none\n";
} catch (Throwable $ex) {
    echo 'empty-get=', get_class($ex), ':', $ex->getMessage(), "\n";
}
try {
    $eb->bindInput('Nope', OnnxRuntime\Tensor::fromArray([1.0], 'float32'));
    echo "unknown-in:none\n";
} catch (Throwable $ex) {
    echo 'unknown-in=', get_class($ex), ':', $ex->getMessage(), "\n";
}
try {
    $eb->bindOutput('Nope');
    echo "unknown-out:none\n";
} catch (Throwable $ex) {
    echo 'unknown-out=', get_class($ex), ':', $ex->getMessage(), "\n";
}
try {
    $e->runWithBinding($eb);
    echo "run-no-input:none\n";
} catch (Throwable $ex) {
    echo 'run-no-input=', get_class($ex), ':', $ex->getMessage(), "\n";
}
$eb->bindInput('X', OnnxRuntime\Tensor::fromArray([[7, 8, 9], [10, 11, 12]],
                                                  'float32'));
try {
    $e->runWithBinding($eb);
    echo "run-no-output:none\n";
} catch (Throwable $ex) {
    echo 'run-no-output=', get_class($ex), ':', $ex->getMessage(), "\n";
}
$eb->bindOutput('Y');
$er = $e->runWithBinding($eb);
unset($e);
echo 'after-session-destroy=', json_encode($eb->getOutputs()['Y']->toArray()),
     "\n";
?>
--EXPECT--
[1,3] [[1.5,2.5,3.5]]
[2,3] [[1,2,3],[4,5,6]]
zero-shape=[0]
multi-keys=["sum","diff"]
multi-sum=[[11,22,33],[44,55,66]]
multi-diff=[[-9,-18,-27],[-36,-45,-54]]
empty-get=OnnxRuntime\ConversionException:I/O binding has not run; call Session::runWithBinding() first
unknown-in=OnnxRuntime\ConversionException:unknown input "Nope"
unknown-out=OnnxRuntime\ConversionException:unknown output "Nope"
run-no-input=OnnxRuntime\ConversionException:input "X" is not bound
run-no-output=OnnxRuntime\ConversionException:output "Y" is not bound
after-session-destroy=[[7,8,9],[10,11,12]]
