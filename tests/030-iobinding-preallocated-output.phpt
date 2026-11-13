--TEST--
IoBinding preallocated output buffer handles fixed shapes and rejects dynamic outputs
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$b = $s->createIoBinding();
$b->bindInput('X', OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                                  'float32'));
$buffer = OnnxRuntime\Tensor::fromArray([[0, 0, 0], [0, 0, 0]], 'float32');
$b->bindOutputBuffer('Y', $buffer);

$r = $s->runWithBinding($b);
echo 'out=', json_encode($r['Y']->toArray()), "\n";
echo 'buffer-unchanged=', json_encode($buffer->toArray()), "\n";

// The binding owns its own copy; the PHP buffer may be destroyed now.
unset($buffer);
$r2 = $s->runWithBinding($b);
echo 'out2=', json_encode($r2['Y']->toArray()), "\n";

try {
    $b->bindOutputBuffer(
        'Y', OnnxRuntime\Tensor::fromArray([[0, 0, 0]], 'float32')
    );
    echo "bad-shape:none\n";
} catch (Throwable $e) {
    echo 'bad-shape=', get_class($e), ':', $e->getMessage(), "\n";
}
try {
    $b->bindOutputBuffer(
        'Y', OnnxRuntime\Tensor::fromArray([[0, 0, 0], [0, 0, 0]])
    );
    echo "bad-dtype:none\n";
} catch (Throwable $e) {
    echo 'bad-dtype=', get_class($e), ':', $e->getMessage(), "\n";
}
try {
    $b->bindOutputBuffer(
        'Nope', OnnxRuntime\Tensor::fromArray([[0, 0, 0], [0, 0, 0]],
                                               'float32')
    );
    echo "unknown-output:none\n";
} catch (Throwable $e) {
    echo 'unknown-output=', get_class($e), ':', $e->getMessage(), "\n";
}

// A rejected rebind must not disturb the previously valid preallocated output.
$r3 = $s->runWithBinding($b);
echo 'out3=', json_encode($r3['Y']->toArray()), "\n";

$d = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
$db = $d->createIoBinding();
$db->bindInput('X', OnnxRuntime\Tensor::fromArray([[1, 2, 3]], 'float32'));
try {
    $db->bindOutputBuffer(
        'Y', OnnxRuntime\Tensor::fromArray([[0, 0, 0]], 'float32')
    );
    echo "dynamic:none\n";
} catch (Throwable $e) {
    echo 'dynamic=', get_class($e), ':', $e->getMessage(), "\n";
}

$z = new OnnxRuntime\Session(__DIR__ . '/fixtures/zero_1d.onnx');
$zb = $z->createIoBinding();
$zb->bindInput('X', OnnxRuntime\Tensor::fromArray([], 'float32'));
$zb->bindOutputBuffer('Y', OnnxRuntime\Tensor::fromArray([], 'float32'));
$zero = $z->runWithBinding($zb)['Y'];
echo 'zero-shape=', json_encode($zero->shape()), "\n";
?>
--EXPECT--
out=[[1,2,3],[4,5,6]]
buffer-unchanged=[[0,0,0],[0,0,0]]
out2=[[1,2,3],[4,5,6]]
bad-shape=OnnxRuntime\ConversionException:output "Y" shape mismatch: model expects [2, 3], got [1, 3]
bad-dtype=OnnxRuntime\ConversionException:output "Y" dtype mismatch: model expects float32, got int64
unknown-output=OnnxRuntime\ConversionException:unknown output "Nope"
out3=[[1,2,3],[4,5,6]]
dynamic=OnnxRuntime\ConversionException:output "Y" has a dynamic shape and cannot use a preallocated buffer
zero-shape=[0]
