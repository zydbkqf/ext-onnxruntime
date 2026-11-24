--TEST--
Phase 7C/7D device tensor allocation and copy on the built-in CPU device
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
if (!OnnxRuntime\Runtime::default()->supports('ep_device')) {
    die('skip EP devices and device tensors require ONNX Runtime 1.23+ (API 23)');
}
?>
--FILE--
<?php
function cpu_device(): OnnxRuntime\EpDevice {
    foreach (OnnxRuntime\Runtime::default()->getEpDevices() as $device) {
        if ($device->epName() === 'CPUExecutionProvider') {
            return $device;
        }
    }
    throw new RuntimeException('no CPU EP device');
}

function fail(callable $fn): void {
    try {
        $fn();
        echo "BAD: no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), ': ', $e->getMessage(), "\n";
    }
}

$cpu = cpu_device();
$t = OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'float32', $cpu);
echo $t->dtype(), ' ', json_encode($t->shape()), ' ', $t->rank(), ' ',
    $t->memoryType(), ' ', $t->device()->epName(), "\n";

$src = OnnxRuntime\Tensor::fromArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
$t->copyFrom($src);
unset($src); // device tensor must not borrow the PHP Tensor buffer
echo json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

$dst = OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'float32', $cpu);
$dst->copyFromDevice($t);
echo json_encode($dst->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

$empty = OnnxRuntime\Tensor::allocateOnDevice([0], 'float32', $cpu);
echo json_encode($empty->shape()), ' ', json_encode($empty->toArray()), "\n";

fail(fn() => OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'string', $cpu));
fail(fn() => OnnxRuntime\Tensor::allocateOnDevice([-1], 'float32', $cpu));
fail(fn() => OnnxRuntime\Tensor::allocateOnDevice(['x' => 1], 'float32', $cpu));
fail(fn() => OnnxRuntime\Tensor::allocateOnDevice([2], 'float32', $cpu, 'gpu'));
fail(fn() => $t->copyFrom(OnnxRuntime\Tensor::fromArray([1.0], 'float32')));
fail(fn() => $t->copyFrom(OnnxRuntime\Tensor::fromArray([[1, 2], [3, 4]], 'int64')));
fail(fn() => new OnnxRuntime\DeviceTensor());
fail(function () use ($t) { clone $t; });

echo "ok\n";
?>
--EXPECTF--
float32 [2,2] 2 device CPUExecutionProvider
[[1.5,2.5],[3.5,4.5]]
[[1.5,2.5],[3.5,4.5]]
[0] []
OnnxRuntime\ConversionException: device tensors support numeric element types only
OnnxRuntime\ConversionException: device tensor dimensions must be non-negative
OnnxRuntime\ConversionException: shape must be a zero-based list of integers
OnnxRuntime\ConversionException: memoryType must be "device" or "host_accessible"
OnnxRuntime\ConversionException: source tensor type/shape does not match the device tensor
OnnxRuntime\ConversionException: source tensor type/shape does not match the device tensor
Error: OnnxRuntime\DeviceTensor cannot be constructed directly; use OnnxRuntime\Tensor::allocateOnDevice()
Error: OnnxRuntime\DeviceTensor is immutable and cannot be cloned
ok
