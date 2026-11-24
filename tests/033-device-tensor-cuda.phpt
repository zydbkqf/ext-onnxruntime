--TEST--
Phase 7C/7D device tensor allocation and copies on a CUDA Plugin EP device
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
if (!OnnxRuntime\Runtime::default()->supports('ep_device')) {
    die('skip EP devices and device tensors require ONNX Runtime 1.23+ (API 23)');
}
if (PHP_OS_FAMILY !== 'Windows') die('skip CUDA Plugin EP test is Windows-only');
$dll = getenv('ONNXRUNTIME_CUDA_PLUGIN');
if ($dll === false || $dll === '') {
    $dll = 'D:\\php\\onnxruntime_providers_cuda.dll';
}
if (!is_file($dll)) die('skip CUDA Plugin EP DLL not found: ' . $dll);
?>
--FILE--
<?php
$rt = OnnxRuntime\Runtime::default();
$dll = getenv('ONNXRUNTIME_CUDA_PLUGIN');
if ($dll === false || $dll === '') {
    $dll = 'D:\\php\\onnxruntime_providers_cuda.dll';
}
$rt->registerExecutionProviderLibrary('cuda', $dll);

$cuda = null;
foreach ($rt->getEpDevices() as $device) {
    if ($device->epName() === 'CUDAExecutionProvider') {
        $cuda = $device;
        break;
    }
}
if ($cuda === null) {
    fwrite(STDERR, "no CUDA EP device after registration\n");
    exit(2);
}
echo "device=", $cuda->epName(), "\n";

$t = OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'float32', $cuda);
echo $t->dtype(), ' ', json_encode($t->shape()), ' ', $t->memoryType(), "\n";

$source = OnnxRuntime\Tensor::fromArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
$t->copyFrom($source);
unset($source);
echo json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

$dst = OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'float32', $cuda);
$dst->copyFromDevice($t);
echo json_encode($dst->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

try {
    $rt->unregisterExecutionProviderLibrary('cuda');
    echo "BAD: unregister succeeded while DeviceTensor alive\n";
} catch (Throwable $e) {
    echo "lease-held: ", get_class($e), "\n";
}

unset($t, $dst);
$rt->unregisterExecutionProviderLibrary('cuda');
echo "ok\n";
?>
--EXPECT--
device=CUDAExecutionProvider
float32 [2,2] device
[[1.5,2.5],[3.5,4.5]]
[[1.5,2.5],[3.5,4.5]]
lease-held: OnnxRuntime\ConversionException
ok
