--TEST--
Runtime exposes ORT version/build info, available providers, and immutable EP devices
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
if (!OnnxRuntime\Runtime::default()->supports('ep_device')) {
    die('skip EP devices and device tensors require ONNX Runtime 1.23+ (API 23)');
}
?>
--FILE--
<?php
$r = OnnxRuntime\Runtime::default();

echo 'version=', $r->getVersion(), "\n";
echo 'build-info=', strlen($r->getBuildInfo()) > 0 ? 'nonempty' : 'empty', "\n";

$providers = $r->getAvailableProviders();
echo 'providers-array=', is_array($providers) ? 'yes' : 'no', "\n";
echo 'cpu-provider=', in_array('CPUExecutionProvider', $providers, true) ? 'yes' : 'no', "\n";

$devices = $r->getEpDevices();
echo 'devices-array=', is_array($devices) ? 'yes' : 'no', "\n";
echo 'device-count=', count($devices) > 0 ? 'positive' : 'zero', "\n";
$first = $devices[0];
echo 'device-object=', $first instanceof OnnxRuntime\EpDevice ? 'yes' : 'no', "\n";
$fields = ['ep_name', 'ep_vendor', 'device_type', 'vendor_id', 'device_id', 'device_vendor', 'ep_metadata', 'ep_options', 'is_plugin', 'registration_name'];
echo 'device-fields=', !array_diff($fields, array_keys($first->toArray())) ? 'yes' : 'no', "\n";
echo 'device-type=', in_array($first->deviceType(), ['cpu', 'gpu', 'npu'], true) ? 'yes' : 'no', "\n";
echo 'plugin-flag=', is_bool($first->isPlugin()) ? 'bool' : 'no', "\n";
echo 'registration=', $first->registrationName() === null ? 'null' : 'string', "\n";

try {
    new OnnxRuntime\Runtime();
    echo 'construct:no-error', "\n";
} catch (Throwable $e) {
    echo 'construct:', get_class($e), "\n";
}

try {
    clone $r;
    echo 'clone:no-error', "\n";
} catch (Throwable $e) {
    echo 'clone:', get_class($e), "\n";
}

try {
    new OnnxRuntime\EpDevice();
    echo 'epdevice-construct:no-error', "\n";
} catch (Throwable $e) {
    echo 'epdevice-construct:', get_class($e), "\n";
}

try {
    clone $first;
    echo 'epdevice-clone:no-error', "\n";
} catch (Throwable $e) {
    echo 'epdevice-clone:', get_class($e), "\n";
}

try {
    $r->registerExecutionProviderLibrary('missing-ep', '/definitely/not/a/plugin.so');
    echo 'bad-register:no-error', "\n";
} catch (OnnxRuntime\Exception $e) {
    echo 'bad-register:Exception', "\n";
}

try {
    $r->unregisterExecutionProviderLibrary('missing-ep');
    echo 'bad-unregister:no-error', "\n";
} catch (OnnxRuntime\Exception $e) {
    echo 'bad-unregister:Exception', "\n";
}
?>
--EXPECTF--
version=%s
build-info=nonempty
providers-array=yes
cpu-provider=yes
devices-array=yes
device-count=positive
device-object=yes
device-fields=yes
device-type=yes
plugin-flag=bool
registration=null
construct:Error
clone:Error
epdevice-construct:Error
epdevice-clone:Error
bad-register:Exception
bad-unregister:Exception
