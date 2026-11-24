--TEST--
Name-resolved Plugin EP devices keep a lease until the Session is destroyed
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
$r = OnnxRuntime\Runtime::default();
if (!$r->supports('ep_device')) {
    die('skip EP devices require ONNX Runtime 1.23+ (API 23)');
}
$dll = getenv('ONNXRUNTIME_OPENVINO_PLUGIN');
if ($dll === false || $dll === '') {
    if (PHP_OS_FAMILY === 'Windows') {
        $dll = 'D:\\php\\openvino-ep\\onnxruntime_providers_openvino_plugin.dll';
    }
}
if ($dll === false || $dll === '' || !is_file($dll)) {
    die('skip set ONNXRUNTIME_OPENVINO_PLUGIN to the OpenVINO plugin library');
}
?>
--FILE--
<?php
$rt = OnnxRuntime\Runtime::default();
$dll = getenv('ONNXRUNTIME_OPENVINO_PLUGIN');
if ($dll === false || $dll === '') {
    if (PHP_OS_FAMILY === 'Windows') {
        $dll = 'D:\\php\\openvino-ep\\onnxruntime_providers_openvino_plugin.dll';
    }
}
$rt->registerExecutionProviderLibrary('openvino', $dll);

$o = new OnnxRuntime\SessionOptions();
// Name-based append: resolveExecutionProviderDevices() resolves this to a
// plugin EpDevice before ORT's generic name appender is called.
$o->appendExecutionProvider('OpenVINO');

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $o);
$x = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], 'float32');
$out = $s->run(['X' => $x]);
echo 'out=', json_encode($out['Y']->toArray()), "\n";

try {
    $rt->unregisterExecutionProviderLibrary('openvino');
    echo "BAD: unregister succeeded while Session alive\n";
} catch (Throwable $e) {
    echo 'lease-held: ', get_class($e), "\n";
}

unset($s, $o);
$rt->unregisterExecutionProviderLibrary('openvino');
echo "ok\n";
?>
--EXPECT--
out=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
lease-held: OnnxRuntime\ConversionException
ok
