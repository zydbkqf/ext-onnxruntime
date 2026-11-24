--TEST--
SessionOptions appendEpDevice uses ORT V2 device append and is separate from built-in EP append
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
$devices = $r->getEpDevices();
$cpu = null;
foreach ($devices as $device) {
    if ($device->epName() === 'CPUExecutionProvider') {
        $cpu = $device;
        break;
    }
}
if ($cpu === null) {
    die('skip CPUExecutionProvider EpDevice unavailable');
}

$o = new OnnxRuntime\SessionOptions();
$o->appendEpDevice($cpu);
$snapshot = $o->toArray();
echo 'builtin-count=', count($snapshot['executionProviders']), "\n";
echo 'device-count=', count($snapshot['epDevices']), "\n";
echo 'device-ep=', $snapshot['epDevices'][0]['device']['ep_name'], "\n";

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $o);
$t = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
echo 'run=', json_encode($s->run(['X' => $t])['Y']->toArray()), "\n";

// The same EpDevice snapshot and the same options snapshot must be reusable.
$s2 = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $o);
echo 'run2=', json_encode($s2->run(['X' => $t])['Y']->toArray()), "\n";
$o->setIntraOpNumThreads(2);
echo 'run2-after-option-mutation=', json_encode($s2->run(['X' => $t])['Y']->toArray()), "\n";

$builtin = new OnnxRuntime\SessionOptions();
$builtin->appendExecutionProvider('CoreML');
$snapshot = $builtin->toArray();
echo 'builtin-ep=', $snapshot['executionProviders'][0]['name'], "\n";
echo 'builtin-devices=', count($snapshot['epDevices']), "\n";

try {
    $o->appendEpDevice(new stdClass());
    echo 'wrong-object:no-error', "\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo 'wrong-object:ConversionException', "\n";
}
?>
--EXPECT--
builtin-count=0
device-count=1
device-ep=CPUExecutionProvider
run=[[1,2,3],[4,5,6]]
run2=[[1,2,3],[4,5,6]]
run2-after-option-mutation=[[1,2,3],[4,5,6]]
builtin-ep=CoreML
builtin-devices=0
wrong-object:ConversionException
