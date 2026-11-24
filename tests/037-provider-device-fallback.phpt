--TEST--
appendExecutionProvider resolves a name to an Env EpDevice when no typed appender exists
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
$r = OnnxRuntime\Runtime::default();
if (!$r->supports('ep_device')) {
    die('skip EP devices require ONNX Runtime 1.23+ (API 23)');
}
$has_cpu = false;
foreach ($r->getEpDevices() as $device) {
    if ($device->epName() === 'CPUExecutionProvider') {
        $has_cpu = true;
        break;
    }
}
if (!$has_cpu) die('skip CPUExecutionProvider EpDevice unavailable');
?>
--FILE--
<?php
// Lowercase "cpu" is not accepted by ORT's case-sensitive name-based
// appender, but it resolves to the enumerated CPU EpDevice.
$o = new OnnxRuntime\SessionOptions();
$o->appendExecutionProvider('cpu');
$snapshot = $o->toArray();
echo 'builtin-count=', count($snapshot['executionProviders']), "\n";
echo 'device-count=', count($snapshot['epDevices']), "\n";
echo 'name=', $snapshot['executionProviders'][0]['name'], "\n";

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $o);
$t = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
echo 'run=', json_encode($s->run(['X' => $t])['Y']->toArray()), "\n";

// Unknown names must still fail instead of silently selecting some device.
$bad = new OnnxRuntime\SessionOptions();
$bad->appendExecutionProvider('DefinitelyNotAnExecutionProvider');
try {
    new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $bad);
    echo 'unknown:no-error', "\n";
} catch (OnnxRuntime\Exception $e) {
    echo 'unknown:Exception', "\n";
}
?>
--EXPECT--
builtin-count=1
device-count=0
name=cpu
run=[[1,2,3],[4,5,6]]
unknown:Exception
