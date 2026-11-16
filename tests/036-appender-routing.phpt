--TEST--
appendExecutionProvider routes CUDA and TensorRT to their dedicated ORT appenders
--SKIPIF--
<?php
if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded');
$providers = OnnxRuntime\Runtime::default()->getAvailableProviders();
if (!in_array('CUDAExecutionProvider', $providers, true)
    && !in_array('TensorrtExecutionProvider', $providers, true)) {
    die('skip this ORT package has no CUDA or TensorRT provider');
}
?>
--FILE--
<?php
// ORT's generic name-based appender rejects 'CUDA' and 'TensorRT' outright
// ("Unknown provider name ..."). Only the dedicated _V2 appenders accept
// them, so the routing in SessionConfig::applyTo() has to survive whatever
// the local GPU runtime does: a missing provider DLL or an absent GPU is a
// legitimate runtime error, a name rejection is a regression.
foreach (['CUDA' => ['device_id' => '0'], 'TensorRT' => ['device_id' => '0']] as $name => $options) {
    $so = new OnnxRuntime\SessionOptions();
    try {
        $so->appendExecutionProvider($name, $options);
        echo $name, "=routed\n";
    } catch (Throwable $e) {
        $rejected = strpos($e->getMessage(), 'Unknown provider name') !== false;
        echo $name, $rejected ? "=rejected\n" : "=routed\n";
    }
}
?>
--EXPECT--
CUDA=routed
TensorRT=routed
