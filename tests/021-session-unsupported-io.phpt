--TEST--
Session rejects non-Tensor I/O and unsupported tensor dtypes explicitly
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$cases = [
    'sequence_io.onnx' => 'not a tensor',
    'float16_io.onnx' => 'float16',
];
foreach ($cases as $file => $needle) {
    try {
        new OnnxRuntime\Session(__DIR__ . '/fixtures/' . $file);
        echo $file, ':no-error', "\n";
    } catch (Throwable $e) {
        $message = $e->getMessage();
        echo $file, ':', get_class($e), ':',
            (str_contains($message, $needle) ? 'explicit' : 'missing-detail'),
            "\n";
    }
}
echo "unsupported-io-ok\n";
?>
--EXPECT--
sequence_io.onnx:OnnxRuntime\ConversionException:explicit
float16_io.onnx:OnnxRuntime\ConversionException:explicit
unsupported-io-ok
