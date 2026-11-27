--TEST--
Tensor::fromStream: raw bytes from a PHP stream, offsets and short reads
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

/* 1. whole tensor from a file handle */
$path = tempnam(sys_get_temp_dir(), 'ort');
file_put_contents($path, pack('g*', 1, 2, 3, 4, 5, 6));
$fp = fopen($path, 'rb');
$t = OnnxRuntime\Tensor::fromStream($fp, [2, 3]);
echo "1 ", $t->dtype(), " ", json_encode($t->toArray()), "\n";
fclose($fp);
unlink($path);

/* 2. two tensors read sequentially from one stream after skipping a header */
$fp = fopen('php://memory', 'w+');
fwrite($fp, str_repeat(chr(0), 8));
fwrite($fp, pack('C*', 10, 20, 30, 40));
fwrite($fp, pack('C*', 50, 60, 70, 80, 90, 100));
rewind($fp);
$t = OnnxRuntime\Tensor::fromStream($fp, [2, 2], 'uint8', ['offset' => 8]);
echo "2a ", json_encode($t->toArray()), "\n";
$t = OnnxRuntime\Tensor::fromStream($fp, [3, 2], 'uint8');
echo "2b ", json_encode($t->toArray()), "\n";

/* 3. layout + cast + normalisation straight off the stream */
fseek($fp, 12);
$t = OnnxRuntime\Tensor::fromStream($fp, [1, 3, 1, 2], 'uint8', [
    'layout' => 'hwc',
    'cast' => 'float32',
    'scale' => 1 / 255,
    'mean' => 0.5,
    'std' => 0.5,
]);
echo "3 ", $t->dtype(), " ", json_encode($t->shape()), "\n";
foreach ($t->toArray()[0] as $plane) {
    echo "  ", json_encode($plane), "\n";
}
fclose($fp);

/* 4. length asserts the payload size up front */
$fp = fopen('php://memory', 'w+');
fwrite($fp, pack('C*', 1, 2, 3, 4));

/* 5. error paths */
function check(string $label, callable $fn): void {
    try {
        $fn();
        echo $label, ": no exception\n";
    } catch (Throwable $e) {
        echo $label, ": ", get_class($e), ": ", $e->getMessage(), "\n";
    }
}

check('4 length', fn() => OnnxRuntime\Tensor::fromStream($fp, [4], 'uint8', ['length' => 8]));
check('5 short', fn() => OnnxRuntime\Tensor::fromStream($fp, [4], 'float32'));
check('6 offset-eof', fn() => OnnxRuntime\Tensor::fromStream($fp, [1], 'uint8', ['offset' => 100]));
check('7 offset-neg', fn() => OnnxRuntime\Tensor::fromStream($fp, [1], 'uint8', ['offset' => -1]));
check('8 unknown', fn() => OnnxRuntime\Tensor::fromStream($fp, [1], 'uint8', ['foo' => 1]));
check('9 not-stream', fn() => OnnxRuntime\Tensor::fromStream('nope', [1], 'uint8'));
check('10 read-after-eof', function () use ($fp) {
    fseek($fp, 4);
    return OnnxRuntime\Tensor::fromStream($fp, [4], 'uint8');
});
fclose($fp);

/* 11. options reuse: offset still rejected by fromBuffer */
check('11 buffer-offset', fn() => OnnxRuntime\Tensor::fromBuffer("\x01", [1], 'uint8', ['offset' => 1]));
--EXPECT--
1 float32 [[1,2,3],[4,5,6]]
2a [[10,20],[30,40]]
2b [[50,60],[70,80],[90,100]]
3 float32 [1,3,1,2]
  [[-0.6078431606292725,-0.37254902720451355]]
  [[-0.529411792755127,-0.29411765933036804]]
  [[-0.45098039507865906,-0.21568627655506134]]
4 length: OnnxRuntime\ConversionException: length 8 does not match shape/dtype (expected 4 bytes)
5 short: OnnxRuntime\ConversionException: stream yielded 0 of 16 bytes; it is empty, exhausted, or not readable
6 offset-eof: OnnxRuntime\ConversionException: stream yielded 0 of 1 bytes; it is empty, exhausted, or not readable
7 offset-neg: ValueError: fromStream() option "offset" must be a non-negative integer number of bytes
8 unknown: ValueError: Unknown fromStream() option "foo"; supported options are layout, cast, scale, mean, std, offset, length
9 not-stream: TypeError: OnnxRuntime\Tensor::fromStream(): Argument #1 ($stream) must be of type resource, string given
10 read-after-eof: OnnxRuntime\ConversionException: stream yielded 0 of 4 bytes; it is empty, exhausted, or not readable
11 buffer-offset: ValueError: Unknown fromBuffer() option "offset"; supported options are layout, cast, scale, mean, std
