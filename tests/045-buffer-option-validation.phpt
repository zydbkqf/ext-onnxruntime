--TEST--
Buffer option validation: scale/mean/std must be finite and std must be positive
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

/* Every rejected case must throw before any tensor storage is built and
 * before the stream is read, so the same checks hold for fromBuffer() and
 * fromStream().  The formula is (x * scale - mean) / std: a zero divisor or a
 * NaN/Inf parameter would silently turn the payload into Inf/NaN. */

function rejects(string $label, callable $fn): void {
    try {
        $fn();
        echo $label, ": no exception\n";
    } catch (ValueError $e) {
        echo $label, ": ValueError: ", $e->getMessage(), "\n";
    }
}

$bytes = pack('C*', 10, 20, 30);

/* 1-5: std must be present, finite and greater than zero */
rejects('std=0 (int)', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'std' => 0]));
rejects('std=0.0', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'std' => 0.0]));
rejects('std=-1', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'std' => -1]));
rejects('std=NAN', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'std' => NAN]));
rejects('std=INF', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'std' => INF]));

/* 6: the per-channel list is checked element by element */
rejects('std list with zero', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'mean' => [0, 0, 0], 'std' => [1, 0, 1]]));

/* 7-9: mean entries must be finite */
rejects('mean list with NAN', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'mean' => [0, NAN, 0]]));
rejects('mean=INF', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'mean' => INF]));
rejects('mean list with -INF', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'mean' => [0, 0, -INF]]));

/* 10-11: scale must be finite */
rejects('scale=NAN', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'scale' => NAN]));
rejects('scale=-INF', fn() => OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'scale' => -INF]));

/* 12: a legal normalisation still works: (v * 1 - 10) / 2 */
$t = OnnxRuntime\Tensor::fromBuffer($bytes, [3], 'uint8',
    ['cast' => 'float32', 'mean' => 10, 'std' => 2]);
echo "12 ", json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

/* 13-16: fromStream() goes through the same parser, and the rejection happens
 * before the stream is touched (position stays at 0). */
$path = tempnam(sys_get_temp_dir(), 'ort');
file_put_contents($path, $bytes);

$fp = fopen($path, 'rb');
rejects('fromStream std=0', fn() => OnnxRuntime\Tensor::fromStream($fp, [3], 'uint8',
    ['cast' => 'float32', 'std' => 0]));
echo "14 stream position after rejected std: ", ftell($fp), "\n";

rejects('fromStream scale=NAN', fn() => OnnxRuntime\Tensor::fromStream($fp, [3], 'uint8',
    ['cast' => 'float32', 'scale' => NAN]));
echo "15 stream position after rejected scale: ", ftell($fp), "\n";

$s = OnnxRuntime\Tensor::fromStream($fp, [3], 'uint8',
    ['cast' => 'float32', 'mean' => 10, 'std' => 2]);
echo "16 ", json_encode($s->toArray(), JSON_PRESERVE_ZERO_FRACTION),
     " consumed ", ftell($fp), " bytes\n";

fclose($fp);
unlink($path);

/* 17: a missing std (default 1) and a missing scale (default 1) stay valid */
$u = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2), [3], 'uint8',
    ['cast' => 'float32']);
echo "17 ", json_encode($u->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";
?>
--EXPECT--
std=0 (int): ValueError: fromBuffer() option "std" must be greater than zero
std=0.0: ValueError: fromBuffer() option "std" must be greater than zero
std=-1: ValueError: fromBuffer() option "std" must be greater than zero
std=NAN: ValueError: fromBuffer() option "std" must contain only finite numbers
std=INF: ValueError: fromBuffer() option "std" must contain only finite numbers
std list with zero: ValueError: fromBuffer() option "std" must be greater than zero
mean list with NAN: ValueError: fromBuffer() option "mean" must contain only finite numbers
mean=INF: ValueError: fromBuffer() option "mean" must contain only finite numbers
mean list with -INF: ValueError: fromBuffer() option "mean" must contain only finite numbers
scale=NAN: ValueError: fromBuffer() option "scale" must be a finite number
scale=-INF: ValueError: fromBuffer() option "scale" must be a finite number
12 [0.0,5.0,10.0]
fromStream std=0: ValueError: fromStream() option "std" must be greater than zero
14 stream position after rejected std: 0
fromStream scale=NAN: ValueError: fromStream() option "scale" must be a finite number
15 stream position after rejected scale: 0
16 [0.0,5.0,10.0] consumed 3 bytes
17 [0.0,1.0,2.0]
