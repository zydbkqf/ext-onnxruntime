--TEST--
Tensor::toBuffer: raw bytes out, inverse layout and dtype cast
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

/* 1. identity: bytes that went in come back unchanged */
$bin = pack('g*', 1.0, 2.0, 3.0, 4.0);
$t = OnnxRuntime\Tensor::fromBuffer($bin, [2, 2]);
echo "1 ", strlen($t->toBuffer()), " ", var_export($t->toBuffer() === $bin, true), "\n";

/* 2. cast float32 -> uint8 saturates */
$t = OnnxRuntime\Tensor::fromBuffer(pack('g*', -1.0, 0.5, 2.5, 300.0), [4]);
echo "2 ", json_encode(unpack('C*', $t->toBuffer(['cast' => 'uint8']), 1)), "\n";

/* 3. layout: CHW tensor back to HWC bytes, and the raw CHW order */
$pixels = pack('C*', 10, 20, 30, 40, 50, 60);
$u = OnnxRuntime\Tensor::fromBuffer($pixels, [3, 1, 2], 'uint8', ['layout' => 'hwc']);
echo "3a ", var_export($u->toBuffer(['layout' => 'hwc']) === $pixels, true), "\n";
echo "3b ", json_encode(array_values(unpack('C*', $u->toBuffer()))), "\n";

/* 4. rank-4 round trip through a cast */
$pixels = pack('C*', 5, 6, 7, 8, 9, 10);
$v = OnnxRuntime\Tensor::fromBuffer($pixels, [1, 3, 1, 2], 'uint8', ['layout' => 'hwc']);
echo "4 ", var_export($v->toBuffer(['cast' => 'float32', 'layout' => 'hwc']) ===
    pack('g*', 5, 6, 7, 8, 9, 10), true), "\n";

/* 5. empty tensor yields an empty string */
echo "5 ", var_export(OnnxRuntime\Tensor::fromBuffer('', [0])->toBuffer(), true), "\n";

/* 6. int64 -> int16 cast saturates */
$t = OnnxRuntime\Tensor::fromBuffer(pack('q*', 1, 40000, -70000), [3], 'int64');
echo "6 ", json_encode(array_values(unpack('s*', $t->toBuffer(['cast' => 'int16'])))), "\n";

function check(string $label, callable $fn): void {
    try {
        $fn();
        echo $label, ": no exception\n";
    } catch (Throwable $e) {
        echo $label, ": ", get_class($e), ": ", $e->getMessage(), "\n";
    }
}

$str = OnnxRuntime\Tensor::fromArray(['a', 'b'], 'string');
check('7 string-tensor', fn() => $str->toBuffer());
check('8 unknown-option', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8')->toBuffer(['foo' => 1]));
check('9 bad-layout', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8')->toBuffer(['layout' => 'nchw']));
check('10 rank2-hwc', fn() => OnnxRuntime\Tensor::fromBuffer("\x00\x00", [2], 'uint8')->toBuffer(['layout' => 'hwc']));
check('11 bad-cast', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8')->toBuffer(['cast' => 'float128']));
check('12 list-options', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8')->toBuffer([1]));
--EXPECT--
1 16 true
2 {"1":0,"2":2,"3":255}
3a true
3b [10,40,20,50,30,60]
4 true
5 ''
6 [1,32767,-32768]
7 string-tensor: OnnxRuntime\ConversionException: toBuffer() does not serialise string tensors; their payload is variable length and has no byte layout
8 unknown-option: ValueError: Unknown toBuffer() option "foo"; supported options are layout, cast
9 bad-layout: ValueError: toBuffer() option "layout" must be "hwc", "chw" or "none", got "nchw"
10 rank2-hwc: OnnxRuntime\ConversionException: layout 'hwc' requires a rank-3 or rank-4 tensor
11 bad-cast: ValueError: Unsupported cast dtype "float128"; supported dtypes are: float32, float64, int64, int32, int16, int8, uint8, bool, string
12 list-options: ValueError: toBuffer() options must be an associative array, a list was given
