--TEST--
Tensor::fromBuffer: raw bytes, layout reorder, cast and normalisation
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

/* 1. plain float32 buffer, no options: bytes are taken verbatim */
$buf = pack('g*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
$t = OnnxRuntime\Tensor::fromBuffer($buf, [2, 3]);
echo "1 ", $t->dtype(), " ", json_encode($t->shape()), " ",
     json_encode($t->toArray()), "\n";

/* 2. uint8 HWC pixels (2x3, 2 channels) -> float32 CHW with scale/mean/std
 *    v = (v * 1/255 - 0.5) / 0.5 */
$pixels = pack('C*', 0, 255,   51, 204,   102, 153,   128, 128,   64, 192,   255, 0);
$t = OnnxRuntime\Tensor::fromBuffer($pixels, [2, 2, 3], 'uint8', [
    'layout' => 'hwc',
    'cast' => 'float32',
    'scale' => 1 / 255,
    'mean' => 0.5,
    'std' => 0.5,
]);
echo "2 ", $t->dtype(), " ", json_encode($t->shape()), "\n";
foreach ($t->toArray()[0] as $plane) {
    echo "  ", json_encode($plane), "\n";
}

/* 3. rank 4 NCHW from NHWC, per-channel mean */
$pixels = pack('C*', 10, 20, 30,   40, 50, 60);
$t = OnnxRuntime\Tensor::fromBuffer($pixels, [1, 3, 1, 2], 'uint8', [
    'layout' => 'hwc',
    'cast' => 'float64',
    'mean' => [1.0, 2.0, 3.0],
]);
echo "3 ", $t->dtype(), " ", json_encode($t->shape()), "\n";
echo "  ", json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";

/* 4. int32 -> uint8 cast saturates and truncates */
$ints = pack('l*', -400, 3, 999, 0);
$t = OnnxRuntime\Tensor::fromBuffer($ints, [2, 2], 'int32', ['cast' => 'uint8']);
echo "4 ", $t->dtype(), " ", json_encode($t->toArray()), "\n";

/* 5. float32 -> int32 cast truncates toward zero */
$t = OnnxRuntime\Tensor::fromBuffer(pack('g*', 1.9, -2.7, 3.0), [3], 'float32',
                                    ['cast' => 'int32']);
echo "5 ", $t->dtype(), " ", json_encode($t->toArray()), "\n";

/* error paths */
function check(string $label, callable $fn): void {
    try {
        $fn();
        echo $label, ": no exception\n";
    } catch (Throwable $e) {
        echo $label, ": ", get_class($e), ": ", $e->getMessage(), "\n";
    }
}

check('6 length', fn() => OnnxRuntime\Tensor::fromBuffer("\x00\x01", [3], 'uint8'));
check('7 dtype', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'float128'));
check('8 string', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'string'));
check('9 option', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8', ['foo' => 1]));
check('10 layout', fn() => OnnxRuntime\Tensor::fromBuffer("\x00\x00", [2], 'uint8', ['layout' => 'hwc']));
check('11 layout-name', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [1], 'uint8', ['layout' => 'nchw']));
check('12 mean-len', fn() => OnnxRuntime\Tensor::fromBuffer("\x00\x00\x00\x00", [1, 2, 1, 2], 'uint8',
                                                            ['layout' => 'hwc', 'cast' => 'float32', 'mean' => [1, 2, 3]]));
check('13 norm-int', fn() => OnnxRuntime\Tensor::fromBuffer("\x00\x00\x00\x00", [1], 'int32', ['scale' => 2]));
check('14 neg-dim', fn() => OnnxRuntime\Tensor::fromBuffer("\x00", [-1], 'uint8'));

/* 15. round-trip through a session input requirement: shape product preserved */
$t = OnnxRuntime\Tensor::fromBuffer(str_repeat("\x00", 12), [2, 3, 2], 'uint8', ['layout' => 'hwc']);
echo "15 ", json_encode($t->shape()), "\n";
--EXPECT--
1 float32 [2,3] [[1,2,3],[4,5,6]]
2 float32 [2,2,3]
  [-1,-0.6000000238418579,-0.20000000298023224]
  [0.003921568859368563,-0.49803921580314636,1]
3 float64 [1,3,1,2]
  [[[[9.0,39.0]],[[18.0,48.0]],[[27.0,57.0]]]]
4 uint8 [[0,3],[255,0]]
5 int32 [1,-2,3]
6 length: OnnxRuntime\ConversionException: buffer length 2 does not match shape/dtype (expected 3 bytes for 3 x 1 bytes)
7 dtype: ValueError: Unsupported dtype "float128"; supported dtypes are: float32, float64, int64, int32, int16, int8, uint8, bool, string
8 string: OnnxRuntime\ConversionException: fromBuffer() does not accept string tensors; raw string payloads are variable length and need their own representation
9 option: ValueError: Unknown fromBuffer() option "foo"; supported options are layout, cast, scale, mean, std
10 layout: OnnxRuntime\ConversionException: layout 'hwc' requires a rank-3 [H,W,C]/[C,H,W] or rank-4 [N,H,W,C]/[N,C,H,W] shape
11 layout-name: ValueError: fromBuffer() option "layout" must be "hwc", "chw" or "none", got "nchw"
12 mean-len: OnnxRuntime\ConversionException: mean has 3 entries but the channel count is 2
13 norm-int: OnnxRuntime\ConversionException: scale/mean/std require a float32 or float64 target dtype
14 neg-dim: OnnxRuntime\ConversionException: tensor dimensions must be non-negative
15 [2,3,2]
