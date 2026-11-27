--TEST--
Tensor buffers keep int64 values byte-exact and saturate outside int64 range
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

/* 1. hwc re-order of int64 values around 2^53 and the int64 extremes: a
 *    re-order must not round-trip the payload through double.
 *    Buffer is HWC with H=1, W=2, C=2; target shape is [C,H,W] = [2,1,2]. */
$src = [
    [[PHP_INT_MIN, -9007199254740993], [9007199254740995, PHP_INT_MAX]],
];
$buf = '';
foreach ($src as $h) {
    foreach ($h as $w) {
        foreach ($w as $c) {
            $buf .= pack('P', $c);
        }
    }
}
$t = OnnxRuntime\Tensor::fromBuffer($buf, [2, 1, 2], 'int64', ['layout' => 'hwc']);
echo "1 ", json_encode($t->shape()), " ", json_encode($t->toArray()), "\n";

/* 2. the same tensor back to HWC bytes, byte for byte */
echo "2 ", var_export($t->toBuffer(['layout' => 'hwc']) === $buf, true), "\n";

/* 3. float64 -> int64 out of range saturates instead of wrapping.  -2^63
 *    itself is a valid int64 and must not saturate the other way; (float)
 *    PHP_INT_MAX rounds up to 2^63 and must land on INT64_MAX. */
$f = OnnxRuntime\Tensor::fromBuffer(
    pack('d*', 1e300, -1e300, 9.3e18, -9223372036854775808.0, (float) PHP_INT_MAX),
    [5],
    'float64',
    ['cast' => 'int64']
);
echo "3 ", json_encode($f->toArray()), "\n";

/* 4. no-op cast (int64 -> int64) with a rank-4 re-order keeps every value.
 *    Buffer is NHWC with N=2, H=1, W=2, C=2; target is [N,C,H,W]. */
$v = OnnxRuntime\Tensor::fromBuffer(
    pack('P*', 9007199254740993, 3, PHP_INT_MIN, 9007199254740995,
         5, 6, 7, PHP_INT_MAX),
    [2, 2, 1, 2],
    'int64',
    ['layout' => 'hwc']
);
echo "4 ", json_encode($v->toArray()), "\n";
?>
--EXPECT--
1 [2,1,2] [[[-9223372036854775808,9007199254740995]],[[-9007199254740993,9223372036854775807]]]
2 true
3 [9223372036854775807,-9223372036854775808,9223372036854775807,-9223372036854775808,9223372036854775807]
4 [[[[9007199254740993,-9223372036854775808]],[[3,9007199254740995]]],[[[5,7]],[[6,9223372036854775807]]]]
