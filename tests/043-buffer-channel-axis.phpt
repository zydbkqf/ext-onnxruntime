--TEST--
Per-channel mean/std follows the target layout's channel axis
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);
$pf = JSON_PRESERVE_ZERO_FRACTION;

/* 1. rank-3 CHW: channels are dim 0, stride = H*W, so the three means apply
 *    to two elements each instead of being rejected as "1 channel". */
$t = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [3, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [1, 2, 3], 'std' => [1, 2, 2]]);
echo "1 ", json_encode($t->toArray(), $pf), "\n";

/* 2. the same shape declared as hwc uses the same channel axis, only the
 *    input byte order differs: NHWC bytes 10,20,30 / 40,50,60 */
$u = OnnxRuntime\Tensor::fromBuffer(pack('C*', 10, 20, 30, 40, 50, 60),
    [3, 1, 2], 'uint8',
    ['layout' => 'hwc', 'cast' => 'float32', 'mean' => [1, 2, 3]]);
echo "2 ", json_encode($u->toArray(), $pf), "\n";

/* 3. rank-2 keeps the last axis as the channel axis */
$v = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [2, 3],
    'uint8', ['cast' => 'float32', 'mean' => [10, 20, 30]]);
echo "3 ", json_encode($v->toArray(), $pf), "\n";

/* 4. rank-4 CHW: channels are dim 1 */
$w = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [1, 2, 3]]);
echo "4 ", json_encode($w->toArray(), $pf), "\n";

/* 5. the count error names the real channel count (3, not H = 1) */
try {
    OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [3, 1, 2],
        'uint8', ['cast' => 'float32', 'mean' => [1, 2]]);
} catch (OnnxRuntime\ConversionException $e) {
    echo "5 ", $e->getMessage(), "\n";
}

/* 6. a single value still broadcasts over all three channels */
$b = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [3, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => 1.0]);
echo "6 ", json_encode($b->toArray(), $pf), "\n";
?>
--EXPECT--
1 [[[-1.0,0.0]],[[0.0,0.5]],[[0.5,1.0]]]
2 [[[9.0,39.0]],[[18.0,48.0]],[[27.0,57.0]]]
3 [[-10.0,-19.0,-28.0],[-7.0,-16.0,-25.0]]
4 [[[[-1.0,0.0]],[[0.0,1.0]],[[1.0,2.0]]]]
5 mean has 2 entries but the channel count is 3
6 [[[-1.0,0.0]],[[1.0,2.0]],[[3.0,4.0]]]
