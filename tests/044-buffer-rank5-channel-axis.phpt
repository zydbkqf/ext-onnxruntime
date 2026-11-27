--TEST--
Per-channel mean/std on rank >= 4 uses the ONNX channel axis (dim 1)
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);
$pf = JSON_PRESERVE_ZERO_FRACTION;

/* 1. rank-5 [N,C,D,H,W] = [1,3,1,1,2]: the channel axis is dim 1, not the
 *    third position from the end (which would be D = 1 here). */
$t = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [1, 2, 3]]);
echo "1 ", json_encode($t->toArray(), $pf), "\n";

/* 2. the channel stride on rank-5 is D*H*W = 4, so every four bytes belong to
 *    one channel and the cycle restarts after C*D*H*W = 12. Two batches. */
$u = OnnxRuntime\Tensor::fromBuffer(pack('C*', ...range(0, 23)), [2, 3, 2, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [10, 20, 30]]);
echo "2 ", json_encode($u->toArray(), $pf), "\n";

/* 3. the count error names the real channel count (3, not D = 1) */
try {
    OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 1, 2],
        'uint8', ['cast' => 'float32', 'mean' => [1, 2]]);
} catch (OnnxRuntime\ConversionException $e) {
    echo "3 ", $e->getMessage(), "\n";
}

/* 4. 'hwc' stays limited to rank 3/4, rank-5 is rejected rather than guessed */
try {
    OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 1, 2],
        'uint8', ['layout' => 'hwc', 'cast' => 'float32']);
} catch (OnnxRuntime\ConversionException $e) {
    echo "4 ", $e->getMessage(), "\n";
}

/* 5. rank-4 is unchanged: channels are dim 1 and the stride is H*W */
$v = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [1, 2, 3]]);
echo "5 ", json_encode($v->toArray(), $pf), "\n";

/* 6. rank-6 [N,C,D1,D2,H,W] follows the same rule: dim 1, stride = D1*D2*H*W */
$w = OnnxRuntime\Tensor::fromBuffer(pack('C*', 0, 1, 2, 3, 4, 5), [1, 3, 1, 1, 1, 2],
    'uint8', ['cast' => 'float32', 'mean' => [1, 2, 3]]);
echo "6 ", json_encode($w->toArray(), $pf), "\n";
?>
--EXPECT--
1 [[[[[-1.0,0.0]]],[[[0.0,1.0]]],[[[1.0,2.0]]]]]
2 [[[[[-10.0,-9.0]],[[-8.0,-7.0]]],[[[-16.0,-15.0]],[[-14.0,-13.0]]],[[[-22.0,-21.0]],[[-20.0,-19.0]]]],[[[[2.0,3.0]],[[4.0,5.0]]],[[[-4.0,-3.0]],[[-2.0,-1.0]]],[[[-10.0,-9.0]],[[-8.0,-7.0]]]]]
3 mean has 2 entries but the channel count is 3
4 layout 'hwc' requires a rank-3 [H,W,C]/[C,H,W] or rank-4 [N,H,W,C]/[N,C,H,W] shape
5 [[[[-1.0,0.0]],[[0.0,1.0]],[[1.0,2.0]]]]
6 [[[[[[-1.0,0.0]]]],[[[[0.0,1.0]]]],[[[[1.0,2.0]]]]]]
