--TEST--
Repeated Run and Session destruction leave returned Tensors valid
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');
$t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], 'float32');

$last = null;
for ($i = 0; $i < 100; $i++) {
    $last = $s->run(['X' => $t])['Y'];
}
echo 'loop=', json_encode($last->toArray()), "\n";

// Drop the Session and input before reading the returned Tensor.
unset($s, $t);
gc_collect_cycles();
echo 'after-session=', json_encode($last->toArray()), "\n";

$m = new OnnxRuntime\Session(__DIR__ . '/fixtures/multi_io.onnx');
$a = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
$b = OnnxRuntime\Tensor::fromArray([[10, 20, 30], [40, 50, 60]], 'float32');
$out = $m->run(['A' => $a, 'B' => $b]);
unset($m, $a, $b);
gc_collect_cycles();
echo 'multi-after-session=', json_encode($out['sum']->toArray()), "\n";
?>
--EXPECT--
loop=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
after-session=[[1.5,2.5,3.5],[4.5,5.5,6.5]]
multi-after-session=[[11,22,33],[44,55,66]]
