--TEST--
OrtValue keeps borrowed numeric buffers alive; repeated conversions; clone/constructor guards
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
// Numeric OrtValue borrows the Tensor buffer: dropping the Tensor must not
// invalidate the OrtValue.
$t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5]], 'float32');
$o = $t->toOrtValue();
unset($t);
gc_collect_cycles();
$r = $o->toTensor();
echo "borrow-ok:", $r->toArray()[0][0], "\n";
unset($o);

// The copied Tensor can itself be converted again independently.
$o2 = $r->toOrtValue();
unset($r);
gc_collect_cycles();
echo "second-gen:", $o2->toTensor()->toArray()[0][1], "\n";
unset($o2);

// String OrtValue owns its ORT-side copy; the source Tensor may die first.
$s = OnnxRuntime\Tensor::fromArray(["a\0b"], 'string');
$so = $s->toOrtValue();
unset($s);
gc_collect_cycles();
echo "string-lifetime:", bin2hex($so->toTensor()->toArray()[0]), "\n";
unset($so);

// Repeated string round-trips, including embedded NUL.
for ($i = 0; $i < 200; $i++) {
    $s = OnnxRuntime\Tensor::fromArray(["a\0b", "v$i"], 'string');
    $v = $s->toOrtValue();
    $x = $v->toTensor();
    unset($s, $v, $x);
}
gc_collect_cycles();
echo "loop-ok\n";

// OrtValue is not cloneable.
$t = OnnxRuntime\Tensor::fromArray([1], 'int64');
$o = $t->toOrtValue();
try {
    $c = clone $o;
    echo "clone-ok\n";
} catch (Error $e) {
    echo "clone-rejected\n";
}

// Direct construction is forbidden.
try {
    new OnnxRuntime\OrtValue();
} catch (Error $e) {
    echo "ctor-rejected\n";
}

// toTensor takes no arguments.
try {
    $o->toTensor(1);
} catch (ArgumentCountError $e) {
    echo "args-rejected\n";
}
?>
--EXPECT--
borrow-ok:1.5
second-gen:2.5
string-lifetime:610062
loop-ok
clone-rejected
ctor-rejected
args-rejected
