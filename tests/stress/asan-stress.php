<?php
// Stress conversions, clones, failure paths and Phase 2 OrtValue round-trips.
// Run with an ASan-instrumented build (see PHASE2.md).

function rows(int $n): array {
    $a = [];
    foreach (range(1, $n) as $i) {
        $a[] = [1.5, (float) $i];
    }
    return $a;
}

$t = OnnxRuntime\Tensor::fromArray(rows(2000), 'float32');
for ($i = 0; $i < 300; $i++) {
    $c = clone $t;
    $x = $c->toArray();
    unset($c, $x);
}

for ($i = 0; $i < 300; $i++) {
    try {
        OnnxRuntime\Tensor::fromArray([[1, 2], [3]], 'float32');
    } catch (Throwable $e) {
    }
}

for ($i = 0; $i < 300; $i++) {
    $s = OnnxRuntime\Tensor::fromArray([['a', 'b'], ['c', 'd']], 'string');
    $c = clone $s;
    $c->toArray();
    unset($s, $c);
}

$big = OnnxRuntime\Tensor::fromArray(
    array_fill(0, 500, array_fill(0, 500, 1.5)),
    'float32'
);
$big->toArray();
unset($big);

// Phase 2: OrtValue round-trips, including borrowed numeric buffers and NULs.
for ($i = 0; $i < 300; $i++) {
    $t = OnnxRuntime\Tensor::fromArray([[1.5, (float) $i]], 'float32');
    $o = $t->toOrtValue();
    unset($t);
    gc_collect_cycles();
    $r = $o->toTensor();
    unset($o, $r);
}
for ($i = 0; $i < 300; $i++) {
    $s = OnnxRuntime\Tensor::fromArray(["a\0b", "v$i"], 'string');
    $o = $s->toOrtValue();
    unset($s);
    $r = $o->toTensor();
    unset($o, $r);
}

// Phase 3: Session load/Run/lifetime and rejected calls.
$fixture = dirname(__DIR__) . '/fixtures/identity.onnx';
$session = new OnnxRuntime\Session($fixture);
for ($i = 0; $i < 150; $i++) {
    $t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], 'float32');
    $o = $session->run(['X' => $t]);
    unset($t);
    $o['Y']->toArray();
    unset($o);
}
for ($i = 0; $i < 30; $i++) {
    try {
        $session->run([]);
    } catch (Throwable $e) {
    }
    try {
        $session->run(['X' => OnnxRuntime\Tensor::fromArray([1, 2, 3], 'float32')]);
    } catch (Throwable $e) {
    }
}
// Zero-element tensors are valid inputs; do not reject them at run().
$zero_session = new OnnxRuntime\Session(dirname(__DIR__) . '/fixtures/zero_2d.onnx');
for ($i = 0; $i < 100; $i++) {
    $zero = OnnxRuntime\Tensor::fromArray([[], []], 'float32');
    $zero_out = $zero_session->run(['X' => $zero]);
    if ($zero_out['Y']->shape() !== [2, 0]) {
        throw new RuntimeException('unexpected zero-element output shape');
    }
    unset($zero, $zero_out);
}
unset($zero_session);

$last = $session->run(['X' => OnnxRuntime\Tensor::fromArray([[7, 8, 9], [10, 11, 12]], 'float32')])['Y'];
unset($session);
gc_collect_cycles();
$last->toArray();
unset($last);

for ($i = 0; $i < 10; $i++) {
    $s = new OnnxRuntime\Session($fixture);
    unset($s);
}
try {
    new OnnxRuntime\Session(__DIR__ . '/../fixtures/corrupt.onnx');
} catch (Throwable $e) {
}

// Non-Tensor I/O and unsupported tensor dtypes must fail at metadata load
// without leaking the partially built Ort::Session / metadata.
foreach (['sequence_io.onnx', 'float16_io.onnx'] as $unsupported) {
    for ($i = 0; $i < 20; $i++) {
        try {
            new OnnxRuntime\Session(dirname(__DIR__) . '/fixtures/' . $unsupported);
        } catch (Throwable $e) {
        }
    }
}
gc_collect_cycles();

// Phase 7A: repeated CPU I/O Binding runs, rebinding inputs and outputs.
$binding_session = new OnnxRuntime\Session($fixture);
$binding = $binding_session->createIoBinding();
$binding->bindOutput('Y');
for ($i = 0; $i < 150; $i++) {
    $input = OnnxRuntime\Tensor::fromArray(
        [[(float) $i, 2.5, 3.5], [4.5, 5.5, 6.5]],
        'float32'
    );
    $binding->bindInput('X', $input);
    unset($input);
    $out = $binding_session->runWithBinding($binding);
    $out['Y']->toArray();
    $binding->getOutputs()['Y']->toArray();
    unset($out);
}
// Dynamic output shapes must not reuse a stale output allocation.
$dynamic_session = new OnnxRuntime\Session(dirname(__DIR__) . '/fixtures/dynamic.onnx');
$dynamic_binding = $dynamic_session->createIoBinding();
$dynamic_binding->bindOutput('Y');
for ($i = 1; $i <= 30; $i++) {
    $rows = [];
    for ($r = 0; $r < $i; $r++) {
        $rows[] = [1.5, 2.5, 3.5];
    }
    $dynamic_binding->bindInput(
        'X', OnnxRuntime\Tensor::fromArray($rows, 'float32')
    );
    $dynamic_out = $dynamic_session->runWithBinding($dynamic_binding)['Y'];
    if (count($dynamic_out->shape()) !== 2 || $dynamic_out->shape()[0] !== $i) {
        throw new RuntimeException('unexpected dynamic I/O binding output shape');
    }
    unset($dynamic_out);
}
unset($dynamic_binding, $dynamic_session);

$binding_out = $binding_session->runWithBinding($binding)['Y'];
unset($binding_session, $binding);
gc_collect_cycles();
$binding_out->toArray();
unset($binding_out);

// Phase 7E: repeated multi-group batch inference with independent Tensor
// lifetimes; each entry is validated and executed separately.
$batch_session = new OnnxRuntime\Session(dirname(__DIR__) . '/fixtures/identity.onnx');
for ($round = 0; $round < 50; $round++) {
    $entries = [];
    for ($entry = 0; $entry < 4; $entry++) {
        $value = (float) ($round * 4 + $entry);
        $entries[] = [
            'X' => OnnxRuntime\Tensor::fromArray(
                [[$value, $value + 1, $value + 2],
                 [$value + 3, $value + 4, $value + 5]],
                'float32'
            ),
        ];
    }
    $results = $batch_session->runBatch($entries);
    if (count($results) !== 4) {
        throw new RuntimeException('unexpected batch result count');
    }
    foreach ($results as $result) {
        $result['Y']->toArray();
    }
    unset($entries, $results);
}
unset($batch_session);

echo "stress-ok\n";
