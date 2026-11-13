--TEST--
SessionOptions validates CPU/runtime settings and snapshots them into a Session
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(array $a): void {
    echo json_encode($a), "\n";
}

$o = new OnnxRuntime\SessionOptions();
show($o->toArray());

$o->setIntraOpNumThreads(4);
$o->setInterOpNumThreads(2);
$o->setGraphOptimizationLevel('ALL');
$o->setExecutionMode('Parallel');
$o->setLogSeverityLevel('Error');
$o->setEnableCpuMemArena(false);
$o->setEnableMemPattern(false);
show($o->toArray());

$clone = clone $o;
$clone->setIntraOpNumThreads(1);
echo 'clone-intra=', $clone->toArray()['intraOpNumThreads'], "\n";
echo 'original-intra=', $o->toArray()['intraOpNumThreads'], "\n";

$ep = new OnnxRuntime\SessionOptions();
$ep->appendExecutionProvider('CoreML', ['MLProgram' => '1']);
echo 'ep=', json_encode($ep->toArray()['executionProviders']), "\n";
try {
    $ep->appendExecutionProvider('');
    echo 'empty-ep:no-error', "\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo 'empty-ep:ConversionException', "\n";
}

$cases = [
    'thread-negative' => fn() => $o->setIntraOpNumThreads(-1),
    'thread-too-large' => fn() => $o->setIntraOpNumThreads(PHP_INT_MAX),
    'graph-unknown' => fn() => $o->setGraphOptimizationLevel('fast'),
    'execution-unknown' => fn() => $o->setExecutionMode('parallel-ish'),
    'log-unknown' => fn() => $o->setLogSeverityLevel('debug'),
    'log-range' => fn() => $o->setLogSeverityLevel(9),
];
foreach ($cases as $label => $call) {
    try {
        $call();
        echo $label, ':no-error', "\n";
    } catch (Throwable $e) {
        echo $label, ':', get_class($e), "\n";
    }
}

$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', $o);
$t = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32');
$o->setIntraOpNumThreads(8); // must not affect the already-created Session
echo 'run=', json_encode($s->run(['X' => $t])['Y']->toArray()), "\n";

$null = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', null);
echo 'null-options=', json_encode($null->run(['X' => $t])['Y']->toArray()), "\n";

try {
    new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx', new stdClass());
    echo 'wrong-options:no-error', "\n";
} catch (OnnxRuntime\ConversionException $e) {
    echo 'wrong-options:ConversionException', "\n";
}

$reflection = new ReflectionClass(OnnxRuntime\SessionOptions::class);
echo 'final=', $reflection->isFinal() ? 'yes' : 'no', "\n";
?>
--EXPECT--
{"intraOpNumThreads":0,"interOpNumThreads":0,"graphOptimizationLevel":"all","executionMode":"sequential","logSeverityLevel":2,"logSeverityName":"warning","enableCpuMemArena":true,"enableMemPattern":true,"executionProviders":[],"epDevices":[]}
{"intraOpNumThreads":4,"interOpNumThreads":2,"graphOptimizationLevel":"all","executionMode":"parallel","logSeverityLevel":3,"logSeverityName":"error","enableCpuMemArena":false,"enableMemPattern":false,"executionProviders":[],"epDevices":[]}
clone-intra=1
original-intra=4
ep=[{"name":"CoreML","options":{"MLProgram":"1"}}]
empty-ep:ConversionException
thread-negative:OnnxRuntime\ConversionException
thread-too-large:OnnxRuntime\ConversionException
graph-unknown:OnnxRuntime\ConversionException
execution-unknown:OnnxRuntime\ConversionException
log-unknown:OnnxRuntime\ConversionException
log-range:OnnxRuntime\ConversionException
run=[[1,2,3],[4,5,6]]
null-options=[[1,2,3],[4,5,6]]
wrong-options:ConversionException
final=yes
