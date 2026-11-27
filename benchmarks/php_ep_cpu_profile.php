<?php
declare(strict_types=1);

/**
 * Measure one execution provider in isolation so that `/usr/bin/time -v` (or
 * `-f "%P"`) reports CPU usage for that provider only.
 *
 * php_ep_bench.php always runs a CPU column first and then the device column in
 * the same process, which makes the process-wide CPU total a mixture of both.
 * This script builds a single session, uses fixed dummy inputs, and leaves the
 * timing to the shell:
 *
 *   /usr/bin/time -f "cpu%=%P wall=%e" php -n -d extension=... \
 *     benchmarks/php_ep_cpu_profile.php --model=model.onnx \
 *     --provider=openvino --openvino-lib=/path/plugin.so --threads=1
 *
 * Usage: php_ep_cpu_profile.php --model FILE [--provider cpu|openvino]
 *   [--openvino-lib SO] [--threads N] [--iterations N] [--warmup N]
 *   [--batch-size N] [--sequence-length N] [--ep-option key:value ...]
 *
 * --ep-option is repeatable and forwards provider options to appendEpDevice().

 */

function argument(array $argv, string $name, ?string $default = null): ?string
{
    $prefix = '--' . $name . '=';
    foreach ($argv as $arg) {
        if (str_starts_with($arg, $prefix)) {
            return substr($arg, strlen($prefix));
        }
    }
    return $default;
}

$model = argument($argv, 'model');
$provider = argument($argv, 'provider', 'cpu');
$openvinoLib = argument($argv, 'openvino-lib');
$threads = max(1, (int) argument($argv, 'threads', '1'));
$iterations = max(1, (int) argument($argv, 'iterations', '20'));
$warmup = max(0, (int) argument($argv, 'warmup', '3'));
$batchSize = max(1, (int) argument($argv, 'batch-size', '8'));
$sequenceLength = max(1, (int) argument($argv, 'sequence-length', '64'));

$epOptions = [];
foreach ($argv as $arg) {
    if (str_starts_with($arg, '--ep-option=')) {
        $pair = substr($arg, strlen('--ep-option='));
        $colon = strpos($pair, ':');
        if ($colon === false) {
            throw new RuntimeException('--ep-option expects key:value, got ' . $pair);
        }
        $epOptions[substr($pair, 0, $colon)] = substr($pair, $colon + 1);
    }
}

if ($model === null || !is_file($model)) {
    throw new RuntimeException('model not found: ' . (string) $model);
}

$options = new OnnxRuntime\SessionOptions();
$options->setIntraOpNumThreads($threads);
if ($provider === 'openvino') {
    if ($openvinoLib === null || !is_file($openvinoLib)) {
        throw new RuntimeException('OpenVINO plugin not found: ' . (string) $openvinoLib);
    }
    $runtime = OnnxRuntime\Runtime::default();
    $runtime->registerExecutionProviderLibrary('openvino', $openvinoLib);
    $selected = null;
    foreach ($runtime->getEpDevices() as $device) {
        if ($device->epName() === 'OpenVINOExecutionProvider') {
            $selected = $device;
            break;
        }
    }
    if ($selected === null) {
        throw new RuntimeException('OpenVINO device not found');
    }
    $options->appendEpDevice($selected, $epOptions);
} elseif ($provider !== 'cpu') {
    throw new RuntimeException('unknown provider: ' . $provider);
}

$session = new OnnxRuntime\Session($model, $options);
$types = $session->inputTypes();
$shapes = $session->inputShapes();
$inputs = [];
foreach ($session->inputNames() as $name) {
    if (count($shapes[$name] ?? []) !== 2) {
        throw new RuntimeException(sprintf(
            "input '%s' is not rank 2; this profiler only builds [batch, sequence] inputs",
            $name
        ));
    }
    $dtype = str_contains($types[$name], 'int') ? 'int64' : 'float32';
    $rows = [];
    for ($i = 0; $i < $batchSize; $i++) {
        $rows[] = array_fill(0, $sequenceLength, $dtype === 'int64' ? 1 : 0.25);
    }
    $inputs[$name] = OnnxRuntime\Tensor::fromArray($rows, $dtype);
}

for ($i = 0; $i < $warmup; $i++) {
    $session->run($inputs);
}

$times = [];
for ($i = 0; $i < $iterations; $i++) {
    $start = hrtime(true);
    $session->run($inputs);
    $times[] = (hrtime(true) - $start) / 1e6;
}
sort($times);
$count = count($times);

printf(
    "provider=%s intra_op_threads=%d px50=%.2f ms pmin=%.2f pmax=%.2f\n",
    $provider,
    $threads,
    $times[(int) floor(($count - 1) * 0.5)],
    $times[0],
    $times[$count - 1]
);
