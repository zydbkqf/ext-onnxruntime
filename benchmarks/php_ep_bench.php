<?php
declare(strict_types=1);

/**
 * Measure CPU vs a device-backed execution provider on one ONNX model.
 *
 * Usage:
 *   php -n -d extension=php_onnxruntime.dll benchmarks/php_ep_bench.php ^
 *     --model model.onnx --provider openvino ^
 *     --openvino-lib D:\php\openvino-ep\onnxruntime_providers_openvino_plugin.dll
 *
 * The model must have static input shapes. Inputs are filled with deterministic
 * dummy values, so this measures execution-provider overhead and throughput on
 * a real graph rather than model accuracy.
 */

function usage(): void
{
    fwrite(STDERR, "usage: php_ep_bench.php --model FILE [--provider cpu|openvino] "
        . "[--openvino-lib DLL] [--openvino-device NAME] "
        . "[--iterations N] [--warmup N] [--threads N] [--batch-size N] "
        . "[--sequence-length N] [--output FILE]\n");
}

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

function percentile(array $times, float $q): float
{
    $index = (int) floor((count($times) - 1) * $q);
    return $times[$index];
}

function dummyValue(array $shape, string $dtype, int &$counter): mixed
{
    if ($shape === []) {
        return match (strtolower($dtype)) {
            'bool' => true,
            'string' => 'bench',
            'float32', 'float64', 'float16', 'bfloat16' => 0.25,
            default => 1,
        };
    }

    $dim = array_shift($shape);
    $values = [];
    for ($i = 0; $i < $dim; $i++) {
        $values[] = dummyValue($shape, $dtype, $counter);
    }
    return $values;
}

/**
 * @return array<string, OnnxRuntime\Tensor>
 */
function modelInputs(OnnxRuntime\Session $session, int $batchSize, int $sequenceLength): array
{
    $names = $session->inputNames();
    $types = $session->inputTypes();
    $shapes = $session->inputShapes();
    $inputs = [];
    $counter = 0;

    foreach ($names as $name) {
        $shape = $shapes[$name] ?? null;
        if (!is_array($shape)) {
            throw new RuntimeException("input '$name' has no shape");
        }
        if (in_array(null, $shape, true)) {
            if (count($shape) !== 2) {
                throw new RuntimeException("input '$name' has an unsupported dynamic shape; pass a static-shape model");
            }
            // Only replace the dynamic dimensions; a [null, 768] input keeps
            // its static hidden size instead of becoming [batch, sequence].
            foreach ($shape as $index => $dim) {
                if ($dim === null) {
                    $shape[$index] = $index === 0 ? $batchSize : $sequenceLength;
                }
            }
        } else {
            $shape = array_map('intval', $shape);
        }
        $inputs[$name] = OnnxRuntime\Tensor::fromArray(
            dummyValue($shape, (string) $types[$name], $counter),
            (string) $types[$name]
        );
    }

    return $inputs;
}

function makeSessionOptions(string $provider, ?string $openvinoLib, ?string $openvinoDevice, int $threads): OnnxRuntime\SessionOptions
{
    $options = new OnnxRuntime\SessionOptions();
    $options->setIntraOpNumThreads($threads);

    if ($provider === 'cpu') {
        return $options;
    }

    if ($provider === 'openvino') {
        if ($openvinoLib === null || !is_file($openvinoLib)) {
            throw new RuntimeException('OpenVINO plugin not found: ' . (string) $openvinoLib);
        }
        $runtime = OnnxRuntime\Runtime::default();
        $runtime->registerExecutionProviderLibrary('openvino', $openvinoLib);
        $selected = null;
        $wanted = $openvinoDevice ?? 'OpenVINOExecutionProvider';
        foreach ($runtime->getEpDevices() as $device) {
            if ($device->epName() === $wanted) {
                $selected = $device;
                break;
            }
        }
        if ($selected === null) {
            throw new RuntimeException('OpenVINO device not found: ' . $wanted);
        }
        $options->appendEpDevice($selected);
        return $options;
    }

    throw new RuntimeException('unknown provider: ' . $provider);
}

function measure(OnnxRuntime\Session $session, array $inputs, int $warmup, int $iterations): array
{
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

    return [
        'mean_ms' => array_sum($times) / $count,
        'p50_ms' => percentile($times, 0.50),
        'p95_ms' => percentile($times, 0.95),
        'min_ms' => $times[0],
        'max_ms' => $times[$count - 1],
    ];
}

$model = argument($argv, 'model');
if ($model === null || !is_file($model)) {
    usage();
    throw new RuntimeException('model not found: ' . (string) $model);
}

$provider = argument($argv, 'provider', 'cpu');
$openvinoLib = argument($argv, 'openvino-lib');
$openvinoDevice = argument($argv, 'openvino-device');
$iterations = max(1, (int) argument($argv, 'iterations', '100'));
$warmup = max(0, (int) argument($argv, 'warmup', '3'));
$threads = max(1, (int) argument($argv, 'threads', '1'));
$batchSize = max(1, (int) argument($argv, 'batch-size', '8'));
$sequenceLength = max(1, (int) argument($argv, 'sequence-length', '64'));
$output = argument($argv, 'output');

$cpuOptions = makeSessionOptions('cpu', null, null, $threads);
$cpuSession = new OnnxRuntime\Session($model, $cpuOptions);
$inputs = modelInputs($cpuSession, $batchSize, $sequenceLength);

$result = [
    'model' => realpath($model),
    'provider' => $provider,
    'php_version' => PHP_VERSION,
    'extension_version' => phpversion('onnxruntime'),
    'onnxruntime_version' => OnnxRuntime\Runtime::default()->getVersion(),
    'iterations' => $iterations,
    'warmup' => $warmup,
    'threads' => $threads,
    'batch_size' => $batchSize,
    'sequence_length' => $sequenceLength,
    'input_shapes' => array_map(static fn(OnnxRuntime\Tensor $t): array => $t->shape(), $inputs),
];

$providers = ['cpu'];
if ($provider !== 'cpu') {
    $providers[] = $provider;
}

foreach ($providers as $name) {
    $sessionOptions = makeSessionOptions($name, $openvinoLib, $openvinoDevice, $threads);
    $session = new OnnxRuntime\Session($model, $sessionOptions);
    $result[$name] = measure($session, $inputs, $warmup, $iterations);
}

$json = json_encode($result, JSON_THROW_ON_ERROR | JSON_PRETTY_PRINT) . "\n";
if ($output !== null) {
    file_put_contents($output, $json);
}
echo $json;
