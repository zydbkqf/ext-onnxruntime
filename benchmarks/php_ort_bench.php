<?php
declare(strict_types=1);

/**
 * Run ORT inference from the PHP extension with pre-tokenized int64 inputs.
 *
 * Usage:
 *   php -n -d extension=modules/onnxruntime.so \
 *     benchmarks/php_ort_bench.php spec.json
 *
 * The spec is produced by benchmarks/fastembed_int8_bpe.py.
 */

if ($argc !== 2) {
    fwrite(STDERR, "usage: php_ort_bench.php <spec.json>\n");
    exit(2);
}

$spec = json_decode(file_get_contents($argv[1]), true, 512, JSON_THROW_ON_ERROR);

$model = $spec['model'];
$provider = $spec['provider'];
$iterations = (int) $spec['iterations'];
$warmup = (int) $spec['warmup'];
$cudaLib = $spec['cuda_lib'] ?? null;

$options = new OnnxRuntime\SessionOptions();
$options->setIntraOpNumThreads((int) ($spec['threads'] ?? 1));

if ($provider === 'coreml') {
    $options->appendExecutionProvider('CoreML');
} elseif ($provider === 'cuda') {
    if (!is_string($cudaLib) || !is_file($cudaLib)) {
        throw new RuntimeException('CUDA provider library not found: ' . (string) $cudaLib);
    }
    $runtime = OnnxRuntime\Runtime::default();
    $runtime->registerExecutionProviderLibrary('cuda', $cudaLib);
    $cuda = null;
    foreach ($runtime->getEpDevices() as $device) {
        if ($device->epName() === 'CUDAExecutionProvider') {
            $cuda = $device;
            break;
        }
    }
    if ($cuda === null) {
        throw new RuntimeException('CUDA EpDevice not found');
    }
    $options->appendEpDevice($cuda);
} elseif ($provider !== 'cpu') {
    throw new RuntimeException('unknown provider: ' . $provider);
}

$inputs = [];
foreach ($spec['inputs'] as $name => $values) {
    // The BGE model uses int64 input_ids / attention_mask / token_type_ids.
    $inputs[$name] = OnnxRuntime\Tensor::fromArray($values, 'int64');
}

$session = new OnnxRuntime\Session($model, $options);

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
$sum = array_sum($times);
$result = [
    'framework' => 'php',
    'provider' => $provider,
    'php_version' => PHP_VERSION,
    'extension_version' => phpversion('onnxruntime'),
    'batch_size' => count($spec['inputs']['input_ids']),
    'sequence_length' => count($spec['inputs']['input_ids'][0]),
    'iterations' => $iterations,
    'mean_ms' => $sum / $count,
    'p50_ms' => $times[(int) floor($count * 0.50)],
    'p95_ms' => $times[min($count - 1, (int) floor($count * 0.95))],
    'min_ms' => $times[0],
    'max_ms' => $times[$count - 1],
];

echo json_encode($result, JSON_THROW_ON_ERROR), "\n";
