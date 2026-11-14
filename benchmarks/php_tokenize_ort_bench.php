<?php
declare(strict_types=1);

/**
 * Benchmark PHP-side WordPiece tokenization + ORT inference.
 *
 * The tokenizer is loaded once; every measured iteration normalizes and
 * tokenizes the batch, builds the tensors, and runs the session. Session
 * creation is excluded, matching the other benchmark columns.
 *
 * Usage:
 *   php -n -d extension=modules/onnxruntime.so \
 *     benchmarks/php_tokenize_ort_bench.php spec.json
 */

require __DIR__ . '/php_wordpiece.php';

if ($argc !== 2) {
    fwrite(STDERR, "usage: php_tokenize_ort_bench.php <spec.json>\n");
    exit(2);
}

$spec = json_decode(file_get_contents($argv[1]), true, 512, JSON_THROW_ON_ERROR);

$model = $spec['model'];
$provider = $spec['provider'];
$iterations = (int) $spec['iterations'];
$warmup = (int) $spec['warmup'];
$cudaLib = $spec['cuda_lib'] ?? null;
$tokenizerPath = $spec['tokenizer_path'] ?? null;
$text = (string) ($spec['text'] ?? 'The quick brown fox jumps over the lazy dog.');
$batchSize = (int) ($spec['batch_size'] ?? 8);
$sequenceLength = (int) ($spec['sequence_length'] ?? 64);

if (!is_string($tokenizerPath) || !is_file($tokenizerPath)) {
    throw new RuntimeException('tokenizer.json not found: ' . (string) $tokenizerPath);
}

$tokenizer = new BenchWordPieceTokenizer($tokenizerPath);

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

$session = new OnnxRuntime\Session($model, $options);

$runOnce = static function () use ($tokenizer, $session, $text, $batchSize, $sequenceLength): void {
    $encoded = $tokenizer->encodeBatch($text, $batchSize, $sequenceLength);
    $inputs = [];
    foreach ($encoded as $name => $values) {
        $inputs[$name] = OnnxRuntime\Tensor::fromArray($values, 'int64');
    }
    $session->run($inputs);
};

for ($i = 0; $i < $warmup; $i++) {
    $runOnce();
}

$times = [];
for ($i = 0; $i < $iterations; $i++) {
    $start = hrtime(true);
    $runOnce();
    $times[] = (hrtime(true) - $start) / 1e6;
}

sort($times);
$count = count($times);
$result = [
    'framework' => 'php_tokenize_ort',
    'provider' => $provider,
    'php_version' => PHP_VERSION,
    'extension_version' => phpversion('onnxruntime'),
    'batch_size' => $batchSize,
    'sequence_length' => $sequenceLength,
    'iterations' => $iterations,
    'mean_ms' => array_sum($times) / $count,
    'p50_ms' => $times[(int) floor($count * 0.50)],
    'p95_ms' => $times[min($count - 1, (int) floor($count * 0.95))],
    'min_ms' => $times[0],
    'max_ms' => $times[$count - 1],
];

echo json_encode($result, JSON_THROW_ON_ERROR), "\n";
