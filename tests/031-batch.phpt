--TEST--
Session::runBatch runs multiple independent input groups and reports batch entry errors
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/identity.onnx');

$batch = [
    ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]], 'float32')],
    ['X' => OnnxRuntime\Tensor::fromArray([[7, 8, 9], [10, 11, 12]],
                                          'float32')],
];
$results = $s->runBatch($batch);
echo 'batch=', json_encode(array_map(
    fn($row) => $row['Y']->toArray(), $results)), "\n";
echo 'empty=', json_encode($s->runBatch([])), "\n";
echo 'count=', count($s->runBatch($batch)), "\n";

$dynamic = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
$dynamic_results = $dynamic->runBatch([
    ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3]], 'float32')],
    ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                          'float32')],
]);
echo 'dynamic-shapes=', json_encode(array_map(
    fn($row) => $row['Y']->shape(), $dynamic_results)), "\n";

foreach ([
    'assoc' => ['first' => [
        'X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                             'float32'),
    ]],
    'hole' => [
        0 => ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                                   'float32')],
        2 => ['X' => OnnxRuntime\Tensor::fromArray([[7, 8, 9], [10, 11, 12]],
                                                   'float32')],
    ],
    'not-array' => [1],
    'missing-second' => [
        ['X' => OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]],
                                              'float32')],
        [],
    ],
] as $label => $bad) {
    try {
        $s->runBatch($bad);
        echo $label, ":none\n";
    } catch (Throwable $e) {
        echo $label, '=', get_class($e), ':', $e->getMessage(), "\n";
    }
}
?>
--EXPECT--
batch=[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
empty=[]
count=2
dynamic-shapes=[[1,3],[2,3]]
assoc=OnnxRuntime\ConversionException:batch input must be a list with sequential integer keys starting at 0
hole=OnnxRuntime\ConversionException:batch input must be a list with sequential integer keys starting at 0
not-array=OnnxRuntime\ConversionException:batch entry 0 must be an array of model inputs
missing-second=OnnxRuntime\ConversionException:batch entry 1: missing input "X"
