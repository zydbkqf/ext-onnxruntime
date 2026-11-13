--TEST--
Session exposes dynamic dimensions as null and runs repeated inference with different batch sizes
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
$s = new OnnxRuntime\Session(__DIR__ . '/fixtures/dynamic.onnx');
echo 'inputShape=', json_encode($s->inputShapes()), "\n";
echo 'outputShape=', json_encode($s->outputShapes()), "\n";

foreach ([
    [[1.5, 2.5, 3.5]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
] as $rows) {
    $out = $s->run(['X' => OnnxRuntime\Tensor::fromArray($rows, 'float32')])['Y'];
    echo json_encode($out->shape()), ' ', json_encode($out->toArray()), "\n";
}
?>
--EXPECT--
inputShape={"X":[null,3]}
outputShape={"Y":[null,3]}
[1,3] [[1.5,2.5,3.5]]
[2,3] [[1,2,3],[4,5,6]]
[3,3] [[7,8,9],[10,11,12],[13,14,15]]
