--TEST--
Dtype inference: int/float mixing, bool, string, empty dims, large arrays
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
function show(OnnxRuntime\Tensor $t): void {
    echo $t->dtype(), ' ', json_encode($t->shape()), ' ',
        json_encode($t->toArray(), JSON_PRESERVE_ZERO_FRACTION), "\n";
}

// int + float widens to float64
show(OnnxRuntime\Tensor::fromArray([1, 2.5]));
show(OnnxRuntime\Tensor::fromArray([[1, 2.5], [3, 4]]));

// bool only
show(OnnxRuntime\Tensor::fromArray([true, false, true]));

// string only
show(OnnxRuntime\Tensor::fromArray(['a', 'b']));

// empty inner dimensions keep the rank
show(OnnxRuntime\Tensor::fromArray([[], []]));

// 100 x 100 int matrix
$rows = [];
for ($i = 0; $i < 100; $i++) {
    $row = [];
    for ($j = 0; $j < 100; $j++) {
        $row[] = $i * 100 + $j;
    }
    $rows[] = $row;
}
$t = OnnxRuntime\Tensor::fromArray($rows);
$flat = $t->toArray();
$sum = 0;
foreach ($flat as $row) {
    $sum += array_sum($row);
}
echo $t->dtype(), ' ', json_encode($t->shape()), ' ', $sum, "\n";
?>
--EXPECT--
float64 [2] [1.0,2.5]
float64 [2,2] [[1.0,2.5],[3.0,4.0]]
bool [3] [true,false,true]
string [2] ["a","b"]
float32 [2,0] [[],[]]
int64 [100,100] 49995000
