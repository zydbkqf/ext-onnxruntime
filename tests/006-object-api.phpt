--TEST--
Tensor object API surface: metadata, clone, constructor guard
--SKIPIF--
<?php if (!extension_loaded('onnxruntime')) die('skip onnxruntime not loaded'); ?>
--FILE--
<?php
declare(strict_types=1);

$t = OnnxRuntime\Tensor::fromArray([[1, 2, 3], [4, 5, 6]]);
var_dump($t->dtype());
var_dump($t->shape());
var_dump($t->rank());

foreach (['dtype', 'shape', 'rank', 'toArray', 'fromArray', 'toOrtValue'] as $method) {
    echo $method, '=', method_exists(OnnxRuntime\Tensor::class, $method) ? 'yes' : 'no', "\n";
}
foreach (['dataTypeCode', 'dataSize', 'elementCount', 'ownership'] as $method) {
    echo $method, '=', method_exists(OnnxRuntime\Tensor::class, $method) ? 'yes' : 'no', "\n";
}

$clone = clone $t;
var_dump($clone->dtype(), $clone->shape(), $clone->toArray());

var_dump($t->__debugInfo());

try {
    new OnnxRuntime\Tensor();
    echo "no-error\n";
} catch (Error $e) {
    echo get_class($e), ': ', $e->getMessage(), "\n";
}

try {
    OnnxRuntime\Tensor::fromArray();
} catch (ArgumentCountError $e) {
    echo get_class($e), "\n";
}

try {
    OnnxRuntime\Tensor::fromArray([1], 123);
} catch (TypeError $e) {
    echo get_class($e), "\n";
}

try {
    $t->dynamic = 1;
    echo "dynamic-allowed\n";
} catch (Error $e) {
    echo get_class($e), "\n";
}

echo get_parent_class(OnnxRuntime\ConversionException::class), "\n";
?>
--EXPECT--
string(5) "int64"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(3)
}
int(2)
dtype=yes
shape=yes
rank=yes
toArray=yes
fromArray=yes
toOrtValue=yes
dataTypeCode=no
dataSize=no
elementCount=no
ownership=no
string(5) "int64"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(3)
}
array(2) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(2)
    [2]=>
    int(3)
  }
  [1]=>
  array(3) {
    [0]=>
    int(4)
    [1]=>
    int(5)
    [2]=>
    int(6)
  }
}
array(3) {
  ["dtype"]=>
  string(5) "int64"
  ["rank"]=>
  int(2)
  ["shape"]=>
  array(2) {
    [0]=>
    int(2)
    [1]=>
    int(3)
  }
}
Error: OnnxRuntime\Tensor cannot be constructed directly; use OnnxRuntime\Tensor::fromArray() or OnnxRuntime\Tensor::fromBuffer()
ArgumentCountError
TypeError
Error
OnnxRuntime\Exception
