<?php
declare(strict_types=1);

const ONNX_MODEL = __DIR__ . '/../fixtures/identity.onnx';

function assert_async_result(OnnxRuntime\Task $task, string $label): void
{
    if ($task->getState() !== 'completed') {
        throw new RuntimeException($label . ': state=' . $task->getState());
    }
    $out = $task->getResult();
    if (!isset($out['Y']) || $out['Y']->shape() !== [2, 3]) {
        throw new RuntimeException($label . ': unexpected output');
    }
    echo $label, ":ok\n";
}

function make_session(): OnnxRuntime\Session
{
    return new OnnxRuntime\Session(ONNX_MODEL);
}

function make_input(int $base): OnnxRuntime\Tensor
{
    return OnnxRuntime\Tensor::fromArray(
        [[$base, $base + 1, $base + 2], [$base + 3, $base + 4, $base + 5]],
        'float32'
    );
}
