<?php
declare(strict_types=1);
require __DIR__ . '/common.php';

$box = new ArrayObject();
$fiber = new Fiber(function () use ($box): void {
    $session = make_session();
    $box['task'] = $session->runAsync(['X' => make_input(7)]);
    Fiber::suspend();
});

$fiber->start();
$task = $box['task'] ?? null;
if (!$task instanceof OnnxRuntime\Task || !$task->wait(5.0)) {
    throw new RuntimeException('fiber: task did not finish');
}
assert_async_result($task, 'fiber');
