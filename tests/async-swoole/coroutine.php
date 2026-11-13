<?php
declare(strict_types=1);
require __DIR__ . '/common.php';

Swoole\Coroutine\run(function (): void {
    $session = make_session();
    $tasks = [];
    for ($i = 0; $i < 4; ++$i) {
        $tasks[$i] = $session->runAsync(['X' => make_input($i * 10)]);
    }
    // Yield so all submissions run concurrently on native workers.
    Swoole\Coroutine::sleep(0.05);
    foreach ($tasks as $i => $task) {
        $deadline = microtime(true) + 5.0;
        while (!$task->isReady()) {
            if (microtime(true) >= $deadline) {
                throw new RuntimeException("coroutine:$i:timeout");
            }
            Swoole\Coroutine::sleep(0.005);
        }
        assert_async_result($task, "coroutine:$i");
    }
});
