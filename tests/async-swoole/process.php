<?php
declare(strict_types=1);
require __DIR__ . '/common.php';

$process = new Swoole\Process(function (Swoole\Process $worker): void {
    try {
        $session = make_session();
        $task = $session->runAsync(['X' => make_input(1)]);
        if (!$task->wait(5.0)) {
            throw new RuntimeException('process: timeout');
        }
        assert_async_result($task, 'process');
        $worker->exit(0);
    } catch (Throwable $e) {
        fwrite(STDERR, 'process:fail:' . $e->getMessage() . "\n");
        $worker->exit(1);
    }
}, false, 0, false);

$process->start();
$status = $process->wait();
exit($status['code'] ?? 1);
