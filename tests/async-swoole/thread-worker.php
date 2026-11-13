<?php
declare(strict_types=1);
require __DIR__ . '/common.php';

try {
    $session = make_session();
    $task = $session->runAsync(['X' => make_input(1)]);
    if (!$task->wait(5.0)) {
        throw new RuntimeException('thread: timeout');
    }
    assert_async_result($task, 'thread');
    exit(0);
} catch (Throwable $e) {
    fwrite(STDERR, 'thread:fail:' . $e->getMessage() . "\n");
    exit(1);
}
