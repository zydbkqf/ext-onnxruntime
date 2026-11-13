<?php
declare(strict_types=1);
require __DIR__ . '/common.php';

$thread = new Swoole\Thread(__DIR__ . '/thread-worker.php');
$thread->join();
$status = $thread->getExitStatus();
exit(is_array($status) ? ($status['code'] ?? 1) : (int) $status);
