#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
php="${PHP_BIN:-php}"
args=(-d extension="$root/modules/onnxruntime.so")
if [[ -n "${SWOOLE_EXT:-}" ]]; then
  args+=(-d extension="$SWOOLE_EXT")
fi
for script in process thread coroutine fiber; do
  echo "== $script =="
  if [[ "$script" == thread ]] && ! "$php" "${args[@]}" -r 'exit(class_exists("Swoole\\Thread") ? 0 : 1);'; then
    echo "thread:skipped (Swoole\\Thread unavailable)"
    continue
  fi
  "$php" "${args[@]}" "tests/async-swoole/$script.php"
done
