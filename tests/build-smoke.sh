#!/usr/bin/env bash
# Build, load, and smoke-test the extension against an ONNX Runtime SDK.
#
# Usage:
#   bash tests/build-smoke.sh /opt/onnxruntime-linux-x64-1.28.2
#
# The directory must be the extracted ONNX Runtime prebuilt package root
# containing include/ and lib/. ONNXRUNTIME_ROOT is accepted as a fallback.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHP_BIN="${PHP_BIN:-php}"
JOBS="${JOBS:-8}"
MODULE="$ROOT/modules/onnxruntime.so"
ORT_ROOT="${1:-${ONNXRUNTIME_ROOT:-}}"

if [[ -z "$ORT_ROOT" ]]; then
    echo "build-smoke: pass the ONNX Runtime SDK root, e.g. bash tests/build-smoke.sh /opt/onnxruntime-linux-x64-1.28.2" >&2
    echo "build-smoke: or set ONNXRUNTIME_ROOT" >&2
    exit 2
fi

cd "$ROOT"

phpize --clean
phpize
./configure --with-onnxruntime="$ORT_ROOT"
make -j"$JOBS"

if [[ "$(uname -s)" == "Darwin" ]]; then
    codesign --force --sign - "$MODULE"
fi

# Let a locally downloaded ORT SDK satisfy the runtime loader even when no
# rpath was embedded at configure time.
export DYLD_LIBRARY_PATH="$ORT_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ORT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PHP_BIN" -n -d "extension=$MODULE" --ri onnxruntime >/dev/null
BUILD_SMOKE_FIXTURE="$ROOT/tests/fixtures/identity.onnx" \
"$PHP_BIN" -n -d "extension=$MODULE" -r '
$t = OnnxRuntime\Tensor::fromArray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], "float32");
$s = new OnnxRuntime\Session(getenv("BUILD_SMOKE_FIXTURE"));
$out = $s->run(["X" => $t]);
if ($out["Y"]->shape() !== [2, 3]) {
    fwrite(STDERR, "unexpected output shape\n");
    exit(1);
}
if ($out["Y"]->toArray() !== [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]) {
    fwrite(STDERR, "unexpected identity output\n");
    exit(1);
}
echo "build-smoke-ok\n";
'

NO_INTERACTION=1 make test
