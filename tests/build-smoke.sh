#!/usr/bin/env bash
# Build, load, and smoke-test the extension against an ONNX Runtime SDK.
#
# External SDK:
#   ONNXRUNTIME_INCLUDEDIR=/opt/ort/include \
#   ONNXRUNTIME_LIBDIR=/opt/ort/lib bash tests/build-smoke.sh
#
# Repository development SDK (macOS arm64 only):
#   ONNXRUNTIME_USE_VENDORED=yes bash tests/build-smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHP_BIN="${PHP_BIN:-php}"
JOBS="${JOBS:-8}"
MODULE="$ROOT/modules/onnxruntime.so"

if [[ -z "${ONNXRUNTIME_INCLUDEDIR:-}" &&
      -z "${ONNXRUNTIME_LIBDIR:-}" &&
      -z "${ONNXRUNTIME_ROOT:-}" &&
      "${ONNXRUNTIME_USE_VENDORED:-no}" != "yes" ]]; then
    echo "build-smoke: set ONNXRUNTIME_INCLUDEDIR/ONNXRUNTIME_LIBDIR (or ONNXRUNTIME_ROOT)" >&2
    echo "build-smoke: or pass ONNXRUNTIME_USE_VENDORED=yes for the repository development SDK" >&2
    exit 2
fi

cd "$ROOT"

phpize --clean
phpize
./configure --enable-onnxruntime
make -j"$JOBS"

if [[ "$(uname -s)" == "Darwin" ]]; then
    codesign --force --sign - "$MODULE"
fi

# Let a locally downloaded ORT SDK satisfy the runtime loader even when no
# rpath was embedded at configure time.
if [[ -n "${ONNXRUNTIME_LIBDIR:-}" ]]; then
    export DYLD_LIBRARY_PATH="${ONNXRUNTIME_LIBDIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ONNXRUNTIME_LIBDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
