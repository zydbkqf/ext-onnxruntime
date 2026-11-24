#!/usr/bin/env bash
# Build, load, and smoke-test the extension.
#
# Usage:
#   bash tests/build-smoke.sh                 # NuGet CPU package (default)
#   bash tests/build-smoke.sh /opt/ort-sdk    # explicit SDK directory
#   bash tests/build-smoke.sh --enable-debug  # any other --flag is passed to configure
#   bash tests/build-smoke.sh --with-cuda13   # extra configure flags
#
# With no directory the configure script downloads the CPU package from
# nuget.org.  A directory is forwarded as --with-onnxruntime=DIR; any further
# arguments (for example --with-cuda13) are passed through to configure.
# ONNXRUNTIME_ROOT is accepted as a fallback for the directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHP_BIN="${PHP_BIN:-php}"
JOBS="${JOBS:-8}"
MODULE="$ROOT/modules/onnxruntime.so"

if ! PHP_BIN_PATH="$(command -v "$PHP_BIN")"; then
    echo "build-smoke: PHP executable not found: $PHP_BIN" >&2
    exit 2
fi
echo "build-smoke: PHP $("$PHP_BIN_PATH" -r 'echo PHP_VERSION;') at $PHP_BIN_PATH"
PHP_BIN="$PHP_BIN_PATH"

ORT_ROOT=""
CONFIGURE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --with-onnxruntime=*)
            if [[ -n "$ORT_ROOT" ]]; then
                echo "build-smoke: SDK directory was given twice: $ORT_ROOT and ${arg#*=}" >&2
                exit 2
            fi
            ORT_ROOT="${arg#*=}"
            ;;
        --*)
            CONFIGURE_ARGS+=("$arg")
            ;;
        *)
            if [[ -z "$ORT_ROOT" ]]; then
                ORT_ROOT="$arg"
            else
                echo "build-smoke: unexpected extra argument: $arg" >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "$ORT_ROOT" ]]; then
    ORT_ROOT="${ONNXRUNTIME_ROOT:-}"
fi
if [[ -n "$ORT_ROOT" ]]; then
    if [[ ! -d "$ORT_ROOT" ]]; then
        echo "build-smoke: ONNX Runtime SDK directory does not exist: $ORT_ROOT" >&2
        exit 2
    fi
    ORT_ROOT="$(cd "$ORT_ROOT" && pwd)"
    # SIP-protected /usr/bin/make strips DYLD_LIBRARY_PATH before running PHP,
    # which made the directory build report every test as skipped.  Bake the
    # SDK lib directory into the module rpath instead.
    if [[ -d "$ORT_ROOT/lib" ]]; then
        export ONNXRUNTIME_RPATH="$ORT_ROOT/lib"
    else
        export ONNXRUNTIME_RPATH="$ORT_ROOT"
    fi
fi

cd "$ROOT"

phpize --clean
phpize
if [[ -n "$ORT_ROOT" ]]; then
    ./configure --with-onnxruntime="$ORT_ROOT" ${CONFIGURE_ARGS[@]+"${CONFIGURE_ARGS[@]}"}
else
    ./configure ${CONFIGURE_ARGS[@]+"${CONFIGURE_ARGS[@]}"}
fi
make -j"$JOBS"

if [[ "$(uname -s)" == "Darwin" ]]; then
    codesign --force --sign - "$MODULE"
fi

# An explicit SDK directory needs a runtime search path; NuGet builds already
# carry an rpath to the downloaded SDK.
if [[ -n "$ORT_ROOT" ]]; then
    export DYLD_LIBRARY_PATH="$ORT_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$ORT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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

PHP_EXECUTABLE="$PHP_BIN" NO_INTERACTION=1 make test PHP_EXECUTABLE="$PHP_BIN"
