#!/usr/bin/env bash
# AddressSanitizer / LeakSanitizer / optional Valgrind runner for native Tensor
# payload ownership.  Does not build or load the PHP extension.
#
#   macOS: ASan detects address errors; LeakSanitizer is NOT supported on
#          Darwin ("detect_leaks is not supported on this platform").
#   Linux: ASan includes LSan; detect_leaks=1 is the leak gate.  Valgrind is
#          used when available (optional, slower).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cxx="${CXX:-clang++}"
asan_out="/tmp/ort_native_stress_asan"
plain_out="/tmp/ort_native_stress"

"$cxx" -std=c++17 -g -O1 -fno-omit-frame-pointer -fsanitize=address \
    -I src -I third_party/onnxruntime/include \
    tests/native_stress.cpp src/onnx_tensor.cpp -o "$asan_out"

if [ "$(uname -s)" = "Linux" ]; then
    ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 "$asan_out"
    leak_status="LSan enabled"
else
    # LeakSanitizer is unavailable on macOS; still catch OOB/UAF/double-free.
    ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 "$asan_out"
    leak_status="ASan only (LSan unsupported on macOS)"
fi

if command -v valgrind >/dev/null 2>&1; then
    "$cxx" -std=c++17 -g -O1 -I src -I third_party/onnxruntime/include \
        tests/native_stress.cpp src/onnx_tensor.cpp -o "$plain_out"
    valgrind --quiet --leak-check=full --errors-for-leak-kinds=definite \
        --error-exitcode=1 "$plain_out"
    echo "sanitizers: ASan clean; Valgrind clean; $leak_status"
else
    echo "sanitizers: ASan clean; Valgrind not installed; $leak_status"
fi
