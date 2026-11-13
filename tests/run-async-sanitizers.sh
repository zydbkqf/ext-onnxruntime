#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
cxx="${CXX:-clang++}"
out="${OUT:-/tmp/ort_native_async_stress_asan}"
ort_root="${ONNXRUNTIME_ROOT:-$root/third_party/onnxruntime}"
"$cxx" -std=c++17 -g -O1 -fno-omit-frame-pointer -fsanitize=address \
  -I src -I "$ort_root/include" \
  tests/native_async_stress.cpp \
  src/async_engine.cpp src/session.cpp src/session_options.cpp \
  src/io_binding.cpp src/ort_value_converter.cpp src/onnx_tensor.cpp \
  src/runtime.cpp \
  src/ep_device.cpp src/ort_environment.cpp \
  -L "$ort_root/lib" -lonnxruntime \
  -Wl,-rpath,"$ort_root/lib" -o "$out"
ASAN_OPTIONS="detect_leaks=${ASAN_DETECT_LEAKS:-0}:abort_on_error=1" "$out" "$root/tests/fixtures/identity.onnx"
