/*
 * PHP ONNX Runtime extension - Tensor <-> OrtValue conversion.
 *
 * This module is the only place that includes the ONNX Runtime C++ API.
 * zval conversion stays in zval_converter.cpp and never sees OrtValue types.
 */

#ifndef PHP_ORT_ORT_VALUE_CONVERTER_H
#define PHP_ORT_ORT_VALUE_CONVERTER_H

#include <onnxruntime_cxx_api.h>

#include "onnx_tensor.h"

namespace php_ort {

/**
 * Build an OrtValue from a validated Tensor.
 *
 * Numeric/bool tensors use CreateTensorWithDataAsOrtValue, so the returned
 * OrtValue borrows Tensor::dataPointer() and performs no copy.  The caller
 * must keep the source Tensor alive, and must not resize or otherwise
 * reallocate its buffers, for as long as the OrtValue is used.
 *
 * This borrow is intentionally limited to the synchronous Run path.  It must
 * not be used as-is for asynchronous execution: after the PHP method returns,
 * the source Tensor may be destroyed while an outstanding OrtValue still
 * points at its buffer.  An async implementation must take an independent
 * Tensor ownership reference or copy the payload before returning.
 *
 * String tensors use CreateTensorAsOrtValue: ORT owns the string payload
 * (there is no public external-buffer string constructor), so the returned
 * OrtValue is independent of the source Tensor.  Exact bytes are written with
 * GetResizedStringTensorElementBuffer(), which supports embedded NUL and
 * empty strings without NUL-terminated C-string construction.
 *
 * Throws ConversionError/ORT exceptions on failure.
 */
Ort::Value tensorToOrtValue(const Tensor &tensor,
                            const Ort::MemoryInfo &memory_info,
                            Ort::AllocatorWithDefaultOptions &allocator);

/**
 * Copy an OrtValue tensor into an independent, owned Tensor.
 *
 * Numeric payloads are copied into Tensor::data; string payloads are copied
 * into Tensor::strings with their exact byte length (embedded NUL preserved).
 * The returned Tensor does not reference the OrtValue after this call.
 *
 * Throws ConversionError/ORT exceptions on failure.
 */
Tensor ortValueToTensor(const Ort::Value &value);

}  // namespace php_ort

#endif  // PHP_ORT_ORT_VALUE_CONVERTER_H
