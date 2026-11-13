#ifndef PHP_ORT_ZVAL_CONVERTER_H
#define PHP_ORT_ZVAL_CONVERTER_H

#include <optional>

extern "C" {
#include "php.h"
}

#include "onnx_tensor.h"

namespace php_ort {

struct ConvertOptions {
    /** Explicit ONNX element type; overrides PHP type inference. */
    std::optional<ONNXTensorElementDataType> dtype;
};

/** Guard against pathological recursion on hostile arrays. */
constexpr size_t kMaxArrayDepth = 64;

/**
 * Converts a PHP value into a tensor.
 *
 * Supported PHP inputs:
 *   - int / bool / string scalars (rank 0)
 *   - packed lists of scalars, nested to any depth up to kMaxArrayDepth
 *
 * The function never stores a pointer into the PHP value: everything is
 * copied into `tensor`.  On failure it throws ConversionError and leaves
 * `tensor` in an unspecified but destructible state.
 */
void zvalToTensor(zval *value, Tensor &tensor, const ConvertOptions &options);

}  // namespace php_ort

#endif  // PHP_ORT_ZVAL_CONVERTER_H
