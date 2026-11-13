#include "zval_converter.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace php_ort {
namespace {

struct ScalarFlags {
    bool has_int = false;
    bool has_double = false;
    bool has_bool = false;
    bool has_string = false;
};

bool isListArray(zend_array *ht) {
#if PHP_VERSION_ID >= 80100
    return zend_array_is_list(ht);
#else
    // zend_array_is_list() was added in PHP 8.1.  Keep the same contract on
    // PHP 8.0: every key must be the next zero-based integer.
    zend_ulong expected = 0;
    zend_ulong index;
    zend_string *key;
    zval *value;
    ZEND_HASH_FOREACH_KEY_VAL(ht, index, key, value) {
        if (key != nullptr || index != expected) {
            return false;
        }
        ++expected;
    } ZEND_HASH_FOREACH_END();
    return true;
#endif
}

std::string shapeToString(const std::vector<int64_t> &shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

std::string childPath(const std::string &path, size_t index) {
    return path + "[" + std::to_string(index) + "]";
}

/** Readable, round-trippable enough for error messages. */
std::string formatDouble(double value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.17g", value);
    return std::string(buffer);
}

void recordScalarType(zval *value, ScalarFlags &flags,
                      const std::string &path) {
    ZVAL_DEREF(value);
    switch (Z_TYPE_P(value)) {
        case IS_LONG:
            flags.has_int = true;
            return;
        case IS_DOUBLE:
            flags.has_double = true;
            return;
        case IS_TRUE:
        case IS_FALSE:
            flags.has_bool = true;
            return;
        case IS_STRING:
            flags.has_string = true;
            return;
        default:
            throw ConversionError(
                path + ": unsupported PHP type \"" +
                zend_zval_type_name(value) +
                "\" in tensor data (expected int, float, bool, string or array)");
    }
}

/**
 * First pass: validates the nested array and derives its shape.
 *
 * A shape is returned for every subtree so that siblings can be compared;
 * irregular sub-arrays are rejected instead of being padded.
 */
std::vector<int64_t> inspectValue(zval *value, size_t depth,
                                  ScalarFlags &flags,
                                  const std::string &path) {
    ZVAL_DEREF(value);

    if (Z_TYPE_P(value) != IS_ARRAY) {
        recordScalarType(value, flags, path);
        return std::vector<int64_t>();
    }

    if (depth >= kMaxArrayDepth) {
        throw ConversionError(path + ": array nesting exceeds " +
                              std::to_string(kMaxArrayDepth) + " levels");
    }

    zend_array *ht = Z_ARRVAL_P(value);
    if (!isListArray(ht)) {
        throw ConversionError(
            path + ": expected a list with sequential integer keys starting "
                   "at 0; associative or sparse arrays are not supported");
    }

    size_t count = zend_hash_num_elements(ht);
    std::vector<int64_t> child_shape;
    bool children_are_arrays = false;
    size_t index = 0;
    zval *item;

    ZEND_HASH_FOREACH_VAL(ht, item) {
        ZVAL_DEREF(item);
        std::string path_here = childPath(path, index);
        bool is_array = Z_TYPE_P(item) == IS_ARRAY;

        if (index == 0) {
            children_are_arrays = is_array;
        } else if (children_are_arrays != is_array) {
            throw ConversionError(
                path_here +
                ": nested array mixes sub-arrays and scalar values");
        }

        if (is_array) {
            std::vector<int64_t> sub =
                inspectValue(item, depth + 1, flags, path_here);
            if (index == 0) {
                child_shape = std::move(sub);
            } else if (sub != child_shape) {
                throw ConversionError(
                    path_here + ": irregular nested array; expected shape " +
                    shapeToString(child_shape) + ", got " +
                    shapeToString(sub));
            }
        } else {
            recordScalarType(item, flags, path_here);
        }
        ++index;
    }
    ZEND_HASH_FOREACH_END();

    std::vector<int64_t> shape;
    shape.reserve(child_shape.size() + 1);
    shape.push_back(static_cast<int64_t>(count));
    shape.insert(shape.end(), child_shape.begin(), child_shape.end());
    return shape;
}

ONNXTensorElementDataType inferElementType(const ScalarFlags &flags) {
    if (flags.has_string &&
        (flags.has_int || flags.has_double || flags.has_bool)) {
        throw ConversionError(
            "cannot infer a single tensor dtype: array mixes string and "
            "numeric/bool values; pass an explicit dtype");
    }
    if (flags.has_string) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    if (flags.has_bool && (flags.has_int || flags.has_double)) {
        throw ConversionError(
            "cannot infer a single tensor dtype: array mixes bool and "
            "numeric values; pass an explicit dtype");
    }
    if (flags.has_bool) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    if (flags.has_double) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    if (flags.has_int) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    // Empty array: no element to inspect.  FLOAT is the ONNX default choice
    // for an empty tensor; callers that need another type pass a dtype.
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

template <typename T>
void appendRaw(Tensor &tensor, T value) {
    size_t offset = tensor.data.size();
    tensor.data.resize(offset + sizeof(T));
    std::memcpy(tensor.data.data() + offset, &value, sizeof(T));
}

void checkIntRange(int64_t value, ONNXTensorElementDataType type,
                   const std::string &path) {
    int64_t lo = 0;
    int64_t hi = 0;
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            lo = std::numeric_limits<int8_t>::min();
            hi = std::numeric_limits<int8_t>::max();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            lo = 0;
            hi = std::numeric_limits<uint8_t>::max();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            lo = std::numeric_limits<int16_t>::min();
            hi = std::numeric_limits<int16_t>::max();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            lo = std::numeric_limits<int32_t>::min();
            hi = std::numeric_limits<int32_t>::max();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return;
        default:
            throw ConversionError(path + ": internal error: non-integer dtype");
    }
    if (value < lo || value > hi) {
        throw ConversionError(path + ": value " + std::to_string(value) +
                              " is out of range for " + elementTypeName(type));
    }
}

double requireDouble(zval *value, const std::string &path) {
    ZVAL_DEREF(value);
    switch (Z_TYPE_P(value)) {
        case IS_LONG:
            return static_cast<double>(Z_LVAL_P(value));
        case IS_DOUBLE:
            return Z_DVAL_P(value);
        case IS_TRUE:
            return 1.0;
        case IS_FALSE:
            return 0.0;
        default:
            throw ConversionError(path + ": cannot convert " +
                                  zend_zval_type_name(value) +
                                  " to a floating point tensor element");
    }
}

int64_t requireInteger(zval *value, const std::string &path) {
    ZVAL_DEREF(value);
    switch (Z_TYPE_P(value)) {
        case IS_LONG:
            return static_cast<int64_t>(Z_LVAL_P(value));
        case IS_TRUE:
            return 1;
        case IS_FALSE:
            return 0;
        case IS_DOUBLE: {
            double d = Z_DVAL_P(value);
            if (!std::isfinite(d)) {
                throw ConversionError(path + ": non-finite float cannot be "
                                              "converted to an integer tensor");
            }
            if (std::floor(d) != d) {
                throw ConversionError(
                    path + ": value " + formatDouble(d) +
                    " is not an integer and would lose precision when "
                    "converted to an integer tensor");
            }
            if (d < -9223372036854775808.0 || d >= 9223372036854775808.0) {
                throw ConversionError(path + ": value " + formatDouble(d) +
                                      " is out of int64 range");
            }
            return static_cast<int64_t>(d);
        }
        default:
            throw ConversionError(path + ": cannot convert " +
                                  zend_zval_type_name(value) +
                                  " to an integer tensor element");
    }
}

void appendScalar(zval *value, Tensor &tensor, const std::string &path) {
    ZVAL_DEREF(value);

    switch (tensor.type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            // Policy: float tensors may lose precision (normal for ML models),
            // but a finite double must never silently overflow to infinity.
            double d = requireDouble(value, path);
            float f = static_cast<float>(d);
            if (std::isfinite(d) && !std::isfinite(f)) {
                throw ConversionError(path + ": value " + formatDouble(d) +
                                      " overflows float32");
            }
            appendRaw(tensor, f);
            return;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            // Integer -> double beyond 2^53 loses precision; accepted by
            // policy, never an overflow for a 64-bit PHP int.
            appendRaw(tensor, requireDouble(value, path));
            return;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            int64_t v = requireInteger(value, path);
            checkIntRange(v, tensor.type, path);
            switch (tensor.type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    appendRaw(tensor, static_cast<int8_t>(v));
                    return;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    appendRaw(tensor, static_cast<uint8_t>(v));
                    return;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                    appendRaw(tensor, static_cast<int16_t>(v));
                    return;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    appendRaw(tensor, static_cast<int32_t>(v));
                    return;
                default:
                    appendRaw(tensor, v);
                    return;
            }
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            bool b = false;
            switch (Z_TYPE_P(value)) {
                case IS_TRUE:
                    b = true;
                    break;
                case IS_FALSE:
                    b = false;
                    break;
                case IS_LONG: {
                    zend_long v = Z_LVAL_P(value);
                    if (v != 0 && v != 1) {
                        throw ConversionError(
                            path + ": value " + std::to_string(v) +
                            " cannot be converted to bool without loss; only "
                            "0 and 1 are accepted");
                    }
                    b = v == 1;
                    break;
                }
                case IS_DOUBLE: {
                    double d = Z_DVAL_P(value);
                    if (d != 0.0 && d != 1.0) {
                        throw ConversionError(
                            path + ": value " + formatDouble(d) +
                            " cannot be converted to bool without loss; only "
                            "0.0 and 1.0 are accepted");
                    }
                    b = d == 1.0;
                    break;
                }
                default:
                    throw ConversionError(
                        path + ": cannot convert " + zend_zval_type_name(value) +
                        " to a bool tensor element");
            }
            appendRaw(tensor, static_cast<uint8_t>(b ? 1 : 0));
            return;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
            if (Z_TYPE_P(value) != IS_STRING) {
                throw ConversionError(
                    path + ": cannot convert " + zend_zval_type_name(value) +
                    " to a string tensor element; values must already be "
                    "PHP strings");
            }
            tensor.strings.emplace_back(Z_STRVAL_P(value), Z_STRLEN_P(value));
            return;
        }
        default:
            throw ConversionError(path + ": unsupported tensor element type");
    }
}

/** Second pass: copies values in row-major order after the shape is known. */
void flattenValue(zval *value, size_t depth, Tensor &tensor,
                  const std::vector<int64_t> &shape,
                  const std::string &path) {
    ZVAL_DEREF(value);

    if (depth == shape.size()) {
        appendScalar(value, tensor, path);
        return;
    }

    if (Z_TYPE_P(value) != IS_ARRAY) {
        throw ConversionError(path + ": expected a nested array for dimension " +
                              std::to_string(depth) + ", got " +
                              zend_zval_type_name(value));
    }

    zend_array *ht = Z_ARRVAL_P(value);
    size_t count = zend_hash_num_elements(ht);
    size_t expected = static_cast<size_t>(shape[depth]);
    if (count != expected) {
        throw ConversionError(path + ": expected " + std::to_string(expected) +
                              " elements at dimension " + std::to_string(depth) +
                              ", got " + std::to_string(count));
    }

    size_t index = 0;
    zval *item;
    ZEND_HASH_FOREACH_VAL(ht, item) {
        flattenValue(item, depth + 1, tensor, shape, childPath(path, index));
        ++index;
    } ZEND_HASH_FOREACH_END();
}

}  // namespace

void zvalToTensor(zval *value, Tensor &tensor, const ConvertOptions &options) {
    if (value == nullptr) {
        throw ConversionError("value: expected a PHP value, got null pointer");
    }

    ScalarFlags flags;
    std::vector<int64_t> shape = inspectValue(value, 0, flags, "value");

    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (options.dtype.has_value()) {
        type = *options.dtype;
    } else {
        type = inferElementType(flags);
    }

    if (!isSupportedElementsType(type)) {
        throw ConversionError(std::string("dtype \"") + elementTypeName(type) +
                              "\" is not supported by this extension build");
    }

    Tensor result;
    result.type = type;
    result.shape = std::move(shape);

    size_t reserved = result.elementCount();
    if (!result.isStringTensor()) {
        size_t element_size = elementByteSize(type);
        if (reserved != 0 &&
            element_size > std::numeric_limits<size_t>::max() / reserved) {
            throw ConversionError("tensor payload size overflows size_t");
        }
        result.data.reserve(reserved * element_size);
    } else {
        result.strings.reserve(reserved);
    }

    flattenValue(value, 0, result, result.shape, "value");
    result.validate();

    tensor = std::move(result);
}

}  // namespace php_ort
