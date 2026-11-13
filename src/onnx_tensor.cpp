#include "onnx_tensor.h"

#include <cstdint>
#include <limits>

namespace php_ort {

size_t Tensor::elementCount() const {
    if (shape.empty()) {
        return 1;
    }

    // Validate every dimension before the zero-element shortcut.  A shape such
    // as {0, -1} must be rejected even though its element count is zero.
    bool has_zero_dimension = false;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw ConversionError("negative tensor dimension");
        }
        if (dim == 0) {
            has_zero_dimension = true;
        }
    }
    if (has_zero_dimension) {
        return 0;
    }

    size_t count = 1;
    for (int64_t dim : shape) {
        uint64_t d = static_cast<uint64_t>(dim);
        if (d > std::numeric_limits<size_t>::max()) {
            throw ConversionError("tensor dimension exceeds size_t");
        }
        size_t dimension = static_cast<size_t>(d);
        if (count > std::numeric_limits<size_t>::max() / dimension) {
            throw ConversionError("tensor element count overflows size_t");
        }
        count *= dimension;
    }
    return count;
}

void Tensor::validate() const {
    if (!isSupportedElementsType(type)) {
        throw ConversionError(
            "tensor element type is undefined or unsupported by this build");
    }

    size_t count = elementCount();

    if (isStringTensor()) {
        if (!data.empty()) {
            throw ConversionError(
                "string tensor must not carry a numeric payload");
        }
        if (strings.size() != count) {
            throw ConversionError(
                "string payload length " + std::to_string(strings.size()) +
                " does not match element count " + std::to_string(count));
        }
        return;
    }

    if (!strings.empty()) {
        throw ConversionError(
            "numeric tensor must not carry a string payload");
    }

    size_t element_size = elementByteSize(type);
    if (count != 0 &&
        element_size > std::numeric_limits<size_t>::max() / count) {
        throw ConversionError("tensor byte size overflows size_t");
    }
    if (data.size() != count * element_size) {
        throw ConversionError(
            "numeric payload size " + std::to_string(data.size()) +
            " does not match shape/dtype (expected " +
            std::to_string(count * element_size) + " bytes)");
    }
}

size_t elementByteSize(ONNXTensorElementDataType type) {    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            throw ConversionError("string tensors have no fixed element size");
        default:
            throw ConversionError("unsupported tensor element type");
    }
}

const char *elementTypeName(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            return "undefined";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
            return "float8e4m3fn";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
            return "float8e4m3fnuz";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
            return "float8e5m2";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
            return "float8e5m2fnuz";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
            return "uint4";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
            return "int4";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
            return "float4e2m1";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
            return "uint2";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
            return "int2";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0:
            return "float8e8m0";
        default:
            return "unknown";
    }
}

bool isSupportedElementsType(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return true;
        default:
            return false;
    }
}

bool elementTypeFromName(const std::string &name,
                         ONNXTensorElementDataType *out) {
    if (out == nullptr) {
        return false;
    }
    if (name == "float32" || name == "float") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (name == "float64" || name == "double") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    } else if (name == "int64") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (name == "int32") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    } else if (name == "int16") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    } else if (name == "int8") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    } else if (name == "uint8") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    } else if (name == "bool") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    } else if (name == "string") {
        *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    } else {
        return false;
    }
    return true;
}

const char *supportedDtypeNames() {
    return "float32, float64, int64, int32, int16, int8, uint8, bool, string";
}

}  // namespace php_ort
