#include "ort_value_converter.h"

#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace php_ort {
namespace {

const int64_t *shapeData(const std::vector<int64_t> &shape) {
    return shape.empty() ? nullptr : shape.data();
}

/**
 * Writes exact UTF-8 bytes into a string tensor.
 *
 * GetTensorMutableData<std::string>() is not a supported C/C++ API contract
 * for string tensors.  Use GetResizedStringTensorElementBuffer(), which
 * allocates a writable buffer of the requested byte length and does not
 * require NUL termination.  This preserves embedded NUL bytes without
 * constructing a temporary array of C strings.
 */
void fillStringTensor(Ort::Value &value,
                      const std::vector<std::string> &strings) {
    const size_t count = strings.size();
    if (count == 0) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        const std::string &element = strings[i];
        char *buffer =
            value.GetResizedStringTensorElementBuffer(i, element.size());
        if (!element.empty()) {
            if (buffer == nullptr) {
                throw ConversionError(
                    "ORT returned a null string tensor element buffer");
            }
            std::memcpy(buffer, element.data(), element.size());
        }
    }

    // Extra defense: verify ORT retained the exact byte length.  The supported
    // writer above should already guarantee this, but a mismatch must fail
    // loudly rather than silently truncating an embedded NUL.
    for (size_t i = 0; i < count; ++i) {
        if (value.GetStringTensorElementLength(i) != strings[i].size()) {
            throw ConversionError(
                "ORT string tensor truncated a string element");
        }
    }
}

}  // namespace

Ort::Value tensorToOrtValue(const Tensor &tensor,
                            const Ort::MemoryInfo &memory_info,
                            Ort::AllocatorWithDefaultOptions &allocator) {
    tensor.validate();

    if (tensor.isStringTensor()) {
        Ort::Value value = Ort::Value::CreateTensor(
            allocator, shapeData(tensor.shape), tensor.shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        fillStringTensor(value, tensor.strings);
        return value;
    }

    // Zero-element tensors are accepted with a null data pointer (verified
    // against ORT 1.28.2); CreateTensorWithDataAsOrtValue never owns the
    // buffer and never writes through it.
    void *data = const_cast<void *>(tensor.dataPointer());
    return Ort::Value::CreateTensor(memory_info, data, tensor.data.size(),
                                    shapeData(tensor.shape), tensor.shape.size(),
                                    tensor.type);
}

Tensor ortValueToTensor(const Ort::Value &value) {
    if (!value.IsTensor()) {
        throw ConversionError("OrtValue is not a tensor");
    }

    Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = info.GetElementType();
    std::vector<int64_t> shape = info.GetShape();
    size_t count = info.GetElementCount();

    Tensor tensor;
    tensor.type = type;
    tensor.shape = std::move(shape);

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        size_t total = value.GetStringTensorDataLength();
        std::vector<char> bytes(total);
        std::vector<size_t> offsets(count);

        value.GetStringTensorContent(bytes.empty() ? nullptr : bytes.data(),
                                     total,
                                     offsets.empty() ? nullptr : offsets.data(),
                                     count);

        // The first element must start at byte 0.  Otherwise the leading
        // bytes belong to no string and would be silently ignored by the
        // per-element begin/end slicing below.
        if (count > 0 && offsets[0] != 0) {
            throw ConversionError(
                "OrtValue string tensor has invalid first offset");
        }

        tensor.strings.resize(count);
        for (size_t i = 0; i < count; ++i) {
            size_t begin = offsets[i];
            size_t end = (i + 1 < count) ? offsets[i + 1] : total;
            if (begin > end || end > total) {
                throw ConversionError("OrtValue string tensor has invalid offsets");
            }
            if (begin == end) {
                tensor.strings[i].clear();
            } else {
                tensor.strings[i].assign(bytes.data() + begin, end - begin);
            }
        }
    } else {
        // elementByteSize() throws for string/unknown/unsupported dtypes, so
        // this cannot produce a zero element size and divide by zero.
        size_t element_size = elementByteSize(type);
        if (element_size == 0) {
            throw ConversionError("unsupported tensor element type");
        }
        if (count != 0 &&
            element_size > std::numeric_limits<size_t>::max() / count) {
            throw ConversionError("OrtValue tensor byte size overflows size_t");
        }

        // Use ORT's actual byte count instead of assuming count * element_size.
        // tensor.validate() below still enforces the expected size for every
        // dtype this extension supports.
#if ORT_API_VERSION >= 23
        size_t bytes = value.GetTensorSizeInBytes();
#else
        size_t bytes = count * element_size;
#endif
        tensor.data.resize(bytes);
        if (bytes != 0) {
            const void *source = value.GetTensorRawData();
            if (source == nullptr) {
                throw ConversionError("OrtValue returned a null tensor buffer");
            }
            std::memcpy(tensor.data.data(), source, bytes);
        }
    }

    tensor.validate();
    return tensor;
}

}  // namespace php_ort
