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
        /* Added in ONNX Runtime 1.20 (ORT_API_VERSION 20). */
#if ORT_API_VERSION >= 20
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
            return "uint4";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
            return "int4";
#endif
        /* Added in ONNX Runtime 1.24 (ORT_API_VERSION 24). */
#if ORT_API_VERSION >= 24
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
            return "float4e2m1";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
            return "uint2";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
            return "int2";
#endif
        /* Added in ONNX Runtime 1.27 (ORT_API_VERSION 27). */
#if ORT_API_VERSION >= 27
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0:
            return "float8e8m0";
#endif
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

namespace {

/** Widens one raw element to double.  Never used for STRING. */
inline double loadScalar(const uint8_t *source, ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float value;
            std::memcpy(&value, source, sizeof(value));
            return static_cast<double>(value);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double value;
            std::memcpy(&value, source, sizeof(value));
            return value;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            int64_t value;
            std::memcpy(&value, source, sizeof(value));
            return static_cast<double>(value);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t value;
            std::memcpy(&value, source, sizeof(value));
            return static_cast<double>(value);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            int16_t value;
            std::memcpy(&value, source, sizeof(value));
            return static_cast<double>(value);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return static_cast<double>(static_cast<int8_t>(*source));
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return static_cast<double>(*source);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return (*source != 0) ? 1.0 : 0.0;
        default:
            throw ConversionError(
                "unsupported buffer element type for raw bytes");
    }
}

inline double clampTo(double value, double low, double high) {
    if (!(value >= low)) {  // also catches NaN
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

/** Narrows a double back to one raw element.  Out-of-range values saturate. */
inline void storeScalar(uint8_t *target, ONNXTensorElementDataType type,
                        double value) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float narrowed = static_cast<float>(value);
            std::memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            std::memcpy(target, &value, sizeof(value));
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            /* The upper bound cannot be written as a double: INT64_MAX rounds
             * up to 2^63, and narrowing any value >= 2^63 is undefined
             * behaviour (x86-64 cvttsd2si yields INT64_MIN there, arm64
             * saturates).  Compare against 0x1p63 before the cast. */
            int64_t narrowed;
            if (!(value >= -9223372036854775808.0)) {  // also catches NaN
                narrowed = std::numeric_limits<int64_t>::min();
            } else if (value >= 0x1p63) {
                narrowed = std::numeric_limits<int64_t>::max();
            } else {
                narrowed = static_cast<int64_t>(value);
            }
            std::memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t narrowed = static_cast<int32_t>(
                clampTo(value, -2147483648.0, 2147483647.0));
            std::memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            int16_t narrowed =
                static_cast<int16_t>(clampTo(value, -32768.0, 32767.0));
            std::memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            *target = static_cast<uint8_t>(static_cast<int8_t>(
                clampTo(value, -128.0, 127.0)));
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            *target = static_cast<uint8_t>(clampTo(value, 0.0, 255.0));
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            *target = (value != 0.0) ? 1 : 0;
            break;
        default:
            throw ConversionError(
                "unsupported buffer element type for raw bytes");
    }
}

/** Product of shape[from..], 1 for an empty tail, checked against overflow. */
size_t tailProduct(const std::vector<int64_t> &shape, size_t from) {
    size_t count = 1;
    for (size_t i = from; i < shape.size(); ++i) {
        size_t dimension = static_cast<size_t>(shape[i]);
        if (dimension != 0 &&
            count > std::numeric_limits<size_t>::max() / dimension) {
            throw ConversionError("tensor element count overflows size_t");
        }
        count *= dimension;
    }
    return count;
}

/** Resolves a scalar-or-per-channel vector against channelCount. */
void resolvePerChannel(const std::vector<double> &values, size_t channels,
                       const char *what, std::vector<double> &out,
                       double fallback) {
    if (values.empty()) {
        out.assign(channels, fallback);
        return;
    }
    if (values.size() == 1) {
        out.assign(channels, values[0]);
        return;
    }
    if (values.size() != channels) {
        throw ConversionError(std::string(what) + " has " +
                              std::to_string(values.size()) +
                              " entries but the channel count is " +
                              std::to_string(channels));
    }
    out = values;
}

}  // namespace

Tensor tensorFromBuffer(const void *buffer, size_t buffer_size,
                        const std::vector<int64_t> &shape,
                        const BufferOptions &options) {
    if (options.source_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
        options.target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        throw ConversionError(
            "fromBuffer() does not accept string tensors; raw string payloads "
            "are variable length and need their own representation");
    }
    if (!isSupportedElementsType(options.source_type) ||
        !isSupportedElementsType(options.target_type)) {
        throw ConversionError(
            "fromBuffer() supports only numeric and bool element types");
    }

    Tensor tensor;
    tensor.shape = shape;
    tensor.type = options.target_type;

    size_t count = tensor.elementCount();
    size_t source_size = elementByteSize(options.source_type);
    size_t target_size = elementByteSize(options.target_type);
    if (count != 0 &&
        source_size > std::numeric_limits<size_t>::max() / count) {
        throw ConversionError("tensor byte size overflows size_t");
    }
    size_t expected = count * source_size;
    if (buffer_size != expected) {
        throw ConversionError(
            "buffer length " + std::to_string(buffer_size) +
            " does not match shape/dtype (expected " + std::to_string(expected) +
            " bytes for " + std::to_string(count) + " x " +
            std::to_string(source_size) + " bytes)");
    }

    bool normalising = options.scale != 1.0 || !options.mean.empty() ||
                       !options.std.empty();
    if (normalising &&
        (options.target_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         options.target_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
        throw ConversionError(
            "scale/mean/std require a float32 or float64 target dtype");
    }

    /* The channel axis follows ONNX's shape conventions:
     *   - rank 3: this library's batch-less [C,H,W], so channels are dim 0.
     *   - rank >= 4: ONNX [N,C,...] ([N,C,H,W], [N,C,D,H,W], ...), so
     *     channels are dim 1 and the stride is the product of the axes
     *     behind it (H*W for 4D, D*H*W for 5D, ...).
     *   - rank 2: the last axis is the channel axis, so values interleave.
     *   - rank 0/1: exactly one channel.
     * 'hwc' is limited to rank 3/4 and picks the same dim for those ranks,
     * so 'hwc' and 'chw'/'none' agree on the channel axis for one shape. */
    size_t channels = 1;
    bool perChannel = false;
    size_t channel_stride = 1;
    const size_t rank = shape.size();
    if (options.hwc) {
        if (rank != 3 && rank != 4) {
            throw ConversionError(
                "layout 'hwc' requires a rank-3 [H,W,C]/[C,H,W] or rank-4 "
                "[N,H,W,C]/[N,C,H,W] shape");
        }
        channels = static_cast<size_t>(shape[rank - 3]);
        perChannel = true;
        channel_stride = 0;  // the hwc loops carry the channel index
    } else if (rank == 3) {
        channels = static_cast<size_t>(shape[0]);
        perChannel = true;
        channel_stride = tailProduct(shape, 1);
    } else if (rank >= 4) {
        channels = static_cast<size_t>(shape[1]);
        perChannel = true;
        channel_stride = tailProduct(shape, 2);
    } else if (rank == 2) {
        channels = static_cast<size_t>(shape[1]);
        perChannel = true;
    }

    // Both vectors are always materialised (scalar broadcast when the caller
    // passed nothing), so the loops below can index them unconditionally.
    std::vector<double> mean;
    std::vector<double> stddev;
    resolvePerChannel(options.mean, perChannel ? channels : 1, "mean", mean,
                      0.0);
    resolvePerChannel(options.std, perChannel ? channels : 1, "std", stddev,
                      1.0);

    if (count != 0 &&
        target_size > std::numeric_limits<size_t>::max() / count) {
        throw ConversionError("tensor byte size overflows size_t");
    }
    tensor.data.resize(count * target_size);
    const uint8_t *source = static_cast<const uint8_t *>(buffer);
    uint8_t *target = tensor.data.empty() ? nullptr : tensor.data.data();

    /* Same dtype and no arithmetic: every element is byte-identical, so the
     * payload is moved without a double round trip.  int64 above 2^53 and
     * any future wide type must not be widened, and re-ordering alone never
     * changes a value. */
    const bool copy_raw =
        options.source_type == options.target_type && !normalising;

    // Fast path: identical layout and dtype, no element-wise transform.
    if (!options.hwc && copy_raw) {
        if (count != 0) {
            std::memcpy(target, source, count * target_size);
        }
        return tensor;
    }

    double scale = options.scale;
    if (channel_stride == 0) {
        // Only the hwc path leaves it unset; a zero-sized block also implies
        // an empty payload, which is handled above.
        channel_stride = 1;
    }

    if (!options.hwc) {
        for (size_t i = 0; i < count; ++i) {
            size_t channel = perChannel ? ((i / channel_stride) % channels) : 0;
            double value = loadScalar(source + i * source_size, options.source_type);
            if (normalising) {
                value = (value * scale - mean[channel]) / stddev[channel];
            }
            storeScalar(target + i * target_size, options.target_type, value);
        }
        return tensor;
    }

    if (shape.size() == 3) {
        size_t c_dim = static_cast<size_t>(shape[0]);
        size_t h_dim = static_cast<size_t>(shape[1]);
        size_t w_dim = static_cast<size_t>(shape[2]);
        size_t plane = h_dim * w_dim;
        for (size_t channel = 0; channel < c_dim; ++channel) {
            for (size_t h = 0; h < h_dim; ++h) {
                for (size_t w = 0; w < w_dim; ++w) {
                    size_t source_index = (h * w_dim + w) * c_dim + channel;
                    size_t target_index = channel * plane + h * w_dim + w;
                    if (copy_raw) {
                        std::memcpy(target + target_index * target_size,
                                    source + source_index * source_size,
                                    target_size);
                    } else {
                        double value =
                            loadScalar(source + source_index * source_size,
                                       options.source_type);
                        if (normalising) {
                            value = (value * scale - mean[channel]) /
                                    stddev[channel];
                        }
                        storeScalar(target + target_index * target_size,
                                    options.target_type, value);
                    }
                }
            }
        }
        return tensor;
    }

    size_t n_dim = static_cast<size_t>(shape[0]);
    size_t c_dim = static_cast<size_t>(shape[1]);
    size_t h_dim = static_cast<size_t>(shape[2]);
    size_t w_dim = static_cast<size_t>(shape[3]);
    size_t plane = h_dim * w_dim;
    size_t image = c_dim * plane;
    for (size_t n = 0; n < n_dim; ++n) {
        for (size_t channel = 0; channel < c_dim; ++channel) {
            for (size_t h = 0; h < h_dim; ++h) {
                for (size_t w = 0; w < w_dim; ++w) {
                    size_t source_index =
                        ((n * h_dim + h) * w_dim + w) * c_dim + channel;
                    size_t target_index =
                        n * image + channel * plane + h * w_dim + w;
                    if (copy_raw) {
                        std::memcpy(target + target_index * target_size,
                                    source + source_index * source_size,
                                    target_size);
                    } else {
                        double value =
                            loadScalar(source + source_index * source_size,
                                       options.source_type);
                        if (normalising) {
                            value = (value * scale - mean[channel]) /
                                    stddev[channel];
                        }
                        storeScalar(target + target_index * target_size,
                                    options.target_type, value);
                    }
                }
            }
        }
    }
    return tensor;
}

std::string tensorToBuffer(const Tensor &tensor,
                           ONNXTensorElementDataType target_type, bool hwc) {
    if (tensor.isStringTensor()) {
        throw ConversionError(
            "toBuffer() does not serialise string tensors; their payload is "
            "variable length and has no byte layout");
    }
    if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        throw ConversionError(
            "toBuffer() cannot cast to string: a string payload is variable "
            "length and has no byte layout");
    }
    if (!isSupportedElementsType(tensor.type) ||
        !isSupportedElementsType(target_type)) {
        throw ConversionError(
            "toBuffer() supports only numeric and bool element types");
    }

    size_t count = tensor.elementCount();
    size_t source_size = elementByteSize(tensor.type);
    size_t target_size = elementByteSize(target_type);
    size_t expected = 0;
    if (count != 0) {
        if (source_size > std::numeric_limits<size_t>::max() / count ||
            target_size > std::numeric_limits<size_t>::max() / count) {
            throw ConversionError("tensor byte size overflows size_t");
        }
        expected = count * source_size;
    }
    if (tensor.data.size() != expected) {
        throw ConversionError(
            "numeric payload size " + std::to_string(tensor.data.size()) +
            " does not match shape/dtype (expected " + std::to_string(expected) +
            " bytes)");
    }

    if (hwc && tensor.shape.size() != 3 && tensor.shape.size() != 4) {
        throw ConversionError(
            "layout 'hwc' requires a rank-3 or rank-4 tensor");
    }

    std::string out(count * target_size, '\0');
    if (count == 0) {
        return out;
    }

    const uint8_t *source = tensor.data.data();

    /* No cast: elements are byte-identical, so re-ordering copies bytes
     * instead of round-tripping them through double. */
    const bool copy_raw = tensor.type == target_type;

    // Fast path: no cast and no re-ordering.
    if (copy_raw && !hwc) {
        std::memcpy(out.data(), source, expected);
        return out;
    }

    uint8_t *target = reinterpret_cast<uint8_t *>(out.data());

    if (!hwc) {
        for (size_t i = 0; i < count; ++i) {
            storeScalar(target + i * target_size, target_type,
                        loadScalar(source + i * source_size, tensor.type));
        }
        return out;
    }

    size_t c_dim;
    size_t h_dim;
    size_t w_dim;
    size_t n_dim = 1;
    if (tensor.shape.size() == 3) {
        c_dim = static_cast<size_t>(tensor.shape[0]);
        h_dim = static_cast<size_t>(tensor.shape[1]);
        w_dim = static_cast<size_t>(tensor.shape[2]);
    } else {
        n_dim = static_cast<size_t>(tensor.shape[0]);
        c_dim = static_cast<size_t>(tensor.shape[1]);
        h_dim = static_cast<size_t>(tensor.shape[2]);
        w_dim = static_cast<size_t>(tensor.shape[3]);
    }
    size_t plane = h_dim * w_dim;
    size_t image = c_dim * plane;
    for (size_t n = 0; n < n_dim; ++n) {
        for (size_t channel = 0; channel < c_dim; ++channel) {
            for (size_t h = 0; h < h_dim; ++h) {
                for (size_t w = 0; w < w_dim; ++w) {
                    size_t source_index =
                        n * image + channel * plane + h * w_dim + w;
                    size_t target_index =
                        ((n * h_dim + h) * w_dim + w) * c_dim + channel;
                    if (copy_raw) {
                        std::memcpy(target + target_index * target_size,
                                    source + source_index * source_size,
                                    target_size);
                    } else {
                        storeScalar(
                            target + target_index * target_size, target_type,
                            loadScalar(source + source_index * source_size,
                                       tensor.type));
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace php_ort
