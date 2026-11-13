/*
 * PHP ONNX Runtime extension - internal tensor representation.
 *
 * This header deliberately has no dependency on the Zend Engine or on
 * ONNX Runtime itself beyond the official C API element type enum.  Keeping
 * the representation free of Zend types lets later phases reuse it for
 * session I/O without dragging PHP request state into the conversion core.
 */

#ifndef PHP_ORT_TENSOR_H
#define PHP_ORT_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_c_api.h>

namespace php_ort {

/**
 * Error raised by the conversion layer.  It is translated into a PHP
 * OnnxRuntime\ConversionException by the Zend binding; the core never throws
 * PHP exceptions itself.
 */
class ConversionError : public std::runtime_error {
public:
    explicit ConversionError(const std::string &message)
        : std::runtime_error(message) {}
};

/**
 * A flat, row-major tensor description.
 *
 * Ownership invariant: a Tensor always owns its payload.
 *   - Numeric and bool payloads live in `data` as native-endian raw bytes.
 *   - String payloads live in `strings`; strings are never packed into `data`
 *     because std::string is not a fixed-size scalar and must keep its own
 *     lifetime management.
 *   - `data` and `strings` are mutually exclusive and selected by `type`.
 *
 * External memory (e.g. an OrtValue output buffer) must not be modeled by
 * adding a borrowed flag here; if a non-owning view is needed later it should
 * be a separate view type with an explicit owner/deleter.  `Tensor -> OrtValue`
 * may borrow `dataPointer()` for the duration of `Run()`, which does not change
 * the Tensor's ownership.
 */
struct Tensor {
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
    std::vector<std::string> strings;

    size_t rank() const { return shape.size(); }

    /** Number of elements implied by the shape. Rank-0 tensors hold one. */
    size_t elementCount() const;

    /** Byte size of the numeric payload. Zero for string tensors. */
    size_t dataSize() const { return data.size(); }

    /** Pointer to the numeric payload, or nullptr when empty/string. */
    const void *dataPointer() const {
        return data.empty() ? nullptr : static_cast<const void *>(data.data());
    }

    bool isStringTensor() const {
        return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }

    /**
     * Single validation entry point for every Tensor, regardless of whether
     * it came from PHP zval conversion or (in a later phase) from OrtValue.
     *
     * Checks:
     *   - supported / defined element type
     *   - non-negative dimensions and element-count overflow
     *   - dtype vs data buffer byte size
     *   - string tensor: string count == element count, no numeric payload
     *   - numeric tensor: no string payload, byte-size product overflow
     *
     * Throws ConversionError on failure.
     */
    void validate() const;
};

/** Byte width of one element of `type`; throws for STRING/UNDEFINED. */
size_t elementByteSize(ONNXTensorElementDataType type);

/** Lower-case ONNX name, e.g. "float32"; returns "unknown" when unmapped. */
const char *elementTypeName(ONNXTensorElementDataType type);

/**
 * Resolves a PHP-facing dtype name.  Returns false for unsupported names;
 * the caller decides how to report it (ValueError at the PHP layer).
 */
bool elementTypeFromName(const std::string &name,
                         ONNXTensorElementDataType *out);

/** Comma separated list of names accepted by elementTypeFromName(). */
const char *supportedDtypeNames();

/** True for the element types Phase 1 implements. */
bool isSupportedElementsType(ONNXTensorElementDataType type);

}  // namespace php_ort

#endif  // PHP_ORT_TENSOR_H
