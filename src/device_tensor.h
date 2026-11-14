/*
 * PHP ONNX Runtime extension - Phase 7C/7D device tensor.
 *
 * A DeviceTensor is a separately allocated Ort::Value whose buffer lives in
 * memory owned by an OrtEpDevice allocator.  It is deliberately not a Tensor:
 * Tensor always owns a CPU-side payload, while DeviceTensor never exposes a
 * host pointer and never assumes the buffer is readable from PHP.
 *
 * The implementation uses only generic ORT APIs (EpDevice::GetMemoryInfo,
 * Env::GetSharedAllocator, Value::CreateTensor and Env::CopyTensor).  No
 * vendor SDK (CUDA Runtime, etc.) is linked by this file.
 */

#ifndef PHP_ORT_DEVICE_TENSOR_H
#define PHP_ORT_DEVICE_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "ep_device.h"
#include "onnx_tensor.h"

namespace php_ort {

/**
 * Owns one device-memory tensor.
 *
 * Lifetime rules:
 *   - `env` keeps the process Ort::Env alive while the allocator/value exist.
 *   - `plugin_registrations` keeps a Plugin EP library loaded for as long as
 *     this object exists.  It is acquired before allocation and released only
 *     after `value` has been destroyed (member destruction order).
 *   - The shared allocator is borrowed from the Env; releasing the Env's
 *     shared allocator independently is not required and would be unsafe while
 *     this value is alive.
 *   - The value owns its device buffer and frees it through the allocator.
 *     There is intentionally no CPU data pointer accessor.
 */
class DeviceTensor {
public:
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;
    DeviceTensor(DeviceTensor &&) = delete;
    DeviceTensor &operator=(DeviceTensor &&) = delete;

    /**
     * Allocate a device tensor.
     *
     * `memory_type` is "device" (OrtDeviceMemoryType_DEFAULT) or
     * "host_accessible" (OrtDeviceMemoryType_HOST_ACCESSIBLE).  String tensors
     * are rejected: ORT's allocator-based CreateTensor does not support them.
     */
    static std::unique_ptr<DeviceTensor> allocate(
        const EpDeviceInfo &device,
        const std::vector<int64_t> &shape,
        ONNXTensorElementDataType type,
        const std::string &memory_type);

    ~DeviceTensor();

    ONNXTensorElementDataType type() const noexcept { return type_; }
    const std::vector<int64_t> &shape() const noexcept { return shape_; }
    const EpDeviceInfo &device() const noexcept { return device_; }
    const std::string &memoryType() const noexcept { return memory_type_; }

    /** Borrowed OrtValue for IoBinding.  Valid while this object is alive. */
    Ort::Value &value() noexcept { return value_; }

    /**
     * Synchronous CPU -> device copy.  The source Tensor is read only for the
     * duration of this call.  Type and shape must match exactly.
     */
    void copyFrom(const Tensor &source);

    /** Synchronous device -> CPU copy.  Returns an independent CPU Tensor. */
    Tensor copyTo() const;

    /**
     * Synchronous device -> device copy through Env::CopyTensor.  Both tensors
     * must have identical dtype and shape.  Cross-EP copies are delegated to
     * ORT; an EP that cannot copy between the two devices reports an error.
     */
    void copyFromDevice(const DeviceTensor &source);

private:
    DeviceTensor() = default;

    /** Throws ConversionError when shape has a negative or overflowing size. */
    static size_t checkedElementCount(const std::vector<int64_t> &shape);
    static std::string normalizeMemoryType(const std::string &memory_type);
    static OrtDeviceMemoryType ortMemoryType(const std::string &normalized);

    void assertCompatible(const DeviceTensor &other) const;
    void copyValue(const Ort::Value &source, Ort::Value &destination) const;
    bool isCpuDevice() const;
    void copyCpuMemory(const void *source, void *destination,
                       size_t bytes) const;

    // Declared first: destroyed after the value and allocator.
    std::vector<std::string> plugin_registrations_;
    std::shared_ptr<Ort::Env> env_;

    EpDeviceInfo device_;
    std::vector<int64_t> shape_;
    ONNXTensorElementDataType type_ =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::string memory_type_ = "device";

    // Borrowed from the Env.  Must outlive value_.
    Ort::UnownedAllocator allocator_{nullptr};
    Ort::Value value_{nullptr};
};

}  // namespace php_ort

#endif  // PHP_ORT_DEVICE_TENSOR_H
