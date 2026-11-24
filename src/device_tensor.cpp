#include "device_tensor.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>

#include "ort_environment.h"
#include "ort_value_converter.h"
#include "runtime.h"

namespace php_ort {

#if PHP_ORT_HAS_EP_DEVICE

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}  // namespace

size_t DeviceTensor::checkedElementCount(const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw ConversionError("negative device tensor dimension");
        }
        if (dim == 0) {
            return 0;
        }
        if (count > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(dim)) {
            throw ConversionError("device tensor element count overflow");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::string DeviceTensor::normalizeMemoryType(
    const std::string &memory_type) {
    std::string normalized = lowercase(memory_type);
    if (normalized == "device" || normalized == "default") {
        return "device";
    }
    if (normalized == "host" || normalized == "host_accessible" ||
        normalized == "host-accessible") {
        return "host_accessible";
    }
    throw ConversionError(
        "memoryType must be \"device\" or \"host_accessible\"");
}

int DeviceTensor::ortMemoryType(
    const std::string &normalized) {
    return normalized == "host_accessible"
               ? static_cast<int>(OrtDeviceMemoryType_HOST_ACCESSIBLE)
               : static_cast<int>(OrtDeviceMemoryType_DEFAULT);
}

std::unique_ptr<DeviceTensor> DeviceTensor::allocate(
    const EpDeviceInfo &device,
    const std::vector<int64_t> &shape,
    ONNXTensorElementDataType type,
    const std::string &memory_type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        throw ConversionError(
            "device tensors support numeric element types only");
    }
    if (!isSupportedElementsType(type)) {
        throw ConversionError(
            "unsupported device tensor element type");
    }
    checkedElementCount(shape);

    std::unique_ptr<DeviceTensor> tensor(new DeviceTensor());
    tensor->device_ = device;
    tensor->shape_ = shape;
    tensor->type_ = type;
    tensor->memory_type_ = normalizeMemoryType(memory_type);
    tensor->env_ = environment();

    // Acquire the Plugin EP lease before asking ORT for an allocator.  The
    // lease is an RAII member declared after env_ and before allocator_/value_,
    // so an exception below destroys this partially built object in the safe
    // order: value, allocator, lease, Env.
    tensor->plugin_lease_.acquire({device});

    Ort::ConstEpDevice ep_device =
        resolveEpDevice(*tensor->env_, device);
    Ort::ConstMemoryInfo memory_info = ep_device.GetMemoryInfo(
        static_cast<OrtDeviceMemoryType>(ortMemoryType(tensor->memory_type_)));

    // EpDevice_MemoryInfo returns nullptr for DEFAULT on an EP that uses
    // CPU memory (including the built-in CPU device).  In that case use
    // the CPU memory info and the Env's matching shared allocator.
    Ort::MemoryInfo cpu_memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::UnownedAllocator allocator =
        memory_info == nullptr
            ? tensor->env_->GetSharedAllocator(cpu_memory)
            : tensor->env_->GetSharedAllocator(memory_info);
    if (allocator == nullptr) {
        throw ConversionError(
            "no shared allocator for the requested device memory type");
    }

    tensor->allocator_ = allocator;
    tensor->value_ = Ort::Value::CreateTensor(
        allocator, tensor->shape_.data(), tensor->shape_.size(),
        tensor->type_);

    return tensor;
}

DeviceTensor::~DeviceTensor() = default;

void DeviceTensor::assertCompatible(const DeviceTensor &other) const {
    if (type_ != other.type_) {
        throw ConversionError(
            "device tensors have different element types");
    }
    if (shape_ != other.shape_) {
        throw ConversionError(
            "device tensors have different shapes");
    }
}

bool DeviceTensor::isCpuDevice() const {
    // The built-in CPU device reports CPU hardware and no Plugin EP
    // registration.  Its shared allocator can return ordinary host memory,
    // and ORT has no CPU->CPU data-transfer implementation for CopyTensor.
    return !device_.isPluginDevice() &&
           device_.device_type == OrtHardwareDeviceType_CPU;
}

void DeviceTensor::copyCpuMemory(const void *source, void *destination,
                                 size_t bytes) const {
    if (bytes != 0) {
        std::memcpy(destination, source, bytes);
    }
}

void DeviceTensor::copyValue(const Ort::Value &source,
                             Ort::Value &destination) const {
    if (!env_ || source == nullptr || destination == nullptr) {
        throw ConversionError("device tensor copy from an uninitialized value");
    }
    // A null stream requests ORT's synchronous copy semantics.  Asynchronous
    // copies are intentionally not exposed until an EP provides a reliable
    // completion stream.  Use the C CopyTensors entry point (present since
    // ORT 1.24); the C++ single-tensor overload is newer and the C++ vector
    // overload cannot be built from non-copyable Ort::Value handles.
    const OrtValue *src_tensors[1] = {source};
    OrtValue *dst_tensors[1] = {destination};
    Ort::Status status(Ort::GetApi().CopyTensors(
        static_cast<OrtEnv *>(*env_), src_tensors, dst_tensors, nullptr, 1));
    if (!status.IsOK()) {
        throw ConversionError("device tensor copy failed: " +
                              status.GetErrorMessage());
    }
}

void DeviceTensor::copyFrom(const Tensor &source) {
    if (source.type != type_ || source.shape != shape_) {
        throw ConversionError(
            "source tensor type/shape does not match the device tensor");
    }
    source.validate();
    if (checkedElementCount(source.shape) == 0) {
        return;
    }

    Ort::MemoryInfo cpu_memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    if (isCpuDevice()) {
        void *destination = value_.GetTensorMutableData<void>();
        copyCpuMemory(source.data.data(), destination, source.data.size());
        return;
    }
    Ort::Value source_value =
        tensorToOrtValue(source, cpu_memory, allocator);
    copyValue(source_value, value_);
}

Tensor DeviceTensor::copyTo() const {
    Tensor result;
    result.type = type_;
    result.shape = shape_;

    size_t element_count = checkedElementCount(shape_);
    if (element_count == 0) {
        return result;
    }
    size_t element_size = elementByteSize(type_);
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        throw ConversionError("device tensor byte size overflow");
    }
    result.data.resize(element_count * element_size);

    Ort::MemoryInfo cpu_memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (isCpuDevice()) {
        const void *source = value_.GetTensorData<void>();
        copyCpuMemory(source, result.data.data(), result.data.size());
        result.validate();
        return result;
    }
    Ort::Value destination = Ort::Value::CreateTensor(
        cpu_memory, result.data.data(), result.data.size(),
        shape_.data(), shape_.size(), type_);
    copyValue(value_, destination);
    result.validate();
    return result;
}

void DeviceTensor::copyFromDevice(const DeviceTensor &source) {
    assertCompatible(source);
    size_t element_count = checkedElementCount(shape_);
    if (element_count == 0) {
        return;
    }
    size_t element_size = elementByteSize(type_);
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        throw ConversionError("device tensor byte size overflow");
    }
    size_t bytes = element_count * element_size;
    if (isCpuDevice() && source.isCpuDevice()) {
        void *destination = value_.GetTensorMutableData<void>();
        const void *source_data = source.value_.GetTensorData<void>();
        copyCpuMemory(source_data, destination, bytes);
        return;
    }
    copyValue(source.value_, value_);
}

#else

size_t DeviceTensor::checkedElementCount(const std::vector<int64_t> &) {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

std::string DeviceTensor::normalizeMemoryType(const std::string &) {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

int DeviceTensor::ortMemoryType(const std::string &) {
    return 0;
}

std::unique_ptr<DeviceTensor> DeviceTensor::allocate(
    const EpDeviceInfo &, const std::vector<int64_t> &,
    ONNXTensorElementDataType, const std::string &) {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

void DeviceTensor::assertCompatible(const DeviceTensor &) const {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

bool DeviceTensor::isCpuDevice() const {
    return false;
}

void DeviceTensor::copyCpuMemory(const void *, void *, size_t) const {
}

void DeviceTensor::copyValue(const Ort::Value &, Ort::Value &) const {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

void DeviceTensor::copyFrom(const Tensor &) {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

Tensor DeviceTensor::copyTo() const {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

void DeviceTensor::copyFromDevice(const DeviceTensor &) {
    throw ConversionError(
        "device tensors require ONNX Runtime 1.23 or newer (API 23)");
}

#endif

}  // namespace php_ort
