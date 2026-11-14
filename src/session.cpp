#include "session.h"
#include "session_options.h"

#include <memory>
#include <utility>

#include "io_binding.h"
#include "ort_environment.h"
#include "ort_value_converter.h"
#include "runtime.h"
#include "win_utils.h"

namespace php_ort {
namespace {

const char *onnxTypeName(ONNXType type) {
    switch (type) {
        case ONNX_TYPE_TENSOR:
            return "tensor";
        case ONNX_TYPE_SEQUENCE:
            return "sequence";
        case ONNX_TYPE_MAP:
            return "map";
        case ONNX_TYPE_OPAQUE:
            return "opaque";
        case ONNX_TYPE_SPARSETENSOR:
            return "sparse_tensor";
        case ONNX_TYPE_OPTIONAL:
            return "optional";
        case ONNX_TYPE_UNKNOWN:
        default:
            return "unknown";
    }
}

TensorInfo readTensorInfo(Ort::Session &session, bool input, size_t index,
                          Ort::AllocatorWithDefaultOptions &allocator) {
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    if (name == nullptr) {
        throw ConversionError("ORT returned a null tensor name");
    }

    TensorInfo info;
    info.name.assign(name.get());

    Ort::TypeInfo type = input ? session.GetInputTypeInfo(index)
                               : session.GetOutputTypeInfo(index);
    ONNXType onnx_type = type.GetONNXType();
    if (onnx_type != ONNX_TYPE_TENSOR) {
        throw ConversionError(
            std::string(input ? "input \"" : "output \"") + info.name +
            "\" is ONNX type " + onnxTypeName(onnx_type) +
            ", not a tensor; Phase 3 supports tensor inputs/outputs only");
    }

    auto shape_info = type.GetTensorTypeAndShapeInfo();
    info.type = shape_info.GetElementType();
    if (!isSupportedElementsType(info.type)) {
        throw ConversionError(
            std::string(input ? "input \"" : "output \"") + info.name +
            "\" has dtype " + elementTypeName(info.type) +
            ", which is recognized but not supported by this extension's "
            "Tensor conversion");
    }
    info.shape = shape_info.GetShape();
    return info;
}

std::vector<TensorInfo> readAllInfos(Ort::Session &session, bool input,
                                    Ort::AllocatorWithDefaultOptions &allocator) {
    size_t count = input ? session.GetInputCount() : session.GetOutputCount();
    std::vector<TensorInfo> infos;
    infos.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        infos.push_back(readTensorInfo(session, input, i, allocator));
    }
    return infos;
}

std::string shapeToString(const std::vector<int64_t> &shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        if (shape[i] < 0) {
            text += "?";
        } else {
            text += std::to_string(shape[i]);
        }
    }
    return text + "]";
}

size_t findInfoIndex(const std::vector<TensorInfo> &infos,
                     const std::string &name, const char *direction) {
    for (size_t i = 0; i < infos.size(); ++i) {
        if (infos[i].name == name) {
            return i;
        }
    }
    throw ConversionError("unknown " + std::string(direction) + " \"" +
                          name + "\"");
}

/**
 * Enforces the extension's own input/output contract before ORT sees the
 * tensor.  Negative model dimensions are symbolic/dynamic and are allowed to
 * vary; non-negative dimensions are static and must match exactly.
 */
void validateTensorAgainstInfo(const Tensor &tensor, const TensorInfo &info,
                               bool input) {
    tensor.validate();

    const char *direction = input ? "input" : "output";
    if (tensor.type != info.type) {
        throw ConversionError(
            std::string(direction) + " \"" + info.name +
            "\" dtype mismatch: model expects " + elementTypeName(info.type) +
            ", got " + elementTypeName(tensor.type));
    }

    if (tensor.shape.size() != info.shape.size()) {
        throw ConversionError(
            std::string(direction) + " \"" + info.name +
            "\" rank mismatch: model expects rank " +
            std::to_string(info.shape.size()) + " " + shapeToString(info.shape) +
            ", got rank " + std::to_string(tensor.shape.size()) + " " +
            shapeToString(tensor.shape));
    }

    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (info.shape[i] >= 0 && tensor.shape[i] != info.shape[i]) {
            throw ConversionError(
                std::string(direction) + " \"" + info.name +
                "\" shape mismatch: model expects " + shapeToString(info.shape) +
                ", got " + shapeToString(tensor.shape));
        }
    }
}

}  // namespace

SessionState::PluginLease::~PluginLease() {
    releasePluginSessions(registrations);
}

void SessionState::PluginLease::acquire(
    const std::vector<EpDeviceInfo> &devices) {
    registrations = acquirePluginSessions(devices);
}

SessionState::SessionState(const std::string &model_path,
                           const SessionConfig *config)
    : env(environment()), session(nullptr) {
    ensureRuntimeAcceptingWork();

    if (config != nullptr) {
        std::vector<EpDeviceInfo> devices;
        for (const ExecutionProviderConfig &provider :
             config->execution_providers) {
            if (provider.kind == ExecutionProviderKind::EpDevice) {
                devices.push_back(provider.device);
            }
        }
        // Reserve the plugin lease before creating the Ort::Session.  A
        // concurrent unregister must see a non-zero count and refuse to unload
        // the library.  If any later step throws, the lease destructor releases
        // the reservation.
        plugin_lease.acquire(devices);
    }

    Ort::SessionOptions options;
    if (config != nullptr) {
        config->applyTo(options, *env);
    }
#ifdef _WIN32
    session = Ort::Session(*env, toWidePath(model_path).c_str(), options);
#else
    session = Ort::Session(*env, model_path.c_str(), options);
#endif
    Ort::AllocatorWithDefaultOptions allocator;
    inputs_ = readAllInfos(session, true, allocator);
    outputs_ = readAllInfos(session, false, allocator);
}

SessionState::~SessionState() = default;

const std::vector<TensorInfo> &SessionState::inputs() const {
    return inputs_;
}

const std::vector<TensorInfo> &SessionState::outputs() const {
    return outputs_;
}

size_t SessionState::inputIndex(const std::string &name) const {
    return findInfoIndex(inputs_, name, "input");
}

size_t SessionState::outputIndex(const std::string &name) const {
    return findInfoIndex(outputs_, name, "output");
}

void SessionState::validateInput(size_t index, const Tensor &tensor) const {
    if (index >= inputs_.size()) {
        throw ConversionError("input index out of range");
    }
    validateTensorAgainstInfo(tensor, inputs_[index], true);
}

void SessionState::validateOutput(size_t index, const Tensor &tensor) const {
    if (index >= outputs_.size()) {
        throw ConversionError("output index out of range");
    }
    validateTensorAgainstInfo(tensor, outputs_[index], false);
}

std::vector<Tensor> SessionState::copyInputsForAsync(
    const std::vector<const Tensor *> &input_tensors) const {
    if (input_tensors.size() != inputs_.size()) {
        throw ConversionError("input tensor count does not match model inputs");
    }
    std::vector<Tensor> result;
    result.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const Tensor *tensor = input_tensors[i];
        if (tensor == nullptr) {
            throw ConversionError("null input tensor");
        }
        validateTensorAgainstInfo(*tensor, inputs_[i], true);
        result.push_back(*tensor);  // deep copy of numeric / string payload
    }
    return result;
}

std::vector<Tensor> SessionState::run(
    const std::vector<const Tensor *> &input_tensors) {
    if (input_tensors.size() != inputs_.size()) {
        throw ConversionError("input tensor count does not match model inputs");
    }

    ensureRuntimeAcceptingWork();

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> input_values;
    input_values.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const Tensor *tensor = input_tensors[i];
        if (tensor == nullptr) {
            throw ConversionError("null input tensor");
        }
        // Validate against the model metadata first, so dtype/rank/static
        // shape errors are ConversionException instead of an opaque ORT Run
        // failure.  Dynamic (negative) model dimensions are skipped.
        validateTensorAgainstInfo(*tensor, inputs_[i], true);
        input_values.push_back(
            tensorToOrtValue(*tensor, memory_info, allocator));
    }

    std::vector<const char *> input_names;
    input_names.reserve(inputs_.size());
    for (const TensorInfo &info : inputs_) {
        input_names.push_back(info.name.c_str());
    }

    std::vector<const char *> output_names;
    output_names.reserve(outputs_.size());
    for (const TensorInfo &info : outputs_) {
        output_names.push_back(info.name.c_str());
    }

    // All input OrtValues stay alive until Run returns.  Numeric values borrow
    // the PHP Tensors' buffers; those Tensors are kept alive by the caller's
    // input array for the same duration.
    Ort::RunOptions run_options{};
    std::vector<Ort::Value> output_values = session.Run(
        run_options, input_names.data(), input_values.data(),
        input_values.size(), output_names.data(), output_names.size());

    if (output_values.size() != outputs_.size()) {
        throw ConversionError(
            "ORT returned an unexpected number of output tensors");
    }

    // Convert before output_values goes out of scope: no borrowed output
    // buffer ever reaches the PHP Tensor layer.
    std::vector<Tensor> results;
    results.reserve(output_values.size());
    for (size_t i = 0; i < output_values.size(); ++i) {
        Tensor result = ortValueToTensor(output_values[i]);
        validateTensorAgainstInfo(result, outputs_[i], false);
        results.push_back(std::move(result));
    }
    return results;
}

Session::Session(const std::string &model_path,
                 const SessionConfig *config)
    : state_(std::make_shared<SessionState>(model_path, config)) {}

const std::vector<TensorInfo> &Session::inputs() const {
    return state_->inputs();
}

const std::vector<TensorInfo> &Session::outputs() const {
    return state_->outputs();
}

std::vector<Tensor> Session::run(
    const std::vector<const Tensor *> &input_tensors) {
    return state_->run(input_tensors);
}

std::vector<Tensor> Session::copyInputsForAsync(
    const std::vector<const Tensor *> &input_tensors) const {
    return state_->copyInputsForAsync(input_tensors);
}

std::shared_ptr<IoBindingState> Session::createIoBinding() {
    return std::make_shared<IoBindingState>(state_, state_->nativeSession());
}

const std::shared_ptr<SessionState> &Session::state() const {
    return state_;
}

}  // namespace php_ort
