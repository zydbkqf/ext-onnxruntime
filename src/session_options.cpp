#include "session_options.h"

#include "onnx_tensor.h"
#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <memory>

namespace php_ort {
namespace {

// ORT's generic SessionOptionsAppendExecutionProvider(name, options) entry
// point only accepts the providers listed in onnxruntime_c_api.h; CUDA and
// TensorRT are deliberately excluded because they have dedicated, typed
// appenders.  A CUDA-enabled Linux build exposes CUDA as a built-in provider
// (it shows up in getAvailableProviders() and has no EpDevice to append), so
// the name has to be routed through SessionOptionsAppendExecutionProvider_CUDA_V2
// instead of falling through to the generic call, which fails with
// "Unknown provider name 'CUDA'".
std::string lowerName(const std::string &name) {
    std::string lower(name);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

bool isCudaProviderName(const std::string &name) {
    const std::string lower = lowerName(name);
    return lower == "cuda" || lower == "cudaexecutionprovider";
}

bool isTensorRtProviderName(const std::string &name) {
    const std::string lower = lowerName(name);
    // "NvTensorRtRtx" is a different provider and *is* handled by the generic
    // name-based appender, so it must not be routed here.
    return lower == "tensorrt" || lower == "tensorrtexecutionprovider";
}

bool isCoreMlProviderName(const std::string &name) {
    const std::string lower = lowerName(name);
    return lower == "coreml" || lower == "coremlexecutionprovider";
}

#if PHP_ORT_HAS_EP_DEVICE
std::string canonicalProviderName(const std::string &name) {
    std::string canonical;
    canonical.reserve(name.size());
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            canonical.push_back(static_cast<char>(std::tolower(c)));
        }
    }

    static const std::string suffix = "executionprovider";
    if (canonical.size() >= suffix.size() &&
        canonical.compare(canonical.size() - suffix.size(), suffix.size(), suffix) == 0) {
        canonical.erase(canonical.size() - suffix.size());
    }
    if (canonical.size() > 2 &&
        canonical.compare(canonical.size() - 2, 2, "ep") == 0) {
        canonical.erase(canonical.size() - 2);
    }
    if (canonical == "directml") {
        canonical = "dml";
    } else if (canonical == "onednn") {
        canonical = "dnnl";
    }
    return canonical;
}
#endif

void appendCudaProvider(Ort::SessionOptions &options,
                        const ExecutionProviderConfig &provider) {
    OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
    std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2 *)> guard(
        cuda_options, [](OrtCUDAProviderOptionsV2 *ptr) {
            if (ptr != nullptr) {
                Ort::GetApi().ReleaseCUDAProviderOptions(ptr);
            }
        });
    for (const auto &option : provider.options) {
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(
            cuda_options, option.first.c_str(),
            const_cast<char *>(option.second.c_str())));
    }
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
}

void appendTensorRtProvider(Ort::SessionOptions &options,
                            const ExecutionProviderConfig &provider) {
    OrtTensorRTProviderOptionsV2 *trt_options = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_options));
    std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2 *)> guard(
        trt_options, [](OrtTensorRTProviderOptionsV2 *ptr) {
            if (ptr != nullptr) {
                Ort::GetApi().ReleaseTensorRTProviderOptions(ptr);
            }
        });
    for (const auto &option : provider.options) {
        Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptionsWithValue(
            trt_options, option.first.c_str(),
            const_cast<char *>(option.second.c_str())));
    }
    options.AppendExecutionProvider_TensorRT_V2(*trt_options);
}

GraphOptimizationLevel graphLevelFromName(const std::string &name) {
    if (name == "disabled") {
        return ORT_DISABLE_ALL;
    }
    if (name == "basic") {
        return ORT_ENABLE_BASIC;
    }
    if (name == "extended") {
        return ORT_ENABLE_EXTENDED;
    }
    if (name == "all") {
        return ORT_ENABLE_ALL;
    }
    throw ConversionError("unknown graph optimization level: " + name);
}

ExecutionMode executionModeFromName(const std::string &name) {
    if (name == "sequential") {
        return ORT_SEQUENTIAL;
    }
    if (name == "parallel") {
        return ORT_PARALLEL;
    }
    throw ConversionError("unknown execution mode: " + name);
}

}  // namespace

void resolveExecutionProviderDevices(SessionConfig &config, Ort::Env &env) {
#if PHP_ORT_HAS_EP_DEVICE
    (void)env;
    bool has_device_candidate = false;
    for (const ExecutionProviderConfig &provider : config.execution_providers) {
        if (provider.kind != ExecutionProviderKind::Builtin) {
            continue;
        }
        if (isCudaProviderName(provider.name) ||
            isTensorRtProviderName(provider.name) ||
            isCoreMlProviderName(provider.name)) {
            continue;
        }
        has_device_candidate = true;
        break;
    }
    if (!has_device_candidate) {
        return;
    }

    // runtimeEpDevices() enriches the Env snapshot with Plugin EP registration
    // names and generations.  readEpDevices() alone leaves those fields empty,
    // so a name-resolved plugin device would acquire no lease and the plugin
    // could be unregistered while its Session is still alive.
    const std::vector<EpDeviceInfo> devices = runtimeEpDevices();
    if (devices.empty()) {
        return;
    }

    for (ExecutionProviderConfig &provider : config.execution_providers) {
        if (provider.kind != ExecutionProviderKind::Builtin) {
            continue;
        }
        if (isCudaProviderName(provider.name) ||
            isTensorRtProviderName(provider.name) ||
            isCoreMlProviderName(provider.name)) {
            continue;
        }

        const std::string wanted = canonicalProviderName(provider.name);
        if (wanted.empty()) {
            continue;
        }
        for (const EpDeviceInfo &device : devices) {
            if (canonicalProviderName(device.ep_name) != wanted) {
                continue;
            }
            // The first matching device in ORT enumeration order wins.  A
            // provider with several devices should use appendEpDevice() so
            // the caller can select one explicitly.
            provider.kind = ExecutionProviderKind::EpDevice;
            provider.device = device;
            break;
        }
    }
#else
    (void)config;
    (void)env;
#endif
}

void SessionConfig::applyTo(Ort::SessionOptions &options,
                            Ort::Env &env) const {
    // 0 means "leave ORT's default thread-pool sizing in place".
    if (intra_op_num_threads > 0) {
        options.SetIntraOpNumThreads(intra_op_num_threads);
    }
    if (inter_op_num_threads > 0) {
        options.SetInterOpNumThreads(inter_op_num_threads);
    }

    options.SetGraphOptimizationLevel(
        graphLevelFromName(graph_optimization_level));
    options.SetExecutionMode(executionModeFromName(execution_mode));
    options.SetLogSeverityLevel(log_severity_level);

    if (enable_cpu_mem_arena) {
        options.EnableCpuMemArena();
    } else {
        options.DisableCpuMemArena();
    }
    if (enable_mem_pattern) {
        options.EnableMemPattern();
    } else {
        options.DisableMemPattern();
    }

    for (const ExecutionProviderConfig &provider : execution_providers) {
        if (provider.kind == ExecutionProviderKind::Builtin) {
            if (isCudaProviderName(provider.name)) {
                appendCudaProvider(options, provider);
                continue;
            }
            if (isTensorRtProviderName(provider.name)) {
                appendTensorRtProvider(options, provider);
                continue;
            }
            if (isCoreMlProviderName(provider.name)) {
#if ORT_API_VERSION < 23
                throw ConversionError(
                    "the CoreML execution provider requires an ONNX Runtime "
                    "1.23 or newer SDK (this build targets API " +
                    std::to_string(ORT_API_VERSION) + ")");
#else
                options.AppendExecutionProvider("CoreML", provider.options);
                continue;
#endif
            }
            options.AppendExecutionProvider(provider.name, provider.options);
            continue;
        }

#if PHP_ORT_HAS_EP_DEVICE
        // Resolve a fresh Env-owned device.  SessionConfig stores only the
        // identity snapshot, so a registration cannot invalidate a raw
        // OrtEpDevice* held by this PHP object.
        auto device = resolveEpDevice(env, provider.device);
        std::vector<Ort::ConstEpDevice> devices;
        devices.push_back(device);
        options.AppendExecutionProvider_V2(env, devices, provider.options);
#else
        (void)provider;
        (void)env;
        (void)options;
        throw ConversionError(
            "appendEpDevice requires ONNX Runtime 1.23 or newer (API 23)");
#endif
    }
}

}  // namespace php_ort
