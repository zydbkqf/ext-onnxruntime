#include "session_options.h"

#include "onnx_tensor.h"

namespace php_ort {
namespace {

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
            options.AppendExecutionProvider(provider.name, provider.options);
            continue;
        }

        // Resolve a fresh Env-owned device.  SessionConfig stores only the
        // identity snapshot, so a registration cannot invalidate a raw
        // OrtEpDevice* held by this PHP object.
        auto device = resolveEpDevice(env, provider.device);
        std::vector<Ort::ConstEpDevice> devices;
        devices.push_back(device);
        options.AppendExecutionProvider_V2(env, devices, provider.options);
    }
}

}  // namespace php_ort
