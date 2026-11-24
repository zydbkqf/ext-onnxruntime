#ifndef PHP_ORT_SESSION_OPTIONS_H
#define PHP_ORT_SESSION_OPTIONS_H

#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "ep_device.h"

namespace php_ort {

enum class ExecutionProviderKind {
    Builtin,
    EpDevice,
};

/**
 * One EP entry, kept in call order.  Built-in entries carry a provider name
 * and provider options.  Device entries carry an EpDeviceInfo snapshot, never
 * an Env-owned OrtEpDevice*.
 */
struct ExecutionProviderConfig {
    ExecutionProviderKind kind = ExecutionProviderKind::Builtin;
    std::string name;
    std::unordered_map<std::string, std::string> options;
    EpDeviceInfo device;
};

/**
 * Framework-independent snapshot of the configurable SessionOptions.
 *
 * A Session copies this snapshot into a fresh Ort::SessionOptions while it
 * is being created.  Mutating a PHP SessionOptions object afterwards has no
 * effect on an existing Session; there is no shared native options object.
 */
struct SessionConfig {
    int intra_op_num_threads = 0;  // 0 keeps the ORT default
    int inter_op_num_threads = 0;  // 0 keeps the ORT default
    std::string graph_optimization_level = "all";
    std::string execution_mode = "sequential";
    int log_severity_level = ORT_LOGGING_LEVEL_WARNING;
    bool enable_cpu_mem_arena = true;
    bool enable_mem_pattern = true;
    std::vector<ExecutionProviderConfig> execution_providers;

    /**
     * Apply the snapshot in call order.  Built-in EPs use ORT's provider
     * append API; device entries resolve the current Env-owned device and
     * use SessionOptionsAppendExecutionProvider_V2.
     */
    void applyTo(Ort::SessionOptions &options, Ort::Env &env) const;
};

/**
 * Rewrite built-in entries that name a device-backed EP into EpDevice entries.
 *
 * Older and plugin providers are often only reachable through
 * SessionOptionsAppendExecutionProvider_V2, while ORT's name-based appender
 * accepts a narrower, case-sensitive set.  This lets
 * `appendExecutionProvider('cpu', ...)` or a vendor short name resolve to the
 * matching Env device when one is enumerated.  CUDA, TensorRT and CoreML keep
 * their dedicated paths.  Runs only on ONNX Runtime 1.23+; otherwise no-op.
 */
void resolveExecutionProviderDevices(SessionConfig &config, Ort::Env &env);

}  // namespace php_ort

#endif  // PHP_ORT_SESSION_OPTIONS_H
