#include "ort_environment.h"

#include <memory>
#include <mutex>

#include "onnx_tensor.h"

namespace php_ort {
namespace {
std::mutex g_environment_mutex;
std::shared_ptr<Ort::Env> g_environment;
}  // namespace

void initializeEnvironment() {
    std::lock_guard<std::mutex> lock(g_environment_mutex);
    if (g_environment == nullptr) {
        g_environment = std::make_shared<Ort::Env>(
            ORT_LOGGING_LEVEL_WARNING, "php_onnxruntime");
    }
}

void shutdownEnvironment() {
    std::lock_guard<std::mutex> lock(g_environment_mutex);
    // Sessions hold their own shared_ptr, so releasing the process-wide
    // reference here cannot invalidate a live Ort::Session.
    g_environment.reset();
}

std::shared_ptr<Ort::Env> environment() {
    std::lock_guard<std::mutex> lock(g_environment_mutex);
    if (g_environment == nullptr) {
        throw ConversionError(
            "ONNX Runtime environment is not initialized");
    }
    return g_environment;
}

}  // namespace php_ort
