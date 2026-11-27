#include "ort_environment.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "onnx_tensor.h"

namespace php_ort {
namespace {
std::mutex g_environment_mutex;
std::shared_ptr<Ort::Env> g_environment;

/* The process-wide Env logger reports ONNX Runtime internals such as device
 * discovery.  Those messages go straight to stderr, where they break PHPT
 * comparisons and pollute CLI/php-fpm output, so errors are the default and
 * warnings stay opt-in through ONNXRUNTIME_LOG_LEVEL. */
OrtLoggingLevel environmentLogLevel() {
    const char *raw = std::getenv("ONNXRUNTIME_LOG_LEVEL");
    if (raw == nullptr || *raw == '\0') {
        return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_ERROR);
    }
    std::string value(raw);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.erase(0, 1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                              value.back() == '\n' || value.back() == '\r')) {
        value.pop_back();
    }
    for (char &c : value) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "0" || value == "verbose") {
        return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_VERBOSE);
    }
    if (value == "1" || value == "info") {
        return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_INFO);
    }
    if (value == "2" || value == "warning") {
        return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_WARNING);
    }
    if (value == "4" || value == "fatal") {
        return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_FATAL);
    }
    return static_cast<OrtLoggingLevel>(ORT_LOGGING_LEVEL_ERROR);
}
}  // namespace

void initializeEnvironment() {
    std::lock_guard<std::mutex> lock(g_environment_mutex);
    if (g_environment == nullptr) {
        g_environment = std::make_shared<Ort::Env>(
            environmentLogLevel(), "php_onnxruntime");
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
