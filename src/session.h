#ifndef PHP_ORT_SESSION_H
#define PHP_ORT_SESSION_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "ep_device.h"
#include "onnx_tensor.h"

namespace php_ort {

struct SessionConfig;
class IoBindingState;

/**
 * Model I/O metadata.  A negative dimension means "dynamic/unknown"; the
 * Zend layer renders those as PHP null, never as zero.
 */
struct TensorInfo {
    std::string name;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> shape;
};

/**
 * Native Session state.  Ownership of the Ort::Session, Env reference, and
 * PluginRegistrationLease lives here, not in the PHP wrapper.  A future Task
 * can hold shared_ptr<SessionState> and keep the plugin lease alive after the
 * PHP Session object is destroyed.
 */
class SessionState {
public:
    explicit SessionState(const std::string &model_path,
                          const SessionConfig *config = nullptr);
    ~SessionState();

    SessionState(const SessionState &) = delete;
    SessionState &operator=(const SessionState &) = delete;
    SessionState(SessionState &&) = delete;
    SessionState &operator=(SessionState &&) = delete;

    const std::vector<TensorInfo> &inputs() const;
    const std::vector<TensorInfo> &outputs() const;

    /** Exact model-name lookup; throws ConversionError when unknown. */
    size_t inputIndex(const std::string &name) const;
    size_t outputIndex(const std::string &name) const;

    void validateInput(size_t index, const Tensor &tensor) const;
    void validateOutput(size_t index, const Tensor &tensor) const;

    /** Native session used by an I/O binding created from this state. */
    Ort::Session &nativeSession() noexcept { return session; }

    /**
     * Runs one synchronous inference.  `input_tensors` are in model input
     * order and must point to tensors alive for the whole call.  Results are
     * deep-copied into owned Tensors before the local OrtValues are destroyed.
     */
    std::vector<Tensor> run(const std::vector<const Tensor *> &input_tensors);

    /**
     * PHP-thread validation plus deep copy for async submission.  The
     * returned Tensors own numeric/string payload independently of the PHP
     * Tensor objects.
     */
    std::vector<Tensor> copyInputsForAsync(
        const std::vector<const Tensor *> &input_tensors) const;

private:
    struct PluginLease {
        std::vector<std::string> registrations;

        PluginLease() = default;
        PluginLease(const PluginLease &) = delete;
        PluginLease &operator=(const PluginLease &) = delete;
        ~PluginLease();

        void acquire(const std::vector<EpDeviceInfo> &devices);
    };

    // Declared before `session`: keeps the Env alive for the Ort::Session,
    // even if the module-level reference is released first.
    std::shared_ptr<Ort::Env> env;

    // Declared before `session` so it is destroyed after the native
    // Ort::Session.  ORT must keep the plugin library loaded until all
    // Sessions using it are gone.
    PluginLease plugin_lease;

    Ort::Session session;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
};

/**
 * Thin PHP-facing wrapper.  The native lifetime is held by SessionState; the
 * PHP object itself does not own the ORT session or plugin lease directly.
 */
class Session {
public:
    explicit Session(const std::string &model_path,
                     const SessionConfig *config = nullptr);

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    const std::vector<TensorInfo> &inputs() const;
    const std::vector<TensorInfo> &outputs() const;
    std::vector<Tensor> run(const std::vector<const Tensor *> &input_tensors);
    std::vector<Tensor> copyInputsForAsync(
        const std::vector<const Tensor *> &input_tensors) const;

    std::shared_ptr<IoBindingState> createIoBinding();

    /** Shared native state; a future Task can retain this independently. */
    const std::shared_ptr<SessionState> &state() const;

private:
    std::shared_ptr<SessionState> state_;
};

}  // namespace php_ort

#endif  // PHP_ORT_SESSION_H
