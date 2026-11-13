#ifndef PHP_ORT_IO_BINDING_H
#define PHP_ORT_IO_BINDING_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "onnx_tensor.h"

namespace php_ort {

class SessionState;

/**
 * Native I/O Binding state.
 *
 * The binding owns a reference to SessionState, so it cannot outlive the
 * native Ort::Session even if the PHP Session object is destroyed first.
 * Inputs are deep-copied into this object before they are bound; numeric
 * OrtValues borrow those owned copies, so the caller may destroy the PHP
 * Tensor immediately after bindInput().
 */
class IoBindingState {
public:
    IoBindingState(std::shared_ptr<SessionState> session,
                   Ort::Session &native_session);
    ~IoBindingState();

    IoBindingState(const IoBindingState &) = delete;
    IoBindingState &operator=(const IoBindingState &) = delete;
    IoBindingState(IoBindingState &&) = delete;
    IoBindingState &operator=(IoBindingState &&) = delete;

    void bindInput(const std::string &name, const Tensor &tensor);
    void bindOutput(const std::string &name);
    void bindOutputBuffer(const std::string &name, const Tensor &buffer);

    /** Run the bound graph and copy outputs into owned Tensors. */
    std::vector<Tensor> run();

    /** Outputs from the latest run(); throws before the first run. */
    const std::vector<Tensor> &outputs() const;

    const std::shared_ptr<SessionState> &session() const noexcept {
        return session_;
    }

private:
    void rebindOutputs();

    std::shared_ptr<SessionState> session_;

    // Keep a deep copy of every input.  Numeric OrtValues below borrow these
    // payloads, so both vectors are reserved/resized before binding and are
    // never reallocated afterwards.
    std::vector<Tensor> input_tensors_;
    std::vector<std::unique_ptr<Ort::Value>> input_values_;
    std::vector<bool> input_bound_;

    std::vector<bool> output_bound_;

    // Phase 7B: optional preallocated outputs.  The Tensor is an owned copy of
    // the caller's shape/dtype template, and the Ort::Value borrows that copy.
    // Declared before binding_ so the binding is destroyed first.
    std::vector<Tensor> output_tensors_;
    std::vector<std::unique_ptr<Ort::Value>> output_values_;
    std::vector<bool> output_preallocated_;

    // Declared before binding_: the binding may retain this memory-info
    // pointer for the lifetime of the IoBinding.
    Ort::MemoryInfo memory_info_;
    Ort::IoBinding binding_;

    std::vector<Tensor> outputs_;
    bool has_run_ = false;
};

}  // namespace php_ort

#endif  // PHP_ORT_IO_BINDING_H
