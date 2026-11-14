#include "io_binding.h"

#include <utility>

#include "ort_value_converter.h"
#include "runtime.h"
#include "session.h"

namespace php_ort {

IoBindingState::IoBindingState(std::shared_ptr<SessionState> session,
                               Ort::Session &native_session)
    : session_(std::move(session)),
      input_tensors_(session_->inputs().size()),
      input_values_(session_->inputs().size()),
      input_bound_(session_->inputs().size(), false),
      output_bound_(session_->outputs().size(), false),
      output_tensors_(session_->outputs().size()),
      output_values_(session_->outputs().size()),
      output_preallocated_(session_->outputs().size(), false),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                               OrtMemTypeDefault)),
      binding_(native_session) {}

IoBindingState::~IoBindingState() = default;

void IoBindingState::ensureMutableLocked() const {
    if (async_submitted_) {
        throw ConversionError(
            "I/O binding is owned by an async Task; read outputs with "
            "Task::getResult() and create a new binding for another run");
    }
    if (running_) {
        throw ConversionError("I/O binding is already running");
    }
}

void IoBindingState::markAsyncSubmitted() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (async_submitted_) {
        throw ConversionError(
            "I/O binding has already been submitted to an async Task");
    }
    if (running_) {
        throw ConversionError("I/O binding is already running");
    }
    async_submitted_ = true;
}

void IoBindingState::clearAsyncSubmitted() {
    std::lock_guard<std::mutex> lock(mutex_);
    async_submitted_ = false;
}

void IoBindingState::bindInput(const std::string &name,
                               const Tensor &tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureMutableLocked();
    size_t index = session_->inputIndex(name);
    session_->validateInput(index, tensor);

    // Build the temporary copy and OrtValue before mutating the binding, so
    // a failed conversion leaves the previously bound input intact.
    Tensor owned = tensor;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Value> value =
        std::make_unique<Ort::Value>(
            tensorToOrtValue(owned, memory_info_, allocator));

    input_tensors_[index] = std::move(owned);
    input_values_[index] = std::move(value);
    binding_.BindInput(name.c_str(), *input_values_[index]);
    input_bound_[index] = true;
}

void IoBindingState::bindOutput(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureMutableLocked();
    size_t index = session_->outputIndex(name);
    // BindOutputToDevice lets ORT allocate this output on the CPU device.
    // Phase 7B uses a separate bindOutputBuffer() path for preallocation.
    Ort::ConstMemoryInfo cpu_memory = memory_info_.GetConst();
    binding_.BindOutput(name.c_str(), cpu_memory);
    output_bound_[index] = true;
    output_preallocated_[index] = false;
    output_values_[index].reset();
    output_tensors_[index] = Tensor();
}

void IoBindingState::bindOutputBuffer(const std::string &name,
                                      const Tensor &buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureMutableLocked();
    size_t index = session_->outputIndex(name);
    const TensorInfo &info = session_->outputs()[index];

    // A preallocated output is only valid for a fixed-shape output.  Dynamic
    // outputs need a fresh allocation per run; Phase 7A's device-bound path
    // already handles that correctly.
    for (int64_t dim : info.shape) {
        if (dim < 0) {
            throw ConversionError(
                "output \"" + name +
                "\" has a dynamic shape and cannot use a preallocated buffer");
        }
    }
    if (buffer.isStringTensor()) {
        throw ConversionError(
            "string output \"" + name +
            "\" cannot use a preallocated numeric buffer");
    }
    session_->validateOutput(index, buffer);

    // Keep a private copy: ORT will write into this buffer, while the caller's
    // PHP Tensor must remain untouched and may be destroyed immediately.
    Tensor owned = buffer;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Value> value = std::make_unique<Ort::Value>(
        tensorToOrtValue(owned, memory_info_, allocator));

    binding_.BindOutput(name.c_str(), *value);
    output_tensors_[index] = std::move(owned);
    output_values_[index] = std::move(value);
    output_preallocated_[index] = true;
    output_bound_[index] = true;
}

void IoBindingState::rebindOutputs() {
    // A previous run leaves the allocated output value attached to the
    // IoBinding.  ORT would then treat that value as a preallocated output on
    // the next run, which breaks models whose output shapes change.  Rebind
    // the requested outputs to the CPU device for every run so ORT allocates
    // fresh output values for the device-bound path.
    //
    // Phase 7B outputs are intentionally different: their owned buffers are
    // reused for every run, so they are bound as values instead of device
    // targets.
    binding_.ClearBoundOutputs();
    Ort::ConstMemoryInfo cpu_memory = memory_info_.GetConst();
    for (size_t i = 0; i < output_bound_.size(); ++i) {
        if (!output_bound_[i]) {
            continue;
        }
        if (output_preallocated_[i]) {
            if (output_values_[i] == nullptr) {
                throw ConversionError(
                    "preallocated output \"" + session_->outputs()[i].name +
                    "\" has no OrtValue");
            }
            binding_.BindOutput(session_->outputs()[i].name.c_str(),
                                *output_values_[i]);
        } else {
            binding_.BindOutput(session_->outputs()[i].name.c_str(),
                                cpu_memory);
        }
    }
}

std::vector<Tensor> IoBindingState::runUnlocked() {
    for (size_t i = 0; i < input_bound_.size(); ++i) {
        if (!input_bound_[i]) {
            throw ConversionError(
                "input \"" + session_->inputs()[i].name +
                "\" is not bound");
        }
    }
    for (size_t i = 0; i < output_bound_.size(); ++i) {
        if (!output_bound_[i]) {
            throw ConversionError(
                "output \"" + session_->outputs()[i].name +
                "\" is not bound");
        }
    }

    // Every input copy is complete before ORT starts, and ORT must finish
    // copying/uploads before Run() is allowed to read them.
    binding_.SynchronizeInputs();
    rebindOutputs();
    ensureRuntimeAcceptingWork();
    session_->nativeSession().Run(Ort::RunOptions{}, binding_);
    // Run() returning is not proof that an EP has finished writing outputs.
    // SynchronizeOutputs() is the completion point for the copy-back path;
    // only after it may GetOutputValues() or the owned output buffers be
    // destroyed.
    binding_.SynchronizeOutputs();

    std::vector<Ort::Value> values = binding_.GetOutputValues();
    if (values.size() != session_->outputs().size()) {
        throw ConversionError(
            "ORT returned an unexpected number of bound outputs");
    }

    std::vector<Tensor> results;
    results.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        Tensor tensor = ortValueToTensor(values[i]);
        session_->validateOutput(i, tensor);
        results.push_back(std::move(tensor));
    }

    outputs_ = results;
    has_run_ = true;
    return results;
}

std::vector<Tensor> IoBindingState::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    ensureMutableLocked();
    running_ = true;
    lock.unlock();

    try {
        std::vector<Tensor> results = runUnlocked();
        lock.lock();
        running_ = false;
        return results;
    } catch (...) {
        lock.lock();
        running_ = false;
        throw;
    }
}

std::vector<Tensor> IoBindingState::runForAsync() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!async_submitted_) {
        throw ConversionError(
            "I/O binding was not submitted to an async Task");
    }
    if (running_) {
        throw ConversionError("I/O binding is already running");
    }
    running_ = true;
    lock.unlock();

    try {
        std::vector<Tensor> results = runUnlocked();
        lock.lock();
        running_ = false;
        return results;
    } catch (...) {
        lock.lock();
        running_ = false;
        throw;
    }
}

std::vector<Tensor> IoBindingState::outputs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (async_submitted_) {
        throw ConversionError(
            "I/O binding outputs are owned by an async Task; use "
            "Task::getResult() after Task::wait()");
    }
    if (running_) {
        throw ConversionError("I/O binding is already running");
    }
    if (!has_run_) {
        throw ConversionError(
            "I/O binding has not run; call Session::runWithBinding() first");
    }
    return outputs_;
}

}  // namespace php_ort
