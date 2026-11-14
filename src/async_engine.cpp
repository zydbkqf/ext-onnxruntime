#include "async_engine.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "onnx_tensor.h"
#include "io_binding.h"
#include "runtime.h"

namespace php_ort {
namespace {

std::mutex g_engine_mutex;
std::shared_ptr<AsyncEngine> g_engine;
size_t g_worker_count = 2;
size_t g_queue_capacity = 64;

std::atomic<uint64_t> g_next_task_id{1};

void validateConfiguration(size_t worker_count, size_t queue_capacity) {
    if (worker_count == 0 || worker_count > 64) {
        throw ConversionError("async worker count must be between 1 and 64");
    }
    if (queue_capacity == 0 || queue_capacity > 1000000) {
        throw ConversionError(
            "async queue capacity must be between 1 and 1000000");
    }
}

}  // namespace

const char *asyncTaskStatusName(AsyncTaskStatus status) noexcept {
    switch (status) {
        case AsyncTaskStatus::Queued:
            return "queued";
        case AsyncTaskStatus::Running:
            return "running";
        case AsyncTaskStatus::Completed:
            return "completed";
        case AsyncTaskStatus::Failed:
            return "failed";
        case AsyncTaskStatus::Cancelled:
            return "cancelled";
    }
    return "unknown";
}

AsyncTaskState::AsyncTaskState(std::shared_ptr<SessionState> session,
                               std::vector<Tensor> inputs,
                               std::vector<std::string> output_names)
    : id_(g_next_task_id.fetch_add(1, std::memory_order_relaxed)),
      session_(std::move(session)),
      inputs_(std::move(inputs)),
      output_names_(std::move(output_names)) {}

AsyncTaskState::AsyncTaskState(std::shared_ptr<SessionState> session,
                               std::shared_ptr<IoBindingState> binding)
    : id_(g_next_task_id.fetch_add(1, std::memory_order_relaxed)),
      session_(std::move(session)),
      binding_(std::move(binding)) {
    output_names_.reserve(session_->outputs().size());
    for (const TensorInfo &info : session_->outputs()) {
        output_names_.push_back(info.name);
    }
}

AsyncTaskStatus AsyncTaskState::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool AsyncTaskState::isReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_ == AsyncTaskStatus::Completed ||
           status_ == AsyncTaskStatus::Failed ||
           status_ == AsyncTaskStatus::Cancelled;
}

bool AsyncTaskState::cancelQueued() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ != AsyncTaskStatus::Queued) {
        return false;
    }
    status_ = AsyncTaskStatus::Cancelled;
    cv_.notify_all();
    return true;
}

bool AsyncTaskState::wait(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto ready = [this]() {
        return status_ == AsyncTaskStatus::Completed ||
               status_ == AsyncTaskStatus::Failed ||
               status_ == AsyncTaskStatus::Cancelled;
    };
    if (timeout_ms < 0) {
        cv_.wait(lock, ready);
        return true;
    }
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
}

bool AsyncTaskState::beginRunning() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ != AsyncTaskStatus::Queued) {
        return false;
    }
    status_ = AsyncTaskStatus::Running;
    return true;
}

void AsyncTaskState::complete(std::vector<Tensor> outputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ != AsyncTaskStatus::Running) {
        return;
    }
    outputs_ = std::move(outputs);
    error_ = nullptr;
    status_ = AsyncTaskStatus::Completed;
    cv_.notify_all();
}

void AsyncTaskState::fail(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ != AsyncTaskStatus::Running) {
        return;
    }
    error_ = error;
    status_ = AsyncTaskStatus::Failed;
    cv_.notify_all();
}

std::vector<Tensor> AsyncTaskState::resultCopy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ != AsyncTaskStatus::Completed) {
        throw ConversionError("async task has no completed result");
    }
    return outputs_;
}

std::exception_ptr AsyncTaskState::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

AsyncEngine::AsyncEngine(size_t worker_count, size_t queue_capacity)
    : worker_count_(worker_count), queue_capacity_(queue_capacity) {
    validateConfiguration(worker_count_, queue_capacity_);
    workers_.reserve(worker_count_);
    try {
        for (size_t i = 0; i < worker_count_; ++i) {
            workers_.emplace_back(&AsyncEngine::workerLoop, this);
        }
    } catch (...) {
        // The object's destructor will not run when a constructor throws.
        // Wake and join every worker that was already created, otherwise
        // std::vector<std::thread> would destroy joinable threads and call
        // std::terminate().
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            for (const std::shared_ptr<AsyncTaskState> &task : queue_) {
                task->cancelQueued();
            }
            queue_.clear();
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        throw;
    }
}

AsyncEngine::~AsyncEngine() {
    shutdown();
}

std::shared_ptr<AsyncTaskState> AsyncEngine::submit(
    std::shared_ptr<SessionState> session,
    std::vector<Tensor> inputs) {
    std::vector<std::string> output_names;
    output_names.reserve(session->outputs().size());
    for (const TensorInfo &info : session->outputs()) {
        output_names.push_back(info.name);
    }

    auto task = std::make_shared<AsyncTaskState>(
        std::move(session), std::move(inputs), std::move(output_names));

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
        throw ConversionError("async engine is shutting down");
    }
    if (queue_.size() >= queue_capacity_) {
        throw ConversionError("async task queue is full");
    }
    queue_.push_back(task);
    cv_.notify_one();
    return task;
}

std::shared_ptr<AsyncTaskState> AsyncEngine::submitBinding(
    std::shared_ptr<SessionState> session,
    std::shared_ptr<IoBindingState> binding) {
    auto task = std::make_shared<AsyncTaskState>(
        std::move(session), std::move(binding));

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
        throw ConversionError("async engine is shutting down");
    }
    if (queue_.size() >= queue_capacity_) {
        throw ConversionError("async task queue is full");
    }
    queue_.push_back(task);
    cv_.notify_one();
    return task;
}

bool AsyncEngine::cancelTask(const std::shared_ptr<AsyncTaskState> &task) {
    if (task == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(
        queue_.begin(), queue_.end(),
        [&task](const std::shared_ptr<AsyncTaskState> &queued) {
            return queued->id() == task->id();
        });
    bool removed = false;
    if (it != queue_.end()) {
        queue_.erase(it);
        removed = true;
    }
    bool cancelled = task->cancelQueued();
    if (removed) {
        cv_.notify_all();
    }
    return cancelled;
}

void AsyncEngine::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ && workers_.empty()) {
            return;
        }
        stopping_ = true;
        for (const std::shared_ptr<AsyncTaskState> &task : queue_) {
            task->cancelQueued();
        }
        queue_.clear();
    }
    cv_.notify_all();

    for (std::thread &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

size_t AsyncEngine::queuedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

size_t AsyncEngine::runningCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

bool AsyncEngine::isShuttingDown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopping_;
}

void AsyncEngine::workerLoop() {
    for (;;) {
        std::shared_ptr<AsyncTaskState> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return stopping_ || !queue_.empty();
            });
            if (stopping_ && queue_.empty()) {
                return;
            }
            task = queue_.front();
            queue_.pop_front();
            ++running_;
        }

        if (!task->beginRunning()) {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_;
            cv_.notify_all();
            continue;
        }

        try {
            std::vector<Tensor> outputs;
            if (task->binding() != nullptr) {
                // Worker touches only native buffers owned by the binding.
                outputs = task->binding()->runForAsync();
            } else {
                const std::vector<Tensor> &inputs = task->inputs();
                std::vector<const Tensor *> pointers;
                pointers.reserve(inputs.size());
                for (const Tensor &input : inputs) {
                    pointers.push_back(&input);
                }
                outputs = task->session()->run(pointers);
            }
            task->complete(std::move(outputs));
        } catch (...) {
            task->fail(std::current_exception());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_;
            cv_.notify_all();
        }
    }
}

void configureAsyncEngine(size_t worker_count, size_t queue_capacity) {
    validateConfiguration(worker_count, queue_capacity);
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_engine != nullptr) {
        throw ConversionError(
            "async engine is already running; configuration is immutable");
    }
    g_worker_count = worker_count;
    g_queue_capacity = queue_capacity;
}

size_t asyncEngineWorkerCount() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_engine != nullptr ? g_engine->workerCount() : g_worker_count;
}

size_t asyncEngineQueueCapacity() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_engine != nullptr ? g_engine->queueCapacity()
                               : g_queue_capacity;
}

bool asyncEngineRunning() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_engine != nullptr;
}

size_t asyncEngineQueuedCount() {
    std::shared_ptr<AsyncEngine> engine;
    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        engine = g_engine;
    }
    return engine == nullptr ? 0 : engine->queuedCount();
}

size_t asyncEngineRunningCount() {
    std::shared_ptr<AsyncEngine> engine;
    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        engine = g_engine;
    }
    return engine == nullptr ? 0 : engine->runningCount();
}

std::shared_ptr<AsyncTaskState> submitAsyncTask(
    std::shared_ptr<SessionState> session,
    std::vector<Tensor> inputs) {
    ensureRuntimeAcceptingWork();

    std::shared_ptr<AsyncEngine> engine;
    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        if (g_engine == nullptr) {
            g_engine = std::make_shared<AsyncEngine>(g_worker_count,
                                                     g_queue_capacity);
        }
        engine = g_engine;
    }
    return engine->submit(std::move(session), std::move(inputs));
}

std::shared_ptr<AsyncTaskState> submitAsyncBindingTask(
    std::shared_ptr<SessionState> session,
    std::shared_ptr<IoBindingState> binding) {
    ensureRuntimeAcceptingWork();

    binding->markAsyncSubmitted();
    try {
        std::shared_ptr<AsyncEngine> engine;
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            if (g_engine == nullptr) {
                g_engine = std::make_shared<AsyncEngine>(g_worker_count,
                                                         g_queue_capacity);
            }
            engine = g_engine;
        }
        std::shared_ptr<AsyncTaskState> task =
            engine->submitBinding(std::move(session), binding);
        binding.reset();
        return task;
    } catch (...) {
        binding->clearAsyncSubmitted();
        throw;
    }
}

bool cancelAsyncTask(const std::shared_ptr<AsyncTaskState> &task) {
    if (task == nullptr) {
        return false;
    }
    std::shared_ptr<AsyncEngine> engine;
    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        engine = g_engine;
    }
    if (engine != nullptr) {
        return engine->cancelTask(task);
    }
    return task->cancelQueued();
}

void shutdownAsyncEngine() {
    std::shared_ptr<AsyncEngine> engine;
    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        engine = std::move(g_engine);
        g_engine.reset();
    }
    if (engine != nullptr) {
        engine->shutdown();
    }
}

}  // namespace php_ort
