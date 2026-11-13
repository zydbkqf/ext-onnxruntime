#ifndef PHP_ORT_ASYNC_ENGINE_H
#define PHP_ORT_ASYNC_ENGINE_H

#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "session.h"

namespace php_ort {

enum class AsyncTaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
};

const char *asyncTaskStatusName(AsyncTaskStatus status) noexcept;

/**
 * Pure native task state.  It contains no zval, Zend object, or PHP callback.
 * A worker may mutate it; the PHP thread may read or wait on it.
 */
class AsyncTaskState {
public:
    AsyncTaskState(std::shared_ptr<SessionState> session,
                   std::vector<Tensor> inputs,
                   std::vector<std::string> output_names);
    ~AsyncTaskState() = default;

    AsyncTaskState(const AsyncTaskState &) = delete;
    AsyncTaskState &operator=(const AsyncTaskState &) = delete;

    uint64_t id() const noexcept { return id_; }
    AsyncTaskStatus status() const;
    bool isReady() const;
    bool cancelQueued();

    /**
     * Wait for completion.  timeout_ms < 0 means infinite.  Returns true when
     * completed, failed, or cancelled; a timeout leaves the state unchanged.
     */
    bool wait(int64_t timeout_ms);

    /** Worker transition; false if the task was already cancelled. */
    bool beginRunning();

    void complete(std::vector<Tensor> outputs);
    void fail(std::exception_ptr error);

    /** Copy outputs.  Only valid in Completed state. */
    std::vector<Tensor> resultCopy() const;
    std::exception_ptr error() const;
    const std::vector<std::string> &outputNames() const noexcept {
        return output_names_;
    }

    std::shared_ptr<SessionState> session() const noexcept { return session_; }
    const std::vector<Tensor> &inputs() const noexcept { return inputs_; }

private:
    const uint64_t id_;
    std::shared_ptr<SessionState> session_;
    std::vector<Tensor> inputs_;
    std::vector<std::string> output_names_;
    std::vector<Tensor> outputs_;
    std::exception_ptr error_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    AsyncTaskStatus status_ = AsyncTaskStatus::Queued;
};

/**
 * Process-level bounded worker pool.  Workers only touch native C++ state.
 */
class AsyncEngine {
public:
    AsyncEngine(size_t worker_count, size_t queue_capacity);
    ~AsyncEngine();

    AsyncEngine(const AsyncEngine &) = delete;
    AsyncEngine &operator=(const AsyncEngine &) = delete;

    std::shared_ptr<AsyncTaskState> submit(
        std::shared_ptr<SessionState> session,
        std::vector<Tensor> inputs);

    void shutdown();

    size_t workerCount() const noexcept { return worker_count_; }
    size_t queueCapacity() const noexcept { return queue_capacity_; }
    size_t queuedCount() const;
    size_t runningCount() const;
    bool isShuttingDown() const;

private:
    void workerLoop();

    const size_t worker_count_;
    const size_t queue_capacity_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<AsyncTaskState>> queue_;
    std::vector<std::thread> workers_;
    size_t running_ = 0;
    bool stopping_ = false;
};

// Process-level configuration and lifecycle.  The engine starts lazily on the
// first submit, so setAsyncThreads/setAsyncQueueCapacity can be called first.
void configureAsyncEngine(size_t worker_count, size_t queue_capacity);
size_t asyncEngineWorkerCount();
size_t asyncEngineQueueCapacity();
bool asyncEngineRunning();
size_t asyncEngineQueuedCount();
size_t asyncEngineRunningCount();

std::shared_ptr<AsyncTaskState> submitAsyncTask(
    std::shared_ptr<SessionState> session,
    std::vector<Tensor> inputs);

/** Cancel queued work, wait for running work, and join all workers. */
void shutdownAsyncEngine();

}  // namespace php_ort

#endif  // PHP_ORT_ASYNC_ENGINE_H
