/*
 * Native async engine stress.  It never touches Zend: SessionState,
 * AsyncEngine, AsyncTaskState and Tensor ownership are exercised directly.
 *
 * Build/run with ASan via tests/run-async-sanitizers.sh.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "async_engine.h"
#include "ort_environment.h"
#include "session.h"

namespace {

php_ort::Tensor make_input() {
    php_ort::Tensor input;
    input.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    input.shape = {2, 3};
    input.data.resize(6 * sizeof(float));
    const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::memcpy(input.data.data(), values, sizeof(values));
    return input;
}

[[noreturn]] void fail(const std::string &message) {
    std::cerr << message << "\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char **argv) {
    const std::string model =
        argc > 1 ? argv[1] : "tests/fixtures/identity.onnx";

    php_ort::initializeEnvironment();
    auto session = std::make_shared<php_ort::SessionState>(model, nullptr);
    auto input = make_input();

    {
        php_ort::AsyncTaskState state(
            session, {input}, {"Y"});
        if (state.wait(0)) {
            fail("queued task was ready before completion");
        }
        if (!state.cancelQueued()) {
            fail("queued task was not cancellable");
        }
        if (state.cancelQueued()) {
            fail("cancelled task was cancellable twice");
        }
        if (state.status() != php_ort::AsyncTaskStatus::Cancelled) {
            fail("cancel did not reach cancelled state");
        }
        if (!state.wait(0)) {
            fail("cancelled task was not ready");
        }
    }

    auto engine = std::make_shared<php_ort::AsyncEngine>(2, 64);
    std::vector<std::shared_ptr<php_ort::AsyncTaskState>> tasks;
    for (int i = 0; i < 32; ++i) {
        tasks.push_back(engine->submit(session, {input}));
    }

    for (const auto &task : tasks) {
        if (!task->wait(10000)) {
            fail("async task did not finish");
        }
        if (task->status() != php_ort::AsyncTaskStatus::Completed) {
            fail("async task failed");
        }
        auto outputs = task->resultCopy();
        if (outputs.size() != 1 || outputs[0].shape != std::vector<int64_t>({2, 3})) {
            fail("async task returned an unexpected output");
        }
    }

    // Submission after stopping must be rejected.
    engine->shutdown();
    try {
        engine->submit(session, {input});
        fail("submit succeeded after shutdown");
    } catch (const php_ort::ConversionError &) {
    }

    session.reset();
    php_ort::shutdownEnvironment();
    std::cout << "native-async-stress-ok\n";
    return 0;
}
