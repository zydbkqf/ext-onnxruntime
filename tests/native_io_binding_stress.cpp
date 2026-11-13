/*
 * Native CPU I/O Binding stress.  It exercises IoBindingState, input deep
 * copies, output ownership and dynamic output shapes without Zend.
 *
 * Build/run with ASan via tests/run-io-binding-sanitizers.sh.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "io_binding.h"
#include "ort_environment.h"
#include "session.h"

namespace {

php_ort::Tensor make_input(int rows) {
    php_ort::Tensor input;
    input.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    input.shape = {rows, 3};
    input.data.resize(static_cast<size_t>(rows) * 3 * sizeof(float));
    std::vector<float> values(input.shape[0] * 3);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) + 1.0f;
    }
    std::memcpy(input.data.data(), values.data(), values.size() * sizeof(float));
    return input;
}

php_ort::Tensor make_output_buffer() {
    php_ort::Tensor buffer;
    buffer.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    buffer.shape = {2, 3};
    buffer.data.assign(6 * sizeof(float), 0);
    return buffer;
}

[[noreturn]] void fail(const std::string &message) {
    std::cerr << message << "\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char **argv) {
    const std::string identity =
        argc > 1 ? argv[1] : "tests/fixtures/identity.onnx";
    const std::string dynamic =
        argc > 2 ? argv[2] : "tests/fixtures/dynamic.onnx";

    php_ort::initializeEnvironment();

    auto session = std::make_shared<php_ort::SessionState>(identity, nullptr);
    auto binding = std::make_unique<php_ort::IoBindingState>(
        session, session->nativeSession());
    auto output_buffer = make_output_buffer();
    binding->bindOutputBuffer("Y", output_buffer);
    // The binding owns its own copy of the buffer template.
    output_buffer = php_ort::Tensor();

    for (int i = 0; i < 150; ++i) {
        auto input = make_input(2);
        binding->bindInput("X", input);
        // The source Tensor may be destroyed right after bindInput().
        input = php_ort::Tensor();
        auto outputs = binding->run();
        if (outputs.size() != 1 || outputs[0].shape != std::vector<int64_t>({2, 3})) {
            fail("identity binding returned an unexpected output");
        }
        if (binding->outputs().size() != 1) {
            fail("binding did not retain its output");
        }
    }

    {
        auto dynamic_session =
            std::make_shared<php_ort::SessionState>(dynamic, nullptr);
        auto dynamic_binding = std::make_unique<php_ort::IoBindingState>(
            dynamic_session, dynamic_session->nativeSession());
        dynamic_binding->bindOutput("Y");
        for (int rows = 1; rows <= 30; ++rows) {
            dynamic_binding->bindInput("X", make_input(rows));
            auto outputs = dynamic_binding->run();
            if (outputs.size() != 1 ||
                outputs[0].shape != std::vector<int64_t>({rows, 3})) {
                fail("dynamic binding reused a stale output shape");
            }
        }
    }

    // The binding owns shared_ptr<SessionState>, so its outputs stay valid
    // after the PHP-facing Session wrapper would have been destroyed.
    session.reset();
    if (binding->outputs().size() != 1) {
        fail("binding output was lost with SessionState");
    }
    binding.reset();

    php_ort::shutdownEnvironment();
    std::cout << "native-io-binding-stress-ok\n";
    return 0;
}
