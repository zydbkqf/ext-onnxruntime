/*
 * Standalone probe for Tensor::validate().
 *
 * Deliberately does not include Zend/PHP headers: this is the same code path a
 * future OrtValue -> Tensor adapter would use.  Build:
 *
 *   clang++ -std=c++17 -I src -I third_party/onnxruntime/include \
 *       tests/validator_probe.cpp src/onnx_tensor.cpp -o /tmp/ort_validator_probe
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "onnx_tensor.h"

namespace {

int failures = 0;

void expect_ok(const char *label, const php_ort::Tensor &tensor) {
    try {
        tensor.validate();
        std::cout << label << ":OK\n";
    } catch (const php_ort::ConversionError &error) {
        std::cout << label << ":UNEXPECTED-THROW(" << error.what() << ")\n";
        ++failures;
    }
}

void expect_reject(const char *label, const php_ort::Tensor &tensor) {
    try {
        tensor.validate();
        std::cout << label << ":NO-THROW\n";
        ++failures;
    } catch (const php_ort::ConversionError &) {
        std::cout << label << ":REJECTED\n";
    }
}

php_ort::Tensor numeric_tensor(ONNXTensorElementDataType type,
                               std::vector<int64_t> shape, size_t bytes) {
    php_ort::Tensor tensor;
    tensor.type = type;
    tensor.shape = std::move(shape);
    tensor.data.assign(bytes, 0);
    return tensor;
}

}  // namespace

int main() {
    expect_ok("valid-int64", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {2}, 16));
    expect_ok("valid-empty", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {0}, 0));
    expect_ok("valid-empty-inner-dim", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 0}, 0));

    expect_reject("undefined-type", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, {1}, 0));

    expect_reject("short-payload", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}, 4));

    expect_reject("long-payload", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {2}, 3));

    expect_reject("negative-dim", numeric_tensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {2, -1}, 0));

    expect_reject("string-count-mismatch", [] {
        php_ort::Tensor tensor;
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        tensor.shape = {2};
        tensor.strings = {"only-one"};
        return tensor;
    }());

    expect_reject("string-with-numeric-payload", [] {
        php_ort::Tensor tensor;
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        tensor.shape = {1};
        tensor.strings = {"a"};
        tensor.data.assign(1, 0);
        return tensor;
    }());

    expect_reject("numeric-with-string-payload", [] {
        php_ort::Tensor tensor;
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        tensor.shape = {1};
        tensor.data.assign(4, 0);
        tensor.strings = {"a"};
        return tensor;
    }());

    std::cout << (failures == 0 ? "probe-passed\n" : "probe-failed\n");
    return failures == 0 ? 0 : 1;
}
