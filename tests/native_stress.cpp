/*
 * Native allocation stress for Tensor payloads.  Run under ASan/LSan:
 *
 *   tests/run-sanitizers.sh
 *
 * This never touches Zend: it exercises std::vector/std::string ownership,
 * deep copy, move and failure-path destruction directly.  It intentionally
 * does not construct Ort::Value; PHP OrtValue ownership/lifetime is covered
 * by tests/013-ortvalue-lifetime-and-errors.phpt and the extension stress
 * script.
 */

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "onnx_tensor.h"

namespace {

php_ort::Tensor make_numeric() {
    php_ort::Tensor tensor;
    tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    tensor.shape = {128, 128};
    tensor.data.assign(128 * 128 * sizeof(float), 0);
    return tensor;
}

php_ort::Tensor make_strings() {
    php_ort::Tensor tensor;
    tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    tensor.shape = {64};
    tensor.strings.reserve(64);
    for (int i = 0; i < 64; ++i) {
        tensor.strings.emplace_back(128, static_cast<char>('a' + (i % 26)));
    }
    return tensor;
}


void check_element_count_contract() {
    auto expect_count = [](std::initializer_list<int64_t> shape,
                           size_t expected) {
        php_ort::Tensor tensor;
        tensor.shape = shape;
        if (tensor.elementCount() != expected) {
            std::cout << "element-count-mismatch\n";
            std::exit(1);
        }
    };
    auto expect_rejected = [](std::initializer_list<int64_t> shape) {
        php_ort::Tensor tensor;
        tensor.shape = shape;
        try {
            tensor.elementCount();
        } catch (const php_ort::ConversionError &) {
            return;
        }
        std::cout << "invalid-element-count-not-rejected\n";
        std::exit(1);
    };

    auto expect_invalid_tensor = [](std::initializer_list<int64_t> shape) {
        php_ort::Tensor tensor;
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        tensor.shape = shape;
        try {
            tensor.validate();
        } catch (const php_ort::ConversionError &) {
            return;
        }
        std::cout << "invalid-tensor-not-rejected\n";
        std::exit(1);
    };

    expect_count({}, 1);  // scalar
    expect_count({2, 3}, 6);
    expect_count({0, std::numeric_limits<int64_t>::max()}, 0);
    expect_rejected({0, -1});
    expect_rejected({-1, 0});
    expect_rejected({3, std::numeric_limits<int64_t>::max()});

    // Exercise the full Tensor validation path, not only elementCount().
    expect_invalid_tensor({0, -1});
    expect_invalid_tensor({-1, 0});
}

}  // namespace

int main() {
    check_element_count_contract();

    for (int i = 0; i < 2000; ++i) {
        php_ort::Tensor tensor = make_numeric();
        tensor.validate();

        php_ort::Tensor copy = tensor;
        copy.validate();

        php_ort::Tensor moved = std::move(copy);
        moved.validate();

        // A moved-from Tensor is not required to satisfy validate(); it must
        // remain safe to destroy and to reassign.
        copy = make_numeric();
        copy.validate();

        php_ort::Tensor strings = make_strings();
        strings.validate();
        php_ort::Tensor strings_copy = strings;
        strings_copy.validate();

        php_ort::Tensor strings_moved = std::move(strings_copy);
        strings_moved.validate();
        strings_copy = make_strings();
        strings_copy.validate();

        try {
            php_ort::Tensor bad = make_numeric();
            bad.data.resize(bad.data.size() - 1);
            bad.validate();
            std::cout << "bad-payload-not-rejected\n";
            return 1;
        } catch (const php_ort::ConversionError &) {
        }
    }

    std::cout << "native-stress-ok\n";
    return 0;
}
