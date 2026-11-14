#ifndef PHP_ORT_WIN_UTILS_H
#define PHP_ORT_WIN_UTILS_H

#ifdef _WIN32
#include <windows.h>

#include <string>

#include "onnx_tensor.h"

namespace php_ort {

// ORT's Windows C++ API takes wide-character paths.  PHP gives us UTF-8
// strings, so convert explicitly instead of relying on the ANSI code page.
inline std::wstring toWidePath(const std::string &utf8_path) {
    if (utf8_path.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                   utf8_path.data(),
                                   static_cast<int>(utf8_path.size()),
                                   nullptr, 0);
    if (size <= 0) {
        throw ConversionError("path is not valid UTF-8");
    }

    std::wstring wide(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path.data(),
                        static_cast<int>(utf8_path.size()), wide.data(), size);
    return wide;
}

}  // namespace php_ort

#endif  // _WIN32

#endif  // PHP_ORT_WIN_UTILS_H
