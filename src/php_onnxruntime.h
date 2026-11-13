#ifndef PHP_ONNXRUNTIME_H
#define PHP_ONNXRUNTIME_H

extern "C" {
#include "php.h"
}

#if PHP_VERSION_ID < 80000
#error "PHP 8.0 or newer is required"
#endif

#define PHP_ONNXRUNTIME_EXTNAME "onnxruntime"
#define PHP_ONNXRUNTIME_VERSION "0.4.0"

extern zend_module_entry onnxruntime_module_entry;
#define phpext_onnxruntime_ptr &onnxruntime_module_entry

#endif  // PHP_ONNXRUNTIME_H
