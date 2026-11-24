#ifndef PHP_ONNXRUNTIME_H
#define PHP_ONNXRUNTIME_H

extern "C" {
#include "php.h"
}

#if PHP_VERSION_ID < 80000
#error "PHP 8.0 or newer is required"
#endif

/* PHP 8.6 removed the legacy XtOffsetOf() macro from zend_portability.h;
 * Zend itself now uses plain offsetof().  Keep the same expansion here. */
#if PHP_VERSION_ID < 80600
# define PHP_ORT_OFFSETOF(s_type, field) XtOffsetOf(s_type, field)
#else
# define PHP_ORT_OFFSETOF(s_type, field) offsetof(s_type, field)
#endif

#define PHP_ONNXRUNTIME_EXTNAME "onnxruntime"
#define PHP_ONNXRUNTIME_VERSION "0.4.12"

/* Provided by config.m4/config.w32; keep the module buildable from IDE
 * projects that do not include the generated config.h. */
#ifndef PHP_ORT_BUILD_API_VERSION
# define PHP_ORT_BUILD_API_VERSION 0
#endif
#ifndef PHP_ORT_ORT_VERSION
# define PHP_ORT_ORT_VERSION "unknown"
#endif
#ifndef PHP_ORT_PACKAGE
# define PHP_ORT_PACKAGE ""
#endif
#ifndef PHP_ORT_SOURCE
# define PHP_ORT_SOURCE ""
#endif

#define PHP_ORT_STRINGIFY_IMPL(x) #x
#define PHP_ORT_STRINGIFY(x) PHP_ORT_STRINGIFY_IMPL(x)

extern zend_module_entry onnxruntime_module_entry;
#define phpext_onnxruntime_ptr &onnxruntime_module_entry

#endif  // PHP_ONNXRUNTIME_H
