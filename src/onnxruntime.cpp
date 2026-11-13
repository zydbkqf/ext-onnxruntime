#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

extern "C" {
#include "php.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "ext/spl/spl_exceptions.h"
#include "ext/standard/info.h"
}

#include "php_onnxruntime.h"
// PHP defines snprintf as a macro; ORT's C++ headers use std::snprintf.
#pragma push_macro("snprintf")
#undef snprintf
#include "async_engine.h"
#include "ep_device.h"
#include "io_binding.h"
#include "runtime.h"
#include "session.h"
#include "session_options.h"
#include "ort_environment.h"
#include "ort_value_converter.h"
#pragma pop_macro("snprintf")
#include "zval_converter.h"

/* PHP 8.4 introduced zend_register_internal_class_with_flags().  Older 8.x
 * releases expose only zend_register_internal_class_ex(); set the same class
 * flags after registration so the public class surface stays identical. */
static zend_class_entry *php_ort_register_class_with_flags(
    zend_class_entry *ce, zend_class_entry *parent, uint32_t flags) {
#if PHP_VERSION_ID >= 80400
    return zend_register_internal_class_with_flags(ce, parent, flags);
#else
    zend_class_entry *registered = zend_register_internal_class_ex(ce, parent);
    if (registered != nullptr) {
        registered->ce_flags |= flags;
    }
    return registered;
#endif
}

/* ----------------------------------------------------------------------------
 * Object layout
 * -------------------------------------------------------------------------*/

typedef struct {
    php_ort::Tensor *tensor;
    zend_object std;
} php_ort_tensor_object;

static inline php_ort_tensor_object *php_ort_tensor_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_tensor_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_tensor_object, std));
}

typedef struct {
    Ort::Value *value;
    zval owner;  // numeric OrtValue borrows this Tensor's buffer
    bool valid;
    zend_object std;
} php_ort_ortvalue_object;

static inline php_ort_ortvalue_object *php_ort_ortvalue_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_ortvalue_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_ortvalue_object, std));
}

typedef struct {
    php_ort::Session *session;
    zend_object std;
} php_ort_session_object;

static inline php_ort_session_object *php_ort_session_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_session_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_session_object, std));
}

typedef struct {
    php_ort::SessionConfig *config;
    zend_object std;
} php_ort_session_options_object;

static inline php_ort_session_options_object *php_ort_session_options_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_session_options_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_session_options_object, std));
}

typedef struct {
    php_ort::EpDeviceInfo *info;
    zend_object std;
} php_ort_epdevice_object;

typedef struct {
    std::shared_ptr<php_ort::AsyncTaskState> *task;
    zend_object std;
} php_ort_task_object;

typedef struct {
    php_ort::IoBindingState *binding;
    zend_object std;
} php_ort_iobinding_object;

static inline php_ort_iobinding_object *php_ort_iobinding_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_iobinding_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_iobinding_object, std));
}

static inline php_ort_epdevice_object *php_ort_epdevice_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_epdevice_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_epdevice_object, std));
}

static inline php_ort_task_object *php_ort_task_from_obj(
    zend_object *object) {
    return reinterpret_cast<php_ort_task_object *>(
        reinterpret_cast<char *>(object) -
        XtOffsetOf(php_ort_task_object, std));
}

typedef struct {
    zend_object std;
} php_ort_runtime_object;

static zend_class_entry *onnx_tensor_ce = nullptr;
static zend_class_entry *onnx_ortvalue_ce = nullptr;
static zend_class_entry *onnx_session_ce = nullptr;
static zend_class_entry *onnx_session_options_ce = nullptr;
static zend_class_entry *onnx_epdevice_ce = nullptr;
static zend_class_entry *onnx_task_ce = nullptr;
static zend_class_entry *onnx_iobinding_ce = nullptr;
static zend_class_entry *onnx_runtime_ce = nullptr;
static zend_class_entry *onnx_exception_ce = nullptr;
static zend_class_entry *onnx_conversion_exception_ce = nullptr;
static zend_object_handlers onnx_tensor_handlers;
static zend_object_handlers onnx_ortvalue_handlers;
static zend_object_handlers onnx_session_handlers;
static zend_object_handlers onnx_session_options_handlers;
static zend_object_handlers onnx_epdevice_handlers;
static zend_object_handlers onnx_task_handlers;
static zend_object_handlers onnx_iobinding_handlers;
static zend_object_handlers onnx_runtime_handlers;

static void php_ort_throw_native(std::exception_ptr exception,
                                 const char *oom_context);

/* ----------------------------------------------------------------------------
 * Object lifecycle
 * -------------------------------------------------------------------------*/

static zend_object *onnx_tensor_create_object(zend_class_entry *ce) {
    php_ort_tensor_object *object = reinterpret_cast<php_ort_tensor_object *>(
        zend_object_alloc(sizeof(php_ort_tensor_object), ce));

    std::exception_ptr allocation_error;
    object->tensor = nullptr;
    try {
        // Default construction of the STL containers normally does not
        // allocate, but do not make Zend object creation depend on that.
        object->tensor = new php_ort::Tensor();
    } catch (...) {
        allocation_error = std::current_exception();
    }

    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_tensor_handlers;
    object_properties_init(&object->std, ce);
    if (object->tensor == nullptr) {
        if (allocation_error != nullptr) {
            php_ort_throw_native(allocation_error,
                                 "failed to allocate tensor object");
        } else {
            zend_throw_exception(onnx_conversion_exception_ce,
                                 "failed to allocate tensor object", 0);
        }
    }
    return &object->std;
}

static void onnx_tensor_free_object(zend_object *object) {
    php_ort_tensor_object *self = php_ort_tensor_from_obj(object);
    delete self->tensor;
    self->tensor = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_tensor_clone_object(zend_object *object) {
    php_ort_tensor_object *source = php_ort_tensor_from_obj(object);
    zend_object *clone = onnx_tensor_create_object(object->ce);
    php_ort_tensor_object *target = php_ort_tensor_from_obj(clone);
    if (source->tensor != nullptr && target->tensor != nullptr) {
        try {
            // Tensor copy assignment deep-copies `data` / `strings`; every
            // Tensor owns its payload, so a clone is an independent owned
            // tensor.
            *target->tensor = *source->tensor;
        } catch (...) {
            // Zend object callbacks must not unwind C++ exceptions through
            // the C ABI.  Keep the (valid, empty) clone object and let the
            // pending PHP exception abort the clone operation.
            php_ort_throw_native(std::current_exception(),
                                 "out of memory while cloning tensor");
        }
    }
    if (EG(exception) == nullptr) {
        zend_objects_clone_members(clone, object);
    }
    return clone;
}

static zend_object *onnx_ortvalue_create_object(zend_class_entry *ce) {
    php_ort_ortvalue_object *object =
        reinterpret_cast<php_ort_ortvalue_object *>(
            zend_object_alloc(sizeof(php_ort_ortvalue_object), ce));
    object->value = nullptr;
    ZVAL_UNDEF(&object->owner);
    object->valid = false;
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_ortvalue_handlers;
    object_properties_init(&object->std, ce);
    return &object->std;
}

static void onnx_ortvalue_free_object(zend_object *object) {
    php_ort_ortvalue_object *self = php_ort_ortvalue_from_obj(object);
    // Release the OrtValue first: a numeric OrtValue borrows the owner's
    // buffer, so the Tensor must outlive it.
    delete self->value;
    self->value = nullptr;
    if (!Z_ISUNDEF(self->owner)) {
        zval_ptr_dtor(&self->owner);
    }
    ZVAL_UNDEF(&self->owner);
    zend_object_std_dtor(&self->std);
}

static HashTable *onnx_ortvalue_get_gc(zend_object *object, zval **table,
                                       int *n) {
    php_ort_ortvalue_object *self = php_ort_ortvalue_from_obj(object);
    if (!Z_ISUNDEF(self->owner)) {
        *table = &self->owner;
        *n = 1;
    } else {
        *table = nullptr;
        *n = 0;
    }
    // get_gc reports both the additional internal zval and the standard
    // properties table; returning nullptr is not the normal contract.
    return zend_std_get_properties(object);
}

static zend_object *onnx_ortvalue_clone_object(zend_object *object) {
    zend_object *clone = onnx_ortvalue_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\OrtValue cannot be cloned; call toTensor() first",
        0);
    return clone;
}

static zend_object *onnx_session_create_object(zend_class_entry *ce) {
    php_ort_session_object *object =
        reinterpret_cast<php_ort_session_object *>(
            zend_object_alloc(sizeof(php_ort_session_object), ce));
    object->session = nullptr;
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_session_handlers;
    object_properties_init(&object->std, ce);
    return &object->std;
}

static void onnx_session_free_object(zend_object *object) {
    php_ort_session_object *self = php_ort_session_from_obj(object);
    delete self->session;
    self->session = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_session_clone_object(zend_object *object) {
    zend_object *clone = onnx_session_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\Session cannot be cloned",
        0);
    return clone;
}

static zend_object *onnx_iobinding_create_object(zend_class_entry *ce) {
    php_ort_iobinding_object *object =
        reinterpret_cast<php_ort_iobinding_object *>(
            zend_object_alloc(sizeof(php_ort_iobinding_object), ce));
    object->binding = nullptr;
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_iobinding_handlers;
    object_properties_init(&object->std, ce);
    return &object->std;
}

static void onnx_iobinding_free_object(zend_object *object) {
    php_ort_iobinding_object *self = php_ort_iobinding_from_obj(object);
    delete self->binding;
    self->binding = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_iobinding_clone_object(zend_object *object) {
    zend_object *clone = onnx_iobinding_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\IoBinding cannot be cloned",
        0);
    return clone;
}

static zend_object *onnx_session_options_create_object(zend_class_entry *ce) {
    php_ort_session_options_object *object =
        reinterpret_cast<php_ort_session_options_object *>(
            zend_object_alloc(sizeof(php_ort_session_options_object), ce));
    std::exception_ptr allocation_error;
    object->config = nullptr;
    try {
        object->config = new php_ort::SessionConfig();
    } catch (...) {
        allocation_error = std::current_exception();
    }
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_session_options_handlers;
    object_properties_init(&object->std, ce);
    if (object->config == nullptr) {
        if (allocation_error != nullptr) {
            php_ort_throw_native(allocation_error,
                                 "failed to allocate SessionOptions object");
        } else {
            zend_throw_exception(
                onnx_conversion_exception_ce,
                "failed to allocate SessionOptions object", 0);
        }
    }
    return &object->std;
}

static void onnx_session_options_free_object(zend_object *object) {
    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(object);
    delete self->config;
    self->config = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_session_options_clone_object(zend_object *object) {
    zend_object *clone = onnx_session_options_create_object(object->ce);
    php_ort_session_options_object *source =
        php_ort_session_options_from_obj(object);
    php_ort_session_options_object *target =
        php_ort_session_options_from_obj(clone);
    if (source->config != nullptr && target->config != nullptr) {
        try {
            *target->config = *source->config;
        } catch (...) {
            php_ort_throw_native(std::current_exception(),
                                 "out of memory while cloning SessionOptions");
        }
    }
    if (EG(exception) == nullptr) {
        zend_objects_clone_members(clone, object);
    }
    return clone;
}

static zend_object *onnx_epdevice_create_object(zend_class_entry *ce) {
    php_ort_epdevice_object *object =
        reinterpret_cast<php_ort_epdevice_object *>(
            zend_object_alloc(sizeof(php_ort_epdevice_object), ce));
    std::exception_ptr allocation_error;
    object->info = nullptr;
    try {
        object->info = new php_ort::EpDeviceInfo();
    } catch (...) {
        allocation_error = std::current_exception();
    }
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_epdevice_handlers;
    object_properties_init(&object->std, ce);
    if (object->info == nullptr) {
        if (allocation_error != nullptr) {
            php_ort_throw_native(allocation_error,
                                 "failed to allocate EpDevice object");
        } else {
            zend_throw_exception(onnx_conversion_exception_ce,
                                 "failed to allocate EpDevice object", 0);
        }
    }
    return &object->std;
}

static void onnx_epdevice_free_object(zend_object *object) {
    php_ort_epdevice_object *self = php_ort_epdevice_from_obj(object);
    delete self->info;
    self->info = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_epdevice_clone_object(zend_object *object) {
    zend_object *clone = onnx_epdevice_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\EpDevice is immutable and cannot be cloned",
        0);
    return clone;
}

static zend_object *onnx_task_create_object(zend_class_entry *ce) {
    php_ort_task_object *object =
        reinterpret_cast<php_ort_task_object *>(
            zend_object_alloc(sizeof(php_ort_task_object), ce));
    std::exception_ptr allocation_error;
    object->task = nullptr;
    try {
        object->task = new std::shared_ptr<php_ort::AsyncTaskState>();
    } catch (...) {
        allocation_error = std::current_exception();
    }
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_task_handlers;
    object_properties_init(&object->std, ce);
    if (object->task == nullptr) {
        if (allocation_error != nullptr) {
            php_ort_throw_native(allocation_error,
                                 "failed to allocate Task object");
        } else {
            zend_throw_exception(onnx_conversion_exception_ce,
                                 "failed to allocate Task object", 0);
        }
    }
    return &object->std;
}

static void onnx_task_free_object(zend_object *object) {
    php_ort_task_object *self = php_ort_task_from_obj(object);
    delete self->task;
    self->task = nullptr;
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_task_clone_object(zend_object *object) {
    zend_object *clone = onnx_task_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error, "OnnxRuntime\\Task cannot be cloned", 0);
    return clone;
}

static zend_object *onnx_runtime_create_object(zend_class_entry *ce) {
    php_ort_runtime_object *object =
        reinterpret_cast<php_ort_runtime_object *>(
            zend_object_alloc(sizeof(php_ort_runtime_object), ce));
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &onnx_runtime_handlers;
    object_properties_init(&object->std, ce);
    return &object->std;
}

static void onnx_runtime_free_object(zend_object *object) {
    php_ort_runtime_object *self =
        reinterpret_cast<php_ort_runtime_object *>(
            reinterpret_cast<char *>(object) -
            XtOffsetOf(php_ort_runtime_object, std));
    zend_object_std_dtor(&self->std);
}

static zend_object *onnx_runtime_clone_object(zend_object *object) {
    zend_object *clone = onnx_runtime_create_object(object->ce);
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\Runtime cannot be cloned",
        0);
    return clone;
}

/* ----------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------*/

static zend_long php_ort_to_long(int64_t value, const char *what) {
    if (value > static_cast<int64_t>(ZEND_LONG_MAX) ||
        value < static_cast<int64_t>(ZEND_LONG_MIN)) {
        throw php_ort::ConversionError(
            std::string(what) + " exceeds zend_long range");
    }
    return static_cast<zend_long>(value);
}

static zend_long php_ort_size_to_long(size_t value, const char *what) {
    if (value > static_cast<size_t>(ZEND_LONG_MAX)) {
        throw php_ort::ConversionError(
            std::string(what) + " exceeds zend_long range");
    }
    return static_cast<zend_long>(value);
}

static uint32_t php_ort_checked_count(size_t count, const char *what) {
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw php_ort::ConversionError(
            std::string(what) + " exceeds the 32-bit array size limit");
    }
    return static_cast<uint32_t>(count);
}
static int php_ort_checked_thread_count(zend_long value, const char *what) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw php_ort::ConversionError(
            std::string(what) + " must be between 0 and " +
            std::to_string(std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

static std::string php_ort_normalized_name(zend_string *value) {
    std::string name(ZSTR_VAL(value), ZSTR_LEN(value));
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
}

static int php_ort_log_level_from_name(const std::string &name) {
    if (name == "verbose") {
        return ORT_LOGGING_LEVEL_VERBOSE;
    }
    if (name == "info") {
        return ORT_LOGGING_LEVEL_INFO;
    }
    if (name == "warning") {
        return ORT_LOGGING_LEVEL_WARNING;
    }
    if (name == "error") {
        return ORT_LOGGING_LEVEL_ERROR;
    }
    if (name == "fatal") {
        return ORT_LOGGING_LEVEL_FATAL;
    }
    throw php_ort::ConversionError("unknown log severity level: " + name);
}

static const char *php_ort_log_level_name(int level) {
    switch (level) {
        case ORT_LOGGING_LEVEL_VERBOSE:
            return "verbose";
        case ORT_LOGGING_LEVEL_INFO:
            return "info";
        case ORT_LOGGING_LEVEL_WARNING:
            return "warning";
        case ORT_LOGGING_LEVEL_ERROR:
            return "error";
        case ORT_LOGGING_LEVEL_FATAL:
            return "fatal";
        default:
            return "unknown";
    }
}

static const char *php_ort_device_type_name(OrtHardwareDeviceType type) {
    switch (type) {
        case OrtHardwareDeviceType_CPU:
            return "cpu";
        case OrtHardwareDeviceType_GPU:
            return "gpu";
        case OrtHardwareDeviceType_NPU:
            return "npu";
        default:
            return "unknown";
    }
}

static void php_ort_epdevice_to_array(const php_ort::EpDeviceInfo &info,
                                      zval *out) {
    array_init(out);
    add_assoc_stringl(out, "ep_name", info.ep_name.c_str(),
                      info.ep_name.size());
    add_assoc_stringl(out, "ep_vendor", info.ep_vendor.c_str(),
                      info.ep_vendor.size());
    add_assoc_stringl(out, "device_type",
                      php_ort_device_type_name(info.device_type),
                      std::strlen(php_ort_device_type_name(
                          info.device_type)));
    add_assoc_long(out, "vendor_id", info.vendor_id);
    add_assoc_long(out, "device_id", info.device_id);
    add_assoc_stringl(out, "device_vendor", info.device_vendor.c_str(),
                      info.device_vendor.size());
    add_assoc_bool(out, "is_plugin", info.isPluginDevice());
    if (info.isPluginDevice()) {
        add_assoc_stringl(out, "registration_name",
                          info.registration_name.c_str(),
                          info.registration_name.size());
    } else {
        add_assoc_null(out, "registration_name");
    }

    zval metadata;
    array_init(&metadata);
    for (const auto &entry : info.ep_metadata) {
        add_assoc_stringl(&metadata, entry.first.c_str(),
                          entry.second.c_str(), entry.second.size());
    }
    add_assoc_zval(out, "ep_metadata", &metadata);

    zval options;
    array_init(&options);
    for (const auto &entry : info.ep_options) {
        add_assoc_stringl(&options, entry.first.c_str(),
                          entry.second.c_str(), entry.second.size());
    }
    add_assoc_zval(out, "ep_options", &options);
}

static zend_object *php_ort_return_epdevice(php_ort::EpDeviceInfo info) {
    zend_object *object = onnx_epdevice_create_object(onnx_epdevice_ce);
    php_ort_epdevice_object *self = php_ort_epdevice_from_obj(object);
    if (self->info == nullptr) {
        zend_objects_store_del(object);
        return nullptr;
    }
    try {
        *self->info = std::move(info);
    } catch (...) {
        zend_objects_store_del(object);
        throw;
    }
    return object;
}

static void php_ort_session_names_to_array(
    zval *out, const std::vector<php_ort::TensorInfo> &infos) {
    array_init_size(
        out, php_ort_checked_count(infos.size(), "model input/output count"));
    for (const php_ort::TensorInfo &info : infos) {
        add_next_index_stringl(out, info.name.c_str(), info.name.size());
    }
}

static void php_ort_session_types_to_array(
    zval *out, const std::vector<php_ort::TensorInfo> &infos) {
    array_init_size(
        out, php_ort_checked_count(infos.size(), "model input/output count"));
    for (const php_ort::TensorInfo &info : infos) {
        const char *type_name = php_ort::elementTypeName(info.type);
        add_assoc_stringl_ex(out, info.name.c_str(), info.name.size(),
                             type_name, std::strlen(type_name));
    }
}

static void php_ort_session_shapes_to_array(
    zval *out, const std::vector<php_ort::TensorInfo> &infos) {
    array_init_size(
        out, php_ort_checked_count(infos.size(), "model input/output count"));
    for (const php_ort::TensorInfo &info : infos) {
        zval shape;
        array_init_size(
            &shape,
            php_ort_checked_count(info.shape.size(), "tensor rank"));
        try {
            for (int64_t dim : info.shape) {
                if (dim < 0) {
                    // ORT uses a negative value for a symbolic/unknown
                    // dimension.  Expose it as null; never coerce it to zero.
                    add_next_index_null(&shape);
                } else {
                    add_next_index_long(
                        &shape, php_ort_to_long(dim, "shape dimension"));
                }
            }
        } catch (...) {
            zval_ptr_dtor(&shape);
            throw;
        }
        add_assoc_zval_ex(out, info.name.c_str(), info.name.size(), &shape);
    }
}

template <typename T>
static T php_ort_read_raw(const php_ort::Tensor &tensor, size_t index) {
    if (index >= tensor.elementCount()) {
        throw php_ort::ConversionError("tensor element index out of range");
    }
    if (index >= tensor.data.size() / sizeof(T)) {
        throw php_ort::ConversionError("tensor payload index out of range");
    }
    T value;
    std::memcpy(&value, tensor.data.data() + index * sizeof(T), sizeof(T));
    return value;
}

static void php_ort_scalar_to_zval(const php_ort::Tensor &tensor,
                                   size_t index, zval *out) {
    switch (tensor.type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            ZVAL_DOUBLE(out,
                        static_cast<double>(php_ort_read_raw<float>(tensor, index)));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            ZVAL_DOUBLE(out, php_ort_read_raw<double>(tensor, index));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            ZVAL_LONG(out, php_ort_read_raw<int8_t>(tensor, index));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            ZVAL_LONG(out, php_ort_read_raw<uint8_t>(tensor, index));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            ZVAL_LONG(out, php_ort_read_raw<int16_t>(tensor, index));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            ZVAL_LONG(out, php_ort_read_raw<int32_t>(tensor, index));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            ZVAL_LONG(out, php_ort_to_long(
                                php_ort_read_raw<int64_t>(tensor, index),
                                "int64 tensor element"));
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            ZVAL_BOOL(out, php_ort_read_raw<uint8_t>(tensor, index) != 0);
            return;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
            if (index >= tensor.strings.size()) {
                throw php_ort::ConversionError(
                    "tensor string payload index out of range");
            }
            const std::string &value = tensor.strings[index];
            ZVAL_STRINGL(out, value.data(), value.size());
            return;
        }
        default:
            ZVAL_NULL(out);
            return;
    }
}

static void php_ort_tensor_to_zval(const php_ort::Tensor &tensor, size_t depth,
                                   size_t &index, zval *out) {
    if (depth == tensor.shape.size()) {
        php_ort_scalar_to_zval(tensor, index, out);
        ++index;
        return;
    }

    int64_t dim = tensor.shape[depth];
    if (dim < 0) {
        throw php_ort::ConversionError("negative tensor dimension");
    }
    // A Tensor may be representable with a very large dimension, but a PHP
    // array cannot.  Reject the impossible traversal before allocating or
    // looping; this is a toArray()/PHP-representation limit, not a Tensor
    // validity rule.
    uint64_t max_array_dimension = static_cast<uint64_t>(
        std::numeric_limits<uint32_t>::max());
    uint64_t zend_array_limit = static_cast<uint64_t>(ZEND_LONG_MAX);
    if (max_array_dimension > zend_array_limit) {
        max_array_dimension = zend_array_limit;
    }
    if (static_cast<uint64_t>(dim) > max_array_dimension) {
        throw php_ort::ConversionError(
            "tensor dimension exceeds PHP array limits");
    }
    array_init_size(out, static_cast<uint32_t>(dim));
    try {
        for (int64_t i = 0; i < dim; ++i) {
            zval child;
            ZVAL_UNDEF(&child);
            try {
                php_ort_tensor_to_zval(tensor, depth + 1, index, &child);
            } catch (...) {
                if (!Z_ISUNDEF(child)) {
                    zval_ptr_dtor(&child);
                }
                throw;
            }
            add_next_index_zval(out, &child);
        }
    } catch (...) {
        // This layer owns `out` until it is returned to its caller.  Recursive
        // failures must release all elements already inserted here; the
        // caller then sees an undefined child and does not double-free it.
        zval_ptr_dtor(out);
        ZVAL_UNDEF(out);
        throw;
    }
}

static zend_object *php_ort_return_tensor(php_ort::Tensor &&tensor) {
    zval holder;
    ZVAL_UNDEF(&holder);
    if (object_init_ex(&holder, onnx_tensor_ce) != SUCCESS) {
        if (Z_TYPE(holder) != IS_UNDEF) {
            zval_ptr_dtor(&holder);
        }
        return nullptr;
    }
    if (EG(exception) != nullptr) {
        zval_ptr_dtor(&holder);
        return nullptr;
    }

    php_ort_tensor_object *self = php_ort_tensor_from_obj(Z_OBJ(holder));
    if (self->tensor == nullptr) {
        zval_ptr_dtor(&holder);
        if (EG(exception) == nullptr) {
            zend_throw_exception(onnx_conversion_exception_ce,
                                 "failed to allocate tensor object", 0);
        }
        return nullptr;
    }

    try {
        *self->tensor = std::move(tensor);
    } catch (...) {
        zval_ptr_dtor(&holder);
        php_ort_throw_native(std::current_exception(),
                             "out of memory while returning tensor");
        return nullptr;
    }
    return Z_OBJ(holder);
}

static void php_ort_results_to_array(
    zval *out, const std::vector<php_ort::Tensor> &results,
    const std::vector<php_ort::TensorInfo> &infos) {
    if (results.size() != infos.size()) {
        throw php_ort::ConversionError(
            "output count does not match model outputs");
    }

    array_init_size(out, php_ort_checked_count(results.size(),
                                               "model output count"));
    for (size_t i = 0; i < results.size(); ++i) {
        zend_object *object =
            php_ort_return_tensor(php_ort::Tensor(results[i]));
        if (object == nullptr) {
            zval_ptr_dtor(out);
            ZVAL_UNDEF(out);
            throw php_ort::ConversionError(
                "failed to allocate output Tensor object");
        }
        zval value;
        ZVAL_OBJ(&value, object);
        add_assoc_zval(out, infos[i].name.c_str(), &value);
    }
}

static void php_ort_results_to_array_move(
    zval *out, std::vector<php_ort::Tensor> &results,
    const std::vector<php_ort::TensorInfo> &infos) {
    if (results.size() != infos.size()) {
        throw php_ort::ConversionError(
            "output count does not match model outputs");
    }

    array_init_size(out, php_ort_checked_count(results.size(),
                                               "model output count"));
    for (size_t i = 0; i < results.size(); ++i) {
        zend_object *object =
            php_ort_return_tensor(std::move(results[i]));
        if (object == nullptr) {
            zval_ptr_dtor(out);
            ZVAL_UNDEF(out);
            throw php_ort::ConversionError(
                "failed to allocate output Tensor object");
        }
        zval value;
        ZVAL_OBJ(&value, object);
        add_assoc_zval(out, infos[i].name.c_str(), &value);
    }
}

static zend_object *php_ort_return_ortvalue(Ort::Value &&value,
                                            zend_object *owner) {
    zval holder;
    ZVAL_UNDEF(&holder);
    if (object_init_ex(&holder, onnx_ortvalue_ce) != SUCCESS) {
        if (Z_TYPE(holder) != IS_UNDEF) {
            zval_ptr_dtor(&holder);
        }
        return nullptr;
    }
    if (EG(exception) != nullptr) {
        zval_ptr_dtor(&holder);
        return nullptr;
    }

    php_ort_ortvalue_object *self =
        php_ort_ortvalue_from_obj(Z_OBJ(holder));
    try {
        // Unlike new (std::nothrow), this catches both allocation failure and
        // any exception thrown by Ort::Value's move constructor.
        self->value = new Ort::Value(std::move(value));
    } catch (...) {
        zval_ptr_dtor(&holder);
        php_ort_throw_native(std::current_exception(),
                             "out of memory while returning OrtValue");
        return nullptr;
    }
    self->valid = true;
    if (owner != nullptr) {
        ZVAL_OBJ(&self->owner, owner);
        GC_ADDREF(owner);
    }
    return Z_OBJ(holder);
}

/**
 * Converts a C++ exception into a PHP exception.  Must be called from a
 * catch handler with std::current_exception(), and must never let the C++
 * exception unwind through Zend's C ABI.
 */
static void php_ort_throw_native(std::exception_ptr exception,
                                 const char *oom_context) {
    try {
        std::rethrow_exception(exception);
    } catch (const Ort::Exception &error) {
        zend_throw_exception(onnx_exception_ce, error.what(), 0);
    } catch (const php_ort::ConversionError &error) {
        zend_throw_exception(onnx_conversion_exception_ce, error.what(), 0);
    } catch (const std::bad_alloc &) {
        zend_throw_exception(onnx_conversion_exception_ce, oom_context, 0);
    } catch (const std::exception &error) {
        zend_throw_exception(onnx_conversion_exception_ce, error.what(), 0);
    } catch (...) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "unknown native exception", 0);
    }
}

/**
 * Reports a module-startup failure.  MINIT has no userland call frame to
 * receive an OnnxRuntime\Exception, so this logs and lets MINIT return
 * FAILURE; it never calls zend_throw_exception().
 */
static void php_ort_log_startup_error(std::exception_ptr exception) {
    const char *prefix =
        "onnxruntime: failed to initialize ONNX Runtime environment: ";
    try {
        std::rethrow_exception(exception);
    } catch (const Ort::Exception &error) {
        php_error_docref(NULL, E_WARNING, "%s%s", prefix, error.what());
    } catch (const php_ort::ConversionError &error) {
        php_error_docref(NULL, E_WARNING, "%s%s", prefix, error.what());
    } catch (const std::bad_alloc &) {
        php_error_docref(NULL, E_WARNING, "%sout of memory", prefix);
    } catch (const std::exception &error) {
        php_error_docref(NULL, E_WARNING, "%s%s", prefix, error.what());
    } catch (...) {
        php_error_docref(NULL, E_WARNING, "%sunknown native exception", prefix);
    }
}

/* ----------------------------------------------------------------------------
 * Methods
 * -------------------------------------------------------------------------*/

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_tensor_from_array, 0, 1,
                                       OnnxRuntime\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, value, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_tensor_dtype, 0, 0,
                                        IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_tensor_shape, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_tensor_rank, 0, 0,
                                        IS_LONG, 0)
ZEND_END_ARG_INFO()





ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_tensor_to_array, 0, 0,
                                        IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_tensor_to_ort_value, 0, 0,
                                       OnnxRuntime\\OrtValue, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_tensor_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_ortvalue_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_ortvalue_to_tensor, 0, 0,
                                       OnnxRuntime\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_ortvalue_debug_info, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_tensor_debug_info, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_session_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, modelPath, IS_STRING, 0)
    ZEND_ARG_OBJ_INFO(0, options, OnnxRuntime\\SessionOptions, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_epdevice_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_epdevice_string, 0, 0,
                                        IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_epdevice_long, 0, 0,
                                        IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_epdevice_bool, 0, 0,
                                        _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_epdevice_array, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_epdevice_nullable_string,
                                        0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_append_epdevice, 0, 1, IS_VOID, 0)
    ZEND_ARG_OBJ_INFO(0, device, OnnxRuntime\\EpDevice, 0)
    ZEND_ARG_TYPE_INFO(0, options, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_session_options_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_set_threads, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, threads, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_set_string, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, value, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_set_log_level, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, level, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_set_bool, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, enabled, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_append_ep, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, options, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_session_options_to_array, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_runtime_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_runtime_default, 0, 0,
                                       OnnxRuntime\\Runtime, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_runtime_version, 0, 0,
                                        IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_string_array, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_ep_devices, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_register, 0, 2, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_unregister, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_set_async_threads, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, threads, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_set_async_queue_capacity, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, capacity, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_runtime_async_info, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_input_names, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_output_names, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_input_types, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_output_types, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_input_shapes, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_output_shapes, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_run, 0, 1,
                                        IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, inputs, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_run_batch, 0, 1,
                                        IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, batch_inputs, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_session_run_async, 0, 1,
                                       OnnxRuntime\\Task, 0)
    ZEND_ARG_TYPE_INFO(0, inputs, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(
    arginfo_onnx_session_create_io_binding, 0, 0,
    OnnxRuntime\\IoBinding, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_session_run_with_binding,
                                        0, 1, IS_ARRAY, 0)
    ZEND_ARG_OBJ_INFO(0, binding, OnnxRuntime\\IoBinding, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_iobinding_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_iobinding_bind_input,
                                        0, 2, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, OnnxRuntime\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_iobinding_bind_output,
                                        0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    arginfo_onnx_iobinding_bind_output_buffer, 0, 2, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_OBJ_INFO(0, buffer, OnnxRuntime\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_iobinding_get_outputs,
                                        0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_onnx_task_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_task_get_state, 0, 0,
                                        IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_task_bool, 0, 0,
                                        _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_task_wait, 0, 0,
                                        _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, timeout, IS_DOUBLE, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_onnx_task_get_result, 0, 0,
                                        IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_onnx_task_get_error, 0, 0,
                                       Throwable, 1)
ZEND_END_ARG_INFO()

PHP_METHOD(OnnxRuntime_Tensor, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\Tensor cannot be constructed directly; use "
        "OnnxRuntime\\Tensor::fromArray()",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_Tensor, fromArray) {
    zval *value = nullptr;
    zend_string *dtype = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(value)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();

    php_ort::ConvertOptions options;
    if (dtype != nullptr) {
        zend_string *lower = zend_string_tolower(dtype);
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        bool known = php_ort::elementTypeFromName(
            std::string(ZSTR_VAL(lower), ZSTR_LEN(lower)), &type);
        zend_string_release(lower);
        if (!known) {
            zend_throw_exception_ex(
                zend_ce_value_error, 0,
                "Unsupported dtype \"%s\"; supported dtypes are: %s",
                ZSTR_VAL(dtype), php_ort::supportedDtypeNames());
            RETURN_THROWS();
        }
        options.dtype = type;
    }

    php_ort::Tensor tensor;
    try {
        php_ort::zvalToTensor(value, tensor, options);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while converting tensor data");
        RETURN_THROWS();
    }

    zend_object *object = php_ort_return_tensor(std::move(tensor));
    if (object == nullptr) {
        RETURN_THROWS();
    }
    RETURN_OBJ(object);
}

PHP_METHOD(OnnxRuntime_Tensor, dtype) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    RETURN_STRING(php_ort::elementTypeName(self->tensor->type));
}

PHP_METHOD(OnnxRuntime_Tensor, shape) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        array_init_size(
            return_value,
            php_ort_checked_count(self->tensor->shape.size(), "tensor rank"));
        for (int64_t dim : self->tensor->shape) {
            add_next_index_long(
                return_value,
                php_ort_to_long(dim, "shape dimension"));
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading tensor metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Tensor, rank) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    zend_long rank;
    try {
        rank = php_ort_size_to_long(self->tensor->rank(), "rank");
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading tensor metadata");
        RETURN_THROWS();
    }
    RETURN_LONG(rank);
}

PHP_METHOD(OnnxRuntime_Tensor, toArray) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        // Every read validates the whole payload first; this also covers tensors
        // created outside the PHP conversion path in a later phase.
        self->tensor->validate();
        size_t index = 0;
        php_ort_tensor_to_zval(*self->tensor, 0, index, return_value);
        if (index != self->tensor->elementCount()) {
            throw php_ort::ConversionError(
                "tensor traversal stopped before all elements were read");
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading tensor metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Tensor, __debugInfo) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    const php_ort::Tensor &tensor = *self->tensor;

    try {
        array_init_size(return_value, 3);
        add_assoc_string(return_value, "dtype",
                         php_ort::elementTypeName(tensor.type));
        add_assoc_long(
            return_value, "rank",
            php_ort_size_to_long(tensor.rank(), "rank"));

        zval shape;
        array_init_size(
            &shape,
            php_ort_checked_count(tensor.shape.size(), "tensor rank"));
        try {
            for (int64_t dim : tensor.shape) {
                add_next_index_long(
                    &shape, php_ort_to_long(dim, "shape dimension"));
            }
        } catch (...) {
            zval_ptr_dtor(&shape);
            throw;
        }
        add_assoc_zval(return_value, "shape", &shape);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading tensor metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Tensor, toOrtValue) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_tensor_object *self =
        php_ort_tensor_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        self->tensor->validate();
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        bool borrows_buffer = !self->tensor->isStringTensor();
        Ort::Value value = php_ort::tensorToOrtValue(
            *self->tensor, memory_info, allocator);
        zend_object *object = php_ort_return_ortvalue(
            std::move(value), borrows_buffer ? Z_OBJ_P(ZEND_THIS) : nullptr);
        if (object == nullptr) {
            RETURN_THROWS();
        }
        RETURN_OBJ(object);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while creating OrtValue");
        RETURN_THROWS();
    }
}

static const zend_function_entry onnx_tensor_methods[] = {
    PHP_ME(OnnxRuntime_Tensor, __construct, arginfo_onnx_tensor_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, fromArray, arginfo_onnx_tensor_from_array,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(OnnxRuntime_Tensor, dtype, arginfo_onnx_tensor_dtype,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, shape, arginfo_onnx_tensor_shape,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, rank, arginfo_onnx_tensor_rank, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, toArray, arginfo_onnx_tensor_to_array,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, toOrtValue, arginfo_onnx_tensor_to_ort_value,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Tensor, __debugInfo, arginfo_onnx_tensor_debug_info,
           ZEND_ACC_PUBLIC)
    PHP_FE_END
};

PHP_METHOD(OnnxRuntime_OrtValue, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\OrtValue cannot be constructed directly; use "
        "OnnxRuntime\\Tensor::toOrtValue()",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_OrtValue, toTensor) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_ortvalue_object *self =
        php_ort_ortvalue_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (!self->valid || self->value == nullptr) {
            throw php_ort::ConversionError("empty OrtValue");
        }
        php_ort::Tensor tensor = php_ort::ortValueToTensor(*self->value);
        zend_object *object = php_ort_return_tensor(std::move(tensor));
        if (object == nullptr) {
            RETURN_THROWS();
        }
        RETURN_OBJ(object);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while converting OrtValue");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_OrtValue, __debugInfo) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_ortvalue_object *self =
        php_ort_ortvalue_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        array_init(return_value);
        if (!self->valid || self->value == nullptr) {
            return;
        }
        Ort::TensorTypeAndShapeInfo info = self->value->GetTensorTypeAndShapeInfo();
        add_assoc_string(return_value, "dtype",
                         php_ort::elementTypeName(info.GetElementType()));
        add_assoc_long(return_value, "rank",
                       php_ort_size_to_long(info.GetDimensionsCount(), "rank"));
        zval shape;
        std::vector<int64_t> dims = info.GetShape();
        array_init_size(
            &shape, php_ort_checked_count(dims.size(), "tensor rank"));
        try {
            for (int64_t dim : dims) {
                add_next_index_long(
                    &shape, php_ort_to_long(dim, "shape dimension"));
            }
        } catch (...) {
            zval_ptr_dtor(&shape);
            throw;
        }
        add_assoc_zval(return_value, "shape", &shape);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading OrtValue metadata");
        RETURN_THROWS();
    }
}

static const zend_function_entry onnx_ortvalue_methods[] = {
    PHP_ME(OnnxRuntime_OrtValue, __construct, arginfo_onnx_ortvalue_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_OrtValue, toTensor, arginfo_onnx_ortvalue_to_tensor,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_OrtValue, __debugInfo, arginfo_onnx_ortvalue_debug_info,
           ZEND_ACC_PUBLIC)
    PHP_FE_END
};

PHP_METHOD(OnnxRuntime_SessionOptions, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
}

PHP_METHOD(OnnxRuntime_SessionOptions, setIntraOpNumThreads) {
    zend_long threads = 0;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(threads)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        self->config->intra_op_num_threads =
            php_ort_checked_thread_count(threads, "intra-op thread count");
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setInterOpNumThreads) {
    zend_long threads = 0;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(threads)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        self->config->inter_op_num_threads =
            php_ort_checked_thread_count(threads, "inter-op thread count");
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setGraphOptimizationLevel) {
    zend_string *value = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(value)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        std::string normalized = php_ort_normalized_name(value);
        if (normalized != "disabled" && normalized != "basic" &&
            normalized != "extended" && normalized != "all") {
            throw php_ort::ConversionError(
                "unknown graph optimization level: " + normalized +
                "; expected disabled, basic, extended, or all");
        }
        self->config->graph_optimization_level = normalized;
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setExecutionMode) {
    zend_string *value = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(value)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        std::string normalized = php_ort_normalized_name(value);
        if (normalized != "sequential" && normalized != "parallel") {
            throw php_ort::ConversionError(
                "unknown execution mode: " + normalized +
                "; expected sequential or parallel");
        }
        self->config->execution_mode = normalized;
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setLogSeverityLevel) {
    zval *value = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        ZVAL_DEREF(value);
        int level = 0;
        if (Z_TYPE_P(value) == IS_LONG) {
            zend_long raw = Z_LVAL_P(value);
            if (raw < ORT_LOGGING_LEVEL_VERBOSE ||
                raw > ORT_LOGGING_LEVEL_FATAL) {
                throw php_ort::ConversionError(
                    "log severity level must be between 0 and 4, or a name "
                    "from verbose/info/warning/error/fatal");
            }
            level = static_cast<int>(raw);
        } else if (Z_TYPE_P(value) == IS_STRING) {
            level = php_ort_log_level_from_name(
                php_ort_normalized_name(Z_STR_P(value)));
        } else {
            zend_type_error(
                "OnnxRuntime\\SessionOptions::setLogSeverityLevel expects "
                "int or string, %s given", zend_zval_type_name(value));
            RETURN_THROWS();
        }
        self->config->log_severity_level = level;
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setEnableCpuMemArena) {
    bool enabled = false;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_BOOL(enabled)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        self->config->enable_cpu_mem_arena = enabled;
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, setEnableMemPattern) {
    bool enabled = false;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_BOOL(enabled)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        self->config->enable_mem_pattern = enabled;
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while configuring SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, appendExecutionProvider) {
    zend_string *name = nullptr;
    HashTable *options = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STR(name)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(options)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        std::string ep_name(ZSTR_VAL(name), ZSTR_LEN(name));
        if (ep_name.empty()) {
            throw php_ort::ConversionError(
                "execution provider name must not be empty");
        }
        if (ep_name.find('\0') != std::string::npos) {
            throw php_ort::ConversionError(
                "execution provider name must not contain NUL");
        }

        php_ort::ExecutionProviderConfig provider;
        provider.name = ep_name;
        if (options != nullptr) {
            zend_string *key = nullptr;
            zval *value = nullptr;
            ZEND_HASH_FOREACH_STR_KEY_VAL(options, key, value) {
                if (key == nullptr) {
                    throw php_ort::ConversionError(
                        "execution provider option keys must be strings");
                }
                ZVAL_DEREF(value);
                if (Z_TYPE_P(value) != IS_STRING) {
                    throw php_ort::ConversionError(
                        "execution provider option values must be strings");
                }
                std::string option_key(ZSTR_VAL(key), ZSTR_LEN(key));
                std::string option_value(Z_STRVAL_P(value), Z_STRLEN_P(value));
                if (option_key.find('\0') != std::string::npos ||
                    option_value.find('\0') != std::string::npos) {
                    throw php_ort::ConversionError(
                        "execution provider options must not contain NUL");
                }
                provider.options[option_key] = option_value;
            } ZEND_HASH_FOREACH_END();
        }
        self->config->execution_providers.push_back(std::move(provider));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to append execution provider");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, appendEpDevice) {
    zval *device = nullptr;
    HashTable *options = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(device)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(options)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        ZVAL_DEREF(device);
        if (Z_TYPE_P(device) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(device), onnx_epdevice_ce)) {
            throw php_ort::ConversionError(
                "appendEpDevice expects an OnnxRuntime\\EpDevice object");
        }
        php_ort_epdevice_object *device_object =
            php_ort_epdevice_from_obj(Z_OBJ_P(device));
        if (device_object->info == nullptr) {
            zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
        }

        // Fail early if the Env-owned device disappeared (for example the
        // plugin was unregistered).  The final authoritative check happens
        // again when the Session is created.
        auto env = php_ort::environment();
        php_ort::resolveEpDevice(*env, *device_object->info);
        php_ort::validatePluginDevice(*device_object->info);

        php_ort::ExecutionProviderConfig provider;
        provider.kind = php_ort::ExecutionProviderKind::EpDevice;
        provider.name = device_object->info->ep_name;
        provider.device = *device_object->info;
        if (options != nullptr) {
            zend_string *key = nullptr;
            zval *value = nullptr;
            ZEND_HASH_FOREACH_STR_KEY_VAL(options, key, value) {
                if (key == nullptr) {
                    throw php_ort::ConversionError(
                        "execution provider option keys must be strings");
                }
                ZVAL_DEREF(value);
                if (Z_TYPE_P(value) != IS_STRING) {
                    throw php_ort::ConversionError(
                        "execution provider option values must be strings");
                }
                std::string option_key(ZSTR_VAL(key), ZSTR_LEN(key));
                std::string option_value(Z_STRVAL_P(value), Z_STRLEN_P(value));
                if (option_key.find('\0') != std::string::npos ||
                    option_value.find('\0') != std::string::npos) {
                    throw php_ort::ConversionError(
                        "execution provider options must not contain NUL");
                }
                provider.options[option_key] = option_value;
            } ZEND_HASH_FOREACH_END();
        }
        self->config->execution_providers.push_back(std::move(provider));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to append EP device");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_SessionOptions, toArray) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_options_object *self =
        php_ort_session_options_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->config == nullptr) {
            throw php_ort::ConversionError("SessionOptions is uninitialized");
        }
        array_init(return_value);
        add_assoc_long(return_value, "intraOpNumThreads",
                       self->config->intra_op_num_threads);
        add_assoc_long(return_value, "interOpNumThreads",
                       self->config->inter_op_num_threads);
        add_assoc_stringl(return_value, "graphOptimizationLevel",
                          self->config->graph_optimization_level.c_str(),
                          self->config->graph_optimization_level.size());
        add_assoc_stringl(return_value, "executionMode",
                          self->config->execution_mode.c_str(),
                          self->config->execution_mode.size());
        add_assoc_long(return_value, "logSeverityLevel",
                       self->config->log_severity_level);
        add_assoc_stringl(return_value, "logSeverityName",
                          php_ort_log_level_name(
                              self->config->log_severity_level),
                          std::strlen(php_ort_log_level_name(
                              self->config->log_severity_level)));
        add_assoc_bool(return_value, "enableCpuMemArena",
                       self->config->enable_cpu_mem_arena);
        add_assoc_bool(return_value, "enableMemPattern",
                       self->config->enable_mem_pattern);

        zval providers;
        array_init(&providers);
        for (const php_ort::ExecutionProviderConfig &provider :
             self->config->execution_providers) {
            if (provider.kind !=
                php_ort::ExecutionProviderKind::Builtin) {
                continue;
            }
            zval entry;
            array_init(&entry);
            add_assoc_stringl(&entry, "name", provider.name.c_str(),
                              provider.name.size());
            zval options;
            array_init(&options);
            for (const auto &option : provider.options) {
                add_assoc_stringl(&options, option.first.c_str(),
                                  option.second.c_str(),
                                  option.second.size());
            }
            add_assoc_zval(&entry, "options", &options);
            add_next_index_zval(&providers, &entry);
        }
        add_assoc_zval(return_value, "executionProviders", &providers);

        zval ep_devices;
        array_init(&ep_devices);
        for (const php_ort::ExecutionProviderConfig &provider :
             self->config->execution_providers) {
            if (provider.kind !=
                php_ort::ExecutionProviderKind::EpDevice) {
                continue;
            }
            zval entry;
            array_init(&entry);
            zval device;
            php_ort_epdevice_to_array(provider.device, &device);
            add_assoc_zval(&entry, "device", &device);
            zval options;
            array_init(&options);
            for (const auto &option : provider.options) {
                add_assoc_stringl(&options, option.first.c_str(),
                                  option.second.c_str(),
                                  option.second.size());
            }
            add_assoc_zval(&entry, "options", &options);
            add_next_index_zval(&ep_devices, &entry);
        }
        add_assoc_zval(return_value, "epDevices", &ep_devices);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading SessionOptions");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\Runtime cannot be constructed directly; use "
        "OnnxRuntime\\Runtime::default()",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_Runtime, default) {
    ZEND_PARSE_PARAMETERS_NONE();
    object_init_ex(return_value, onnx_runtime_ce);
}

PHP_METHOD(OnnxRuntime_Runtime, getVersion) {
    ZEND_PARSE_PARAMETERS_NONE();
    try {
        std::string version = Ort::GetVersionString();
        RETURN_STRINGL(version.c_str(), version.size());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to query ONNX Runtime version");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, getBuildInfo) {
    ZEND_PARSE_PARAMETERS_NONE();
    try {
        std::string info = Ort::GetBuildInfoString();
        RETURN_STRINGL(info.c_str(), info.size());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to query ONNX Runtime build info");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, getAvailableProviders) {
    ZEND_PARSE_PARAMETERS_NONE();
    try {
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        array_init_size(
            return_value,
            php_ort_checked_count(providers.size(), "provider count"));
        for (const std::string &provider : providers) {
            add_next_index_stringl(return_value, provider.c_str(),
                                   provider.size());
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to query ONNX Runtime providers");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, getEpDevices) {
    ZEND_PARSE_PARAMETERS_NONE();
    try {
        auto devices = php_ort::runtimeEpDevices();
        array_init_size(
            return_value,
            php_ort_checked_count(devices.size(), "EP device count"));
        for (const auto &device : devices) {
            zend_object *object = php_ort_return_epdevice(device);
            if (object == nullptr) {
                RETURN_THROWS();
            }
            zval value;
            ZVAL_OBJ(&value, object);
            add_next_index_zval(return_value, &value);
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to query ONNX Runtime EP devices");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, registerExecutionProviderLibrary) {
    zend_string *name = nullptr;
    zend_string *path = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(name)
        Z_PARAM_STR(path)
    ZEND_PARSE_PARAMETERS_END();

    try {
        php_ort::registerExecutionProviderLibrary(
            std::string(ZSTR_VAL(name), ZSTR_LEN(name)),
            std::string(ZSTR_VAL(path), ZSTR_LEN(path)));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to register execution provider library");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, unregisterExecutionProviderLibrary) {
    zend_string *name = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    try {
        php_ort::unregisterExecutionProviderLibrary(
            std::string(ZSTR_VAL(name), ZSTR_LEN(name)));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to unregister execution provider library");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, setAsyncThreads) {
    zend_long threads = 0;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(threads)
    ZEND_PARSE_PARAMETERS_END();

    try {
        php_ort::configureAsyncEngine(
            static_cast<size_t>(threads),
            php_ort::asyncEngineQueueCapacity());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to configure async worker count");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, setAsyncQueueCapacity) {
    zend_long capacity = 0;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(capacity)
    ZEND_PARSE_PARAMETERS_END();

    try {
        php_ort::configureAsyncEngine(
            php_ort::asyncEngineWorkerCount(),
            static_cast<size_t>(capacity));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to configure async queue capacity");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Runtime, asyncInfo) {
    ZEND_PARSE_PARAMETERS_NONE();
    try {
        array_init_size(return_value, 5);
        add_assoc_long(return_value, "workers",
                       static_cast<zend_long>(
                           php_ort::asyncEngineWorkerCount()));
        add_assoc_long(return_value, "queueCapacity",
                       static_cast<zend_long>(
                           php_ort::asyncEngineQueueCapacity()));
        add_assoc_long(return_value, "queued",
                       static_cast<zend_long>(
                           php_ort::asyncEngineQueuedCount()));
        add_assoc_long(return_value, "running",
                       static_cast<zend_long>(
                           php_ort::asyncEngineRunningCount()));
        add_assoc_bool(return_value, "started",
                       php_ort::asyncEngineRunning());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to query async engine information");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, __construct) {
    zend_string *model_path = nullptr;
    zval *options = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STR(model_path)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(options)
    ZEND_PARSE_PARAMETERS_END();

    try {
        const php_ort::SessionConfig *config = nullptr;
        if (options != nullptr) {
            ZVAL_DEREF(options);
            if (Z_TYPE_P(options) != IS_NULL) {
                if (Z_TYPE_P(options) != IS_OBJECT ||
                    !instanceof_function(Z_OBJCE_P(options),
                                         onnx_session_options_ce)) {
                    throw php_ort::ConversionError(
                        "Session options must be an "
                        "OnnxRuntime\\SessionOptions object or null");
                }
                php_ort_session_options_object *options_object =
                    php_ort_session_options_from_obj(Z_OBJ_P(options));
                if (options_object->config == nullptr) {
                    throw php_ort::ConversionError(
                        "SessionOptions has no configuration storage");
                }
                config = options_object->config;
            }
        }

        php_ort_session_object *self =
            php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
        std::string path(ZSTR_VAL(model_path), ZSTR_LEN(model_path));
        if (path.find('\0') != std::string::npos) {
            throw php_ort::ConversionError("model path contains NUL");
        }
        std::unique_ptr<php_ort::Session> next(
            new php_ort::Session(path, config));
        delete self->session;
        self->session = next.release();
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while loading ONNX model");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, inputNames) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_names_to_array(return_value, self->session->inputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, outputNames) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_names_to_array(return_value, self->session->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, inputTypes) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_types_to_array(return_value, self->session->inputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, outputTypes) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_types_to_array(return_value, self->session->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, inputShapes) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_shapes_to_array(return_value, self->session->inputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, outputShapes) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        php_ort_session_shapes_to_array(return_value, self->session->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading model metadata");
        RETURN_THROWS();
    }
}

static std::vector<const php_ort::Tensor *> php_ort_session_input_tensors(
    const php_ort::Session &session, zend_array *input_array) {
    const std::vector<php_ort::TensorInfo> &infos = session.inputs();

    // Reject unknown keys before the missing-input pass so an input map
    // like ["Bogus" => Tensor] reports the unknown name, not "missing X".
    zend_string *key = nullptr;
    zval *unused = nullptr;
    ZEND_HASH_FOREACH_STR_KEY_VAL(input_array, key, unused) {
        if (key == nullptr) {
            throw php_ort::ConversionError(
                "unknown numeric input key; model inputs are named");
        }
        bool known = false;
        for (const php_ort::TensorInfo &info : infos) {
            if (info.name == ZSTR_VAL(key) &&
                info.name.size() == ZSTR_LEN(key)) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw php_ort::ConversionError(
                "unknown input \"" +
                std::string(ZSTR_VAL(key), ZSTR_LEN(key)) + "\"");
        }
    }
    ZEND_HASH_FOREACH_END();

    std::vector<const php_ort::Tensor *> tensors;
    tensors.reserve(infos.size());
    for (const php_ort::TensorInfo &info : infos) {
        zval *entry = zend_hash_str_find(input_array, info.name.data(),
                                         info.name.size());
        if (entry == nullptr) {
            throw php_ort::ConversionError(
                "missing input \"" + info.name + "\"");
        }
        ZVAL_DEREF(entry);
        if (Z_TYPE_P(entry) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(entry), onnx_tensor_ce)) {
            throw php_ort::ConversionError(
                "input \"" + info.name +
                "\" must be an OnnxRuntime\\Tensor");
        }
        php_ort_tensor_object *tensor_object =
            php_ort_tensor_from_obj(Z_OBJ_P(entry));
        if (tensor_object->tensor == nullptr) {
            // This means object allocation failed, not that the tensor
            // has zero elements.  Zero-element tensors are valid and are
            // left for Tensor::validate()/ORT to judge.
            throw php_ort::ConversionError(
                "input \"" + info.name +
                "\" has no tensor storage (uninitialized Tensor)");
        }
        tensors.push_back(tensor_object->tensor);
    }
    return tensors;
}

PHP_METHOD(OnnxRuntime_Session, run) {
    zval *inputs = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(inputs)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }

        zend_array *input_array = Z_ARRVAL_P(inputs);
        std::vector<const php_ort::Tensor *> tensors =
            php_ort_session_input_tensors(*self->session, input_array);

        // The input PHP zvals and their Tensors stay alive for the entire
        // call; the core keeps only raw Tensor pointers during Run.
        std::vector<php_ort::Tensor> results =
            self->session->run(tensors);

        php_ort_results_to_array_move(return_value, results,
                                      self->session->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while running ONNX model");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, runBatch) {
    zval *batch_inputs = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(batch_inputs)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    zval result;
    ZVAL_UNDEF(&result);
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }

        zend_array *batch = Z_ARRVAL_P(batch_inputs);
        array_init(&result);

        zend_long index = 0;
        zend_ulong num_key = 0;
        zend_string *key = nullptr;
        zval *entry = nullptr;
        ZEND_HASH_FOREACH_KEY_VAL(batch, num_key, key, entry) {
            if (key != nullptr ||
                num_key != static_cast<zend_ulong>(index)) {
                throw php_ort::ConversionError(
                    "batch input must be a list with sequential integer keys "
                    "starting at 0");
            }
            ZVAL_DEREF(entry);
            if (Z_TYPE_P(entry) != IS_ARRAY) {
                throw php_ort::ConversionError(
                    "batch entry " + std::to_string(index) +
                    " must be an array of model inputs");
            }

            try {
                std::vector<const php_ort::Tensor *> tensors =
                    php_ort_session_input_tensors(
                        *self->session, Z_ARRVAL_P(entry));
                std::vector<php_ort::Tensor> results =
                    self->session->run(tensors);

                zval entry_result;
                ZVAL_UNDEF(&entry_result);
                php_ort_results_to_array_move(
                    &entry_result, results, self->session->outputs());
                add_next_index_zval(&result, &entry_result);
            } catch (const php_ort::ConversionError &e) {
                throw php_ort::ConversionError(
                    "batch entry " + std::to_string(index) + ": " + e.what());
            }
            ++index;
        }
        ZEND_HASH_FOREACH_END();

        RETVAL_ZVAL(&result, 0, 0);
    } catch (...) {
        if (!Z_ISUNDEF(result)) {
            zval_ptr_dtor(&result);
            ZVAL_UNDEF(&result);
        }
        php_ort_throw_native(std::current_exception(),
                             "out of memory while running ONNX model batch");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, runAsync) {
    zval *inputs = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(inputs)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }

        zend_array *input_array = Z_ARRVAL_P(inputs);
        std::vector<const php_ort::Tensor *> tensors =
            php_ort_session_input_tensors(*self->session, input_array);

        // Validate on the PHP thread and deep-copy every input.  The worker
        // must never touch the PHP Tensor objects or their request-owned
        // payload.
        std::vector<php_ort::Tensor> owned_inputs =
            self->session->copyInputsForAsync(tensors);

        std::shared_ptr<php_ort::SessionState> state =
            self->session->state();
        std::shared_ptr<php_ort::AsyncTaskState> task =
            php_ort::submitAsyncTask(std::move(state),
                                      std::move(owned_inputs));

        object_init_ex(return_value, onnx_task_ce);
        php_ort_task_object *task_object =
            php_ort_task_from_obj(Z_OBJ_P(return_value));
        if (task_object->task == nullptr) {
            throw php_ort::ConversionError(
                "failed to allocate async Task object");
        }
        *task_object->task = std::move(task);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to submit async ONNX inference");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, createIoBinding) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }

        std::unique_ptr<php_ort::IoBindingState> binding =
            self->session->createIoBinding();
        object_init_ex(return_value, onnx_iobinding_ce);
        php_ort_iobinding_object *binding_object =
            php_ort_iobinding_from_obj(Z_OBJ_P(return_value));
        binding_object->binding = binding.release();
        if (binding_object->binding == nullptr) {
            throw php_ort::ConversionError(
                "failed to allocate I/O binding object");
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to create I/O binding");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Session, runWithBinding) {
    zval *binding = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(binding)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_session_object *self =
        php_ort_session_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->session == nullptr) {
            throw php_ort::ConversionError("session is not initialized");
        }
        if (Z_TYPE_P(binding) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(binding), onnx_iobinding_ce)) {
            throw php_ort::ConversionError(
                "runWithBinding expects an OnnxRuntime\\IoBinding");
        }
        php_ort_iobinding_object *binding_object =
            php_ort_iobinding_from_obj(Z_OBJ_P(binding));
        if (binding_object->binding == nullptr) {
            throw php_ort::ConversionError(
                "I/O binding is not initialized");
        }
        if (binding_object->binding->session().get() !=
            self->session->state().get()) {
            throw php_ort::ConversionError(
                "I/O binding belongs to a different Session");
        }

        std::vector<php_ort::Tensor> results =
            binding_object->binding->run();
        php_ort_results_to_array(return_value, results,
                                 self->session->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to run with I/O binding");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_IoBinding, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\IoBinding is created by Session::createIoBinding "
        "and cannot be constructed",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_IoBinding, bindInput) {
    zend_string *name = nullptr;
    zval *tensor = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(name)
        Z_PARAM_OBJECT(tensor)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_iobinding_object *self =
        php_ort_iobinding_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->binding == nullptr) {
            throw php_ort::ConversionError(
                "I/O binding is not initialized");
        }
        if (Z_TYPE_P(tensor) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(tensor), onnx_tensor_ce)) {
            throw php_ort::ConversionError(
                "bindInput expects an OnnxRuntime\\Tensor");
        }
        php_ort_tensor_object *tensor_object =
            php_ort_tensor_from_obj(Z_OBJ_P(tensor));
        if (tensor_object->tensor == nullptr) {
            throw php_ort::ConversionError(
                "input Tensor is uninitialized");
        }

        self->binding->bindInput(
            std::string(ZSTR_VAL(name), ZSTR_LEN(name)),
            *tensor_object->tensor);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to bind input");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_IoBinding, bindOutput) {
    zend_string *name = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_iobinding_object *self =
        php_ort_iobinding_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->binding == nullptr) {
            throw php_ort::ConversionError(
                "I/O binding is not initialized");
        }
        self->binding->bindOutput(
            std::string(ZSTR_VAL(name), ZSTR_LEN(name)));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to bind output");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_IoBinding, bindOutputBuffer) {
    zend_string *name = nullptr;
    zval *buffer = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(name)
        Z_PARAM_OBJECT(buffer)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_iobinding_object *self =
        php_ort_iobinding_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->binding == nullptr) {
            throw php_ort::ConversionError(
                "I/O binding is not initialized");
        }
        if (Z_TYPE_P(buffer) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(buffer), onnx_tensor_ce)) {
            throw php_ort::ConversionError(
                "bindOutputBuffer expects an OnnxRuntime\\Tensor");
        }
        php_ort_tensor_object *buffer_object =
            php_ort_tensor_from_obj(Z_OBJ_P(buffer));
        if (buffer_object->tensor == nullptr) {
            throw php_ort::ConversionError(
                "output buffer Tensor is uninitialized");
        }

        self->binding->bindOutputBuffer(
            std::string(ZSTR_VAL(name), ZSTR_LEN(name)),
            *buffer_object->tensor);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to bind output buffer");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_IoBinding, getOutputs) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_iobinding_object *self =
        php_ort_iobinding_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->binding == nullptr) {
            throw php_ort::ConversionError(
                "I/O binding is not initialized");
        }
        php_ort_results_to_array(
            return_value, self->binding->outputs(),
            self->binding->session()->outputs());
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to read I/O binding outputs");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Task, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\Task is created by Session::runAsync and cannot be constructed",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_Task, getState) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->task == nullptr || *self->task == nullptr) {
        zend_throw_exception(onnx_exception_ce,
                             "Task is not initialized", 0);
        RETURN_THROWS();
    }
    const char *status =
        php_ort::asyncTaskStatusName((*self->task)->status());
    RETURN_STRING(status);
}

PHP_METHOD(OnnxRuntime_Task, isReady) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->task == nullptr || *self->task == nullptr) {
        zend_throw_exception(onnx_exception_ce,
                             "Task is not initialized", 0);
        RETURN_THROWS();
    }
    RETURN_BOOL((*self->task)->isReady());
}

PHP_METHOD(OnnxRuntime_Task, wait) {
    zval *timeout = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(timeout)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->task == nullptr || *self->task == nullptr) {
            throw php_ort::ConversionError("Task is not initialized");
        }

        int64_t timeout_ms = -1;
        if (timeout != nullptr) {
            ZVAL_DEREF(timeout);
            if (Z_TYPE_P(timeout) == IS_NULL) {
                timeout_ms = -1;
            } else if (Z_TYPE_P(timeout) == IS_LONG) {
                zend_long value = Z_LVAL_P(timeout);
                if (value < 0) {
                    throw php_ort::ConversionError(
                        "wait timeout must not be negative");
                }
                timeout_ms = static_cast<int64_t>(value) * 1000;
            } else if (Z_TYPE_P(timeout) == IS_DOUBLE) {
                double value = Z_DVAL_P(timeout);
                if (!std::isfinite(value) || value < 0) {
                    throw php_ort::ConversionError(
                        "wait timeout must be a finite non-negative number");
                }
                const double max_ms =
                    static_cast<double>(std::numeric_limits<int64_t>::max());
                if (value > max_ms / 1000.0) {
                    timeout_ms = std::numeric_limits<int64_t>::max();
                } else {
                    timeout_ms = static_cast<int64_t>(value * 1000.0);
                }
            } else {
                throw php_ort::ConversionError(
                    "wait timeout must be a number or null");
            }
        }

        RETURN_BOOL((*self->task)->wait(timeout_ms));
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to wait for async task");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Task, getResult) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->task == nullptr || *self->task == nullptr) {
            throw php_ort::ConversionError("Task is not initialized");
        }
        std::shared_ptr<php_ort::AsyncTaskState> task = *self->task;
        php_ort::AsyncTaskStatus status = task->status();
        if (status == php_ort::AsyncTaskStatus::Failed) {
            std::exception_ptr error = task->error();
            if (error != nullptr) {
                std::rethrow_exception(error);
            }
            throw php_ort::ConversionError("async task failed");
        }
        if (status != php_ort::AsyncTaskStatus::Completed) {
            throw php_ort::ConversionError(
                std::string("async task is ") +
                php_ort::asyncTaskStatusName(status) +
                "; call wait() before getResult()");
        }

        std::vector<php_ort::Tensor> results = task->resultCopy();
        const std::vector<std::string> &output_names = task->outputNames();
        if (results.size() != output_names.size()) {
            throw php_ort::ConversionError(
                "async output count does not match model outputs");
        }
        array_init_size(
            return_value,
            php_ort_checked_count(results.size(), "async output count"));
        for (size_t i = 0; i < results.size(); ++i) {
            zend_object *object =
                php_ort_return_tensor(std::move(results[i]));
            if (object == nullptr) {
                RETURN_THROWS();
            }
            zval value;
            ZVAL_OBJ(&value, object);
            add_assoc_zval(return_value, output_names[i].c_str(), &value);
        }
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "failed to read async task result");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_Task, getError) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->task == nullptr || *self->task == nullptr ||
        (*self->task)->status() != php_ort::AsyncTaskStatus::Failed) {
        RETURN_NULL();
    }

    std::exception_ptr error = (*self->task)->error();
    if (error == nullptr) {
        RETURN_NULL();
    }

    std::string message = "async task failed";
    bool conversion = false;
    try {
        std::rethrow_exception(error);
    } catch (const php_ort::ConversionError &e) {
        message = e.what();
        conversion = true;
    } catch (const std::exception &e) {
        message = e.what();
    } catch (...) {
        message = "unknown native async exception";
    }

    zend_class_entry *exception_ce =
        conversion ? onnx_conversion_exception_ce : onnx_exception_ce;
    object_init_ex(return_value, exception_ce);
    zend_update_property_stringl(
        exception_ce, Z_OBJ_P(return_value), "message",
        sizeof("message") - 1, message.data(), message.size());
}

PHP_METHOD(OnnxRuntime_Task, cancel) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_task_object *self =
        php_ort_task_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->task == nullptr || *self->task == nullptr) {
        zend_throw_exception(onnx_exception_ce,
                             "Task is not initialized", 0);
        RETURN_THROWS();
    }
    RETURN_BOOL((*self->task)->cancelQueued());
}

PHP_METHOD(OnnxRuntime_EpDevice, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(
        zend_ce_error,
        "OnnxRuntime\\EpDevice cannot be constructed directly; use "
        "OnnxRuntime\\Runtime::getEpDevices()",
        0);
    RETURN_THROWS();
}

PHP_METHOD(OnnxRuntime_EpDevice, epName) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_STRINGL(self->info->ep_name.c_str(), self->info->ep_name.size());
}

PHP_METHOD(OnnxRuntime_EpDevice, epVendor) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_STRINGL(self->info->ep_vendor.c_str(),
                   self->info->ep_vendor.size());
}

PHP_METHOD(OnnxRuntime_EpDevice, deviceType) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    const char *name = php_ort_device_type_name(self->info->device_type);
    RETURN_STRINGL(name, std::strlen(name));
}

PHP_METHOD(OnnxRuntime_EpDevice, vendorId) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_LONG(static_cast<zend_long>(self->info->vendor_id));
}

PHP_METHOD(OnnxRuntime_EpDevice, deviceId) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_LONG(static_cast<zend_long>(self->info->device_id));
}

PHP_METHOD(OnnxRuntime_EpDevice, deviceVendor) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_STRINGL(self->info->device_vendor.c_str(),
                   self->info->device_vendor.size());
}

PHP_METHOD(OnnxRuntime_EpDevice, metadata) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    array_init(return_value);
    for (const auto &entry : self->info->ep_metadata) {
        add_assoc_stringl(return_value, entry.first.c_str(),
                          entry.second.c_str(), entry.second.size());
    }
}

PHP_METHOD(OnnxRuntime_EpDevice, options) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    array_init(return_value);
    for (const auto &entry : self->info->ep_options) {
        add_assoc_stringl(return_value, entry.first.c_str(),
                          entry.second.c_str(), entry.second.size());
    }
}

PHP_METHOD(OnnxRuntime_EpDevice, registrationName) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    if (!self->info->isPluginDevice()) {
        RETURN_NULL();
    }
    RETURN_STRINGL(self->info->registration_name.c_str(),
                   self->info->registration_name.size());
}

PHP_METHOD(OnnxRuntime_EpDevice, isPlugin) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    if (self->info == nullptr) {
        zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
    }
    RETURN_BOOL(self->info->isPluginDevice());
}

PHP_METHOD(OnnxRuntime_EpDevice, toArray) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->info == nullptr) {
            zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
        }
        php_ort_epdevice_to_array(*self->info, return_value);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading EpDevice");
        RETURN_THROWS();
    }
}

PHP_METHOD(OnnxRuntime_EpDevice, __debugInfo) {
    ZEND_PARSE_PARAMETERS_NONE();
    php_ort_epdevice_object *self =
        php_ort_epdevice_from_obj(Z_OBJ_P(ZEND_THIS));
    try {
        if (self->info == nullptr) {
            zend_throw_exception(onnx_conversion_exception_ce,
                             "EpDevice is uninitialized", 0);
        RETURN_THROWS();
        }
        php_ort_epdevice_to_array(*self->info, return_value);
    } catch (...) {
        php_ort_throw_native(std::current_exception(),
                             "out of memory while reading EpDevice");
        RETURN_THROWS();
    }
}

static const zend_function_entry onnx_epdevice_methods[] = {
    PHP_ME(OnnxRuntime_EpDevice, __construct, arginfo_onnx_epdevice_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, epName, arginfo_onnx_epdevice_string,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, epVendor, arginfo_onnx_epdevice_string,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, deviceType, arginfo_onnx_epdevice_string,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, vendorId, arginfo_onnx_epdevice_long,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, deviceId, arginfo_onnx_epdevice_long,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, deviceVendor, arginfo_onnx_epdevice_string,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, metadata, arginfo_onnx_epdevice_array,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, options, arginfo_onnx_epdevice_array,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, registrationName,
           arginfo_onnx_epdevice_nullable_string, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, isPlugin, arginfo_onnx_epdevice_bool,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, toArray, arginfo_onnx_epdevice_array,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_EpDevice, __debugInfo, arginfo_onnx_epdevice_array,
           ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry onnx_session_options_methods[] = {
    PHP_ME(OnnxRuntime_SessionOptions, __construct,
           arginfo_onnx_session_options_construct, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setIntraOpNumThreads,
           arginfo_onnx_session_options_set_threads, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setInterOpNumThreads,
           arginfo_onnx_session_options_set_threads, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setGraphOptimizationLevel,
           arginfo_onnx_session_options_set_string, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setExecutionMode,
           arginfo_onnx_session_options_set_string, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setLogSeverityLevel,
           arginfo_onnx_session_options_set_log_level, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setEnableCpuMemArena,
           arginfo_onnx_session_options_set_bool, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, setEnableMemPattern,
           arginfo_onnx_session_options_set_bool, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, appendExecutionProvider,
           arginfo_onnx_session_options_append_ep, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, appendEpDevice,
           arginfo_onnx_session_options_append_epdevice, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_SessionOptions, toArray,
           arginfo_onnx_session_options_to_array, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry onnx_runtime_methods[] = {
    PHP_ME(OnnxRuntime_Runtime, __construct, arginfo_onnx_runtime_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, default, arginfo_onnx_runtime_default,
           ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(OnnxRuntime_Runtime, getVersion, arginfo_onnx_runtime_version,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, getBuildInfo, arginfo_onnx_runtime_version,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, getAvailableProviders,
           arginfo_onnx_runtime_string_array, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, getEpDevices,
           arginfo_onnx_runtime_ep_devices, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, registerExecutionProviderLibrary,
           arginfo_onnx_runtime_register, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, unregisterExecutionProviderLibrary,
           arginfo_onnx_runtime_unregister, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, setAsyncThreads,
           arginfo_onnx_runtime_set_async_threads, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, setAsyncQueueCapacity,
           arginfo_onnx_runtime_set_async_queue_capacity, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Runtime, asyncInfo,
           arginfo_onnx_runtime_async_info, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry onnx_session_methods[] = {
    PHP_ME(OnnxRuntime_Session, __construct, arginfo_onnx_session_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, inputNames, arginfo_onnx_session_input_names,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, outputNames, arginfo_onnx_session_output_names,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, inputTypes, arginfo_onnx_session_input_types,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, outputTypes, arginfo_onnx_session_output_types,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, inputShapes, arginfo_onnx_session_input_shapes,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, outputShapes, arginfo_onnx_session_output_shapes,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, run, arginfo_onnx_session_run, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, runBatch, arginfo_onnx_session_run_batch,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, runAsync, arginfo_onnx_session_run_async,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, createIoBinding,
           arginfo_onnx_session_create_io_binding, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Session, runWithBinding,
           arginfo_onnx_session_run_with_binding, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry onnx_iobinding_methods[] = {
    PHP_ME(OnnxRuntime_IoBinding, __construct,
           arginfo_onnx_iobinding_construct, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_IoBinding, bindInput,
           arginfo_onnx_iobinding_bind_input, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_IoBinding, bindOutput,
           arginfo_onnx_iobinding_bind_output, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_IoBinding, bindOutputBuffer,
           arginfo_onnx_iobinding_bind_output_buffer, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_IoBinding, getOutputs,
           arginfo_onnx_iobinding_get_outputs, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry onnx_task_methods[] = {
    PHP_ME(OnnxRuntime_Task, __construct, arginfo_onnx_task_construct,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, getState, arginfo_onnx_task_get_state,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, isReady, arginfo_onnx_task_bool,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, wait, arginfo_onnx_task_wait, ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, getResult, arginfo_onnx_task_get_result,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, getError, arginfo_onnx_task_get_error,
           ZEND_ACC_PUBLIC)
    PHP_ME(OnnxRuntime_Task, cancel, arginfo_onnx_task_bool,
           ZEND_ACC_PUBLIC)
    PHP_FE_END
};

/* ----------------------------------------------------------------------------
 * Module lifecycle
 * -------------------------------------------------------------------------*/

PHP_MINIT_FUNCTION(onnxruntime) {
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "Exception", nullptr);
    onnx_exception_ce =
        zend_register_internal_class_ex(&ce, spl_ce_RuntimeException);

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "ConversionException", nullptr);
    onnx_conversion_exception_ce =
        zend_register_internal_class_ex(&ce, onnx_exception_ce);

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "SessionOptions",
                        onnx_session_options_methods);
    onnx_session_options_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_session_options_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_session_options_handlers.offset =
        XtOffsetOf(php_ort_session_options_object, std);
    onnx_session_options_handlers.free_obj =
        onnx_session_options_free_object;
    onnx_session_options_handlers.clone_obj =
        onnx_session_options_clone_object;
    onnx_session_options_ce->create_object =
        onnx_session_options_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "EpDevice",
                        onnx_epdevice_methods);
    onnx_epdevice_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_epdevice_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_epdevice_handlers.offset =
        XtOffsetOf(php_ort_epdevice_object, std);
    onnx_epdevice_handlers.free_obj = onnx_epdevice_free_object;
    onnx_epdevice_handlers.clone_obj = onnx_epdevice_clone_object;
    onnx_epdevice_ce->create_object = onnx_epdevice_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "Task", onnx_task_methods);
    onnx_task_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_task_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_task_handlers.offset = XtOffsetOf(php_ort_task_object, std);
    onnx_task_handlers.free_obj = onnx_task_free_object;
    onnx_task_handlers.clone_obj = onnx_task_clone_object;
    onnx_task_ce->create_object = onnx_task_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "Runtime", onnx_runtime_methods);
    onnx_runtime_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_runtime_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_runtime_handlers.offset = XtOffsetOf(php_ort_runtime_object, std);
    onnx_runtime_handlers.free_obj = onnx_runtime_free_object;
    onnx_runtime_handlers.clone_obj = onnx_runtime_clone_object;
    onnx_runtime_ce->create_object = onnx_runtime_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "OrtValue", onnx_ortvalue_methods);
    onnx_ortvalue_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_ortvalue_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_ortvalue_handlers.offset = XtOffsetOf(php_ort_ortvalue_object, std);
    onnx_ortvalue_handlers.free_obj = onnx_ortvalue_free_object;
    onnx_ortvalue_handlers.clone_obj = onnx_ortvalue_clone_object;
    onnx_ortvalue_handlers.get_gc = onnx_ortvalue_get_gc;
    onnx_ortvalue_ce->create_object = onnx_ortvalue_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "Tensor", onnx_tensor_methods);
    onnx_tensor_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_tensor_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_tensor_handlers.offset = XtOffsetOf(php_ort_tensor_object, std);
    onnx_tensor_handlers.free_obj = onnx_tensor_free_object;
    onnx_tensor_handlers.clone_obj = onnx_tensor_clone_object;
    onnx_tensor_ce->create_object = onnx_tensor_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "Session", onnx_session_methods);
    onnx_session_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_session_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_session_handlers.offset = XtOffsetOf(php_ort_session_object, std);
    onnx_session_handlers.free_obj = onnx_session_free_object;
    onnx_session_handlers.clone_obj = onnx_session_clone_object;
    onnx_session_ce->create_object = onnx_session_create_object;

    INIT_NS_CLASS_ENTRY(ce, "OnnxRuntime", "IoBinding",
                        onnx_iobinding_methods);
    onnx_iobinding_ce = php_ort_register_class_with_flags(
        &ce, nullptr, ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES);

    std::memcpy(&onnx_iobinding_handlers, zend_get_std_object_handlers(),
                sizeof(zend_object_handlers));
    onnx_iobinding_handlers.offset = XtOffsetOf(php_ort_iobinding_object, std);
    onnx_iobinding_handlers.free_obj = onnx_iobinding_free_object;
    onnx_iobinding_handlers.clone_obj = onnx_iobinding_clone_object;
    onnx_iobinding_ce->create_object = onnx_iobinding_create_object;

    try {
        php_ort::initializeEnvironment();
    } catch (...) {
        // MINIT is module startup, not a userland method call: log the
        // native failure and return FAILURE instead of leaving a pending
        // PHP exception on the startup path.
        php_ort_log_startup_error(std::current_exception());
        return FAILURE;
    }

    return SUCCESS;
}

PHP_RINIT_FUNCTION(onnxruntime) {
#if defined(ZTS) && defined(COMPILE_DL_ONNXRUNTIME)
    // Required before any EG(...) access in a ZTS build.
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(onnxruntime) {
    php_ort::shutdownAsyncEngine();
    php_ort::shutdownRuntime();
    php_ort::shutdownEnvironment();
    return SUCCESS;
}

PHP_MINFO_FUNCTION(onnxruntime) {
    php_info_print_table_start();
    php_info_print_table_header(2, "onnxruntime support", "enabled");
    php_info_print_table_row(2, "extension version",
                             PHP_ONNXRUNTIME_VERSION);
    php_info_print_table_row(2, "ONNX Runtime C API",
                             "v1.28.2 (ORT_API_VERSION 28, linked)");
    php_info_print_table_row(2, "phase",
                             "6: native async inference");
    php_info_print_table_end();
}

zend_module_entry onnxruntime_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_ONNXRUNTIME_EXTNAME,
    nullptr, /* functions */
    PHP_MINIT(onnxruntime),
    PHP_MSHUTDOWN(onnxruntime),
    PHP_RINIT(onnxruntime),
    nullptr, /* RSHUTDOWN */
    PHP_MINFO(onnxruntime),
    PHP_ONNXRUNTIME_VERSION,
    STANDARD_MODULE_PROPERTIES,
};

#ifdef COMPILE_DL_ONNXRUNTIME
#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif
ZEND_GET_MODULE(onnxruntime)
#endif
