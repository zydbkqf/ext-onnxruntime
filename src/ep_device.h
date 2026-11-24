#ifndef PHP_ORT_EP_DEVICE_H
#define PHP_ORT_EP_DEVICE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

/* EP-device and DeviceTensor support needs OrtEpDevice (API 22) plus
 * OrtDeviceMemoryType (API 23).  Builds against older ONNX Runtime versions
 * keep the PHP class surface but every device-tensor operation reports that
 * the feature is unavailable. */
#if ORT_API_VERSION >= 23
# define PHP_ORT_HAS_EP_DEVICE 1
#else
# define PHP_ORT_HAS_EP_DEVICE 0
#endif

#if ORT_API_VERSION < 22
/* API versions before 1.22 have no OrtHardwareDeviceType.  Mirror the values
 * used by newer runtimes so the identity snapshot keeps its layout. */
typedef enum OrtHardwareDeviceType {
    OrtHardwareDeviceType_CPU,
    OrtHardwareDeviceType_GPU,
    OrtHardwareDeviceType_NPU
} OrtHardwareDeviceType;
#endif

namespace php_ort {

/**
 * Framework-independent identity of an OrtEpDevice.
 *
 * ORT exposes device pointers that are owned by the process-level Env and
 * invalidated when a plugin library is unregistered.  This snapshot is safe
 * to keep in a PHP object or SessionConfig; the pointer itself is resolved
 * again from the Env immediately before SessionOptionsAppendExecutionProvider_V2.
 *
 * The key is valid only for this Env lifetime.  It is not a cross-process or
 * cross-Env persistent identifier.  registration_generation prevents an old
 * snapshot from silently binding to a newly registered library with the same
 * registration name.
 */
struct EpDeviceInfo {
    std::string ep_name;
    std::string ep_vendor;
    std::string device_vendor;
    OrtHardwareDeviceType device_type = OrtHardwareDeviceType_CPU;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;

    // Distinguishes identical devices exposed by the same plugin.  Assigned
    // by readEpDevices() within one Env enumeration.
    uint32_t instance_ordinal = 0;

    // Zero for a built-in device; a positive registry generation for a
    // device owned by a Plugin EP registration.
    uint64_t registration_generation = 0;

    std::map<std::string, std::string> ep_metadata;
    std::map<std::string, std::string> ep_options;

    // Set when this device belongs to a registered Plugin EP library.  It is
    // informational identity, not an ownership reference.
    std::string registration_name;

    bool isPluginDevice() const {
        return !registration_name.empty() && registration_generation != 0;
    }

    /** Stable identity within one process Env; used only for registry lookup. */
    std::string key() const;

    /** Device identity without registration name; used to find its owner. */
    std::string identityKey() const;

    /** Identity without the duplicate-device ordinal. */
    std::string baseKey() const;

#if PHP_ORT_HAS_EP_DEVICE
    bool matches(const Ort::ConstEpDevice &device) const;
#endif
};

#if PHP_ORT_HAS_EP_DEVICE
EpDeviceInfo readEpDevice(const Ort::ConstEpDevice &device);
#endif

/**
 * Read every device and assign instance_ordinal to duplicate identities
 * exposed by the same EP.  Returns an empty list when the build targets an
 * ONNX Runtime older than 1.23.
 */
std::vector<EpDeviceInfo> readEpDevices(Ort::Env &env);

#if PHP_ORT_HAS_EP_DEVICE
/**
 * Resolve a snapshot to the current Env-owned OrtEpDevice.  Throws
 * ConversionError if the device disappeared (for example after unregister).
 */
Ort::ConstEpDevice resolveEpDevice(Ort::Env &env, const EpDeviceInfo &info);
#endif

}  // namespace php_ort

#endif  // PHP_ORT_EP_DEVICE_H
