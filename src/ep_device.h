#ifndef PHP_ORT_EP_DEVICE_H
#define PHP_ORT_EP_DEVICE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

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

    bool matches(const Ort::ConstEpDevice &device) const;
};

EpDeviceInfo readEpDevice(const Ort::ConstEpDevice &device);

/**
 * Read every device and assign instance_ordinal to duplicate identities
 * exposed by the same EP.  This lets two otherwise identical plugin devices
 * retain distinct registry keys.
 */
std::vector<EpDeviceInfo> readEpDevices(Ort::Env &env);

std::vector<Ort::ConstEpDevice> getEpDevices(Ort::Env &env);

/**
 * Resolve a snapshot to the current Env-owned OrtEpDevice.  Throws
 * ConversionError if the device disappeared (for example after unregister).
 */
Ort::ConstEpDevice resolveEpDevice(Ort::Env &env, const EpDeviceInfo &info);

}  // namespace php_ort

#endif  // PHP_ORT_EP_DEVICE_H
