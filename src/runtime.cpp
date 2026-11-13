#include "runtime.h"

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_tensor.h"
#include "ort_environment.h"

namespace php_ort {
namespace {

struct PluginRegistration {
    std::string path;
    uint64_t generation = 0;
    size_t active_leases = 0;
    std::set<std::string> device_keys;
};

std::mutex g_plugin_mutex;
std::unordered_map<std::string, PluginRegistration> g_plugin_libraries;
// Device key -> registration name.  Device keys are snapshots from
// EpDeviceInfo::key(), never raw OrtEpDevice pointers.
std::unordered_map<std::string, std::string> g_device_owners;
std::atomic<bool> g_shutting_down{false};
uint64_t g_next_generation = 1;

std::set<std::string> currentDeviceKeys(Ort::Env &env) {
    std::set<std::string> keys;
    for (EpDeviceInfo device : readEpDevices(env)) {
        auto owner = g_device_owners.find(device.identityKey());
        if (owner != g_device_owners.end()) {
            device.registration_name = owner->second;
        }
        keys.insert(device.key());
    }
    return keys;
}

}  // namespace

void ensureRuntimeAcceptingWork() {
    if (g_shutting_down.load(std::memory_order_acquire)) {
        throw ConversionError(
            "ONNX Runtime extension is shutting down; new Sessions and "
            "plugin registrations are disabled");
    }
}

void registerExecutionProviderLibrary(const std::string &registration_name,
                                      const std::string &library_path) {
    ensureRuntimeAcceptingWork();
    if (registration_name.empty()) {
        throw ConversionError("execution provider registration name is empty");
    }
    if (library_path.empty()) {
        throw ConversionError("execution provider library path is empty");
    }
    if (registration_name.find('\0') != std::string::npos ||
        library_path.find('\0') != std::string::npos) {
        throw ConversionError(
            "execution provider registration name/path contains NUL");
    }

    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    if (g_plugin_libraries.find(registration_name) !=
        g_plugin_libraries.end()) {
        throw ConversionError(
            "execution provider library is already registered: " +
            registration_name);
    }

    auto env = environment();
    const std::set<std::string> before = currentDeviceKeys(*env);

    env->RegisterExecutionProviderLibrary(registration_name.c_str(),
                                          library_path.c_str());

    PluginRegistration registration;
    registration.path = library_path;
    registration.generation = g_next_generation++;
    try {
        for (const EpDeviceInfo &raw_device : readEpDevices(*env)) {
            EpDeviceInfo device = raw_device;
            device.registration_name = registration_name;
            device.registration_generation = registration.generation;
            const std::string key = device.key();
            if (before.find(key) == before.end()) {
                registration.device_keys.insert(key);
                g_device_owners[device.identityKey()] = registration_name;
            }
        }
    } catch (...) {
        // Registration itself succeeded but device bookkeeping failed.  Do
        // not leave a half-tracked library or device-owner entries.
        for (auto owner = g_device_owners.begin();
             owner != g_device_owners.end();) {
            if (owner->second == registration_name) {
                owner = g_device_owners.erase(owner);
            } else {
                ++owner;
            }
        }
        try {
            env->UnregisterExecutionProviderLibrary(registration_name.c_str());
        } catch (...) {
        }
        throw;
    }

    g_plugin_libraries.emplace(registration_name, std::move(registration));
}

void unregisterExecutionProviderLibrary(const std::string &registration_name) {
    ensureRuntimeAcceptingWork();
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    auto it = g_plugin_libraries.find(registration_name);
    if (it == g_plugin_libraries.end()) {
        throw ConversionError(
            "execution provider library is not registered: " +
            registration_name);
    }
    if (it->second.active_leases > 0) {
        throw ConversionError(
            "execution provider library still has " +
            std::to_string(it->second.active_leases) +
            " active SessionState lease(s): " + registration_name);
    }

    auto env = environment();
    env->UnregisterExecutionProviderLibrary(registration_name.c_str());

    for (auto owner = g_device_owners.begin();
         owner != g_device_owners.end();) {
        if (owner->second == registration_name) {
            owner = g_device_owners.erase(owner);
        } else {
            ++owner;
        }
    }
    g_plugin_libraries.erase(it);
}

void validatePluginDevice(const EpDeviceInfo &device) {
    if (device.registration_name.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    if (device.registration_generation == 0) {
        throw ConversionError(
            "EpDevice snapshot has no Plugin EP registration generation");
    }
    auto it = g_plugin_libraries.find(device.registration_name);
    if (it == g_plugin_libraries.end()) {
        throw ConversionError(
            "Plugin EP was unregistered: " + device.registration_name);
    }
    if (it->second.generation != device.registration_generation) {
        throw ConversionError(
            "EpDevice belongs to an older Plugin EP registration generation: " +
            device.registration_name);
    }
    if (it->second.device_keys.find(device.key()) ==
        it->second.device_keys.end()) {
        throw ConversionError(
            "EP device does not belong to the registered Plugin EP: " +
            device.registration_name);
    }
}

std::vector<EpDeviceInfo> runtimeEpDevices() {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    auto env = environment();
    std::vector<EpDeviceInfo> result = readEpDevices(*env);
    for (EpDeviceInfo &device : result) {
        auto owner = g_device_owners.find(device.identityKey());
        if (owner == g_device_owners.end()) {
            continue;
        }
        auto registration = g_plugin_libraries.find(owner->second);
        if (registration == g_plugin_libraries.end()) {
            continue;
        }
        device.registration_name = owner->second;
        device.registration_generation = registration->second.generation;
    }
    return result;
}

std::vector<std::string> acquirePluginSessions(
    const std::vector<EpDeviceInfo> &devices) {
    ensureRuntimeAcceptingWork();
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    std::set<std::string> names;
    for (const EpDeviceInfo &device : devices) {
        if (device.registration_name.empty()) {
            continue;
        }
        if (device.registration_generation == 0) {
            throw ConversionError(
                "EpDevice snapshot has no Plugin EP registration generation");
        }
        auto it = g_plugin_libraries.find(device.registration_name);
        if (it == g_plugin_libraries.end()) {
            throw ConversionError(
                "Plugin EP was unregistered before Session creation: " +
                device.registration_name);
        }
        if (it->second.generation != device.registration_generation) {
            throw ConversionError(
                "EpDevice belongs to an older Plugin EP registration generation: " +
                device.registration_name);
        }
        if (it->second.device_keys.find(device.key()) ==
            it->second.device_keys.end()) {
            throw ConversionError(
                "EP device does not belong to the registered Plugin EP: " +
                device.registration_name);
        }
        names.insert(device.registration_name);
    }

    std::vector<std::string> acquired(names.begin(), names.end());
    for (const std::string &name : acquired) {
        ++g_plugin_libraries[name].active_leases;
    }
    return acquired;
}

void releasePluginSessions(const std::vector<std::string> &registrations) {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    for (const std::string &name : registrations) {
        auto it = g_plugin_libraries.find(name);
        if (it != g_plugin_libraries.end() &&
            it->second.active_leases > 0) {
            --it->second.active_leases;
        }
    }
}

void shutdownRuntime() {
    // Stop phase: no new Sessions or plugin registrations may start.
    g_shutting_down.store(true, std::memory_order_release);

    // Cleanup phase: unload only registrations with no SessionState lease.
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    if (g_plugin_libraries.empty()) {
        return;
    }

    try {
        auto env = environment();
        for (const auto &entry : g_plugin_libraries) {
            // SessionState objects should already be gone at MSHUTDOWN.  If a
            // teardown path left one alive, refusing unload is safer than
            // invalidating a live Ort::Session.  The Env remains owned by the
            // SessionState shared_ptr, so releasing the global reference later
            // cannot destroy it underneath a live Session.
            if (entry.second.active_leases > 0) {
                continue;
            }
            try {
                env->UnregisterExecutionProviderLibrary(
                    entry.first.c_str());
            } catch (...) {
                // Shutdown must not throw through the Zend module callback.
                // ORT may already have torn down the provider if another
                // shutdown path failed first.
            }
        }
    } catch (...) {
        // Environment is already gone; the process is exiting.
    }
    g_plugin_libraries.clear();
    g_device_owners.clear();
}

}  // namespace php_ort
