#ifndef PHP_ORT_RUNTIME_H
#define PHP_ORT_RUNTIME_H

#include <string>
#include <vector>

#include "ep_device.h"

namespace php_ort {

/**
 * Register an ONNX Runtime execution-provider plugin library on the
 * process-level Env.  Registration is process-scoped and survives individual
 * PHP Runtime objects.
 */
void registerExecutionProviderLibrary(const std::string &registration_name,
                                      const std::string &library_path);

/**
 * Unregister a plugin library.  The call is rejected while any SessionState
 * still holds a lease from that registration.
 */
void unregisterExecutionProviderLibrary(const std::string &registration_name);

/** Reject Session/plugin work once module shutdown has started. */
void ensureRuntimeAcceptingWork();

/**
 * Validate that a Plugin EpDevice snapshot belongs to the current
 * registration generation and still maps to a registered device.  Built-in
 * devices are ignored.
 */
void validatePluginDevice(const EpDeviceInfo &device);

/** Snapshot Env EP devices, enriched with their Plugin EP registration. */
std::vector<EpDeviceInfo> runtimeEpDevices();

/**
 * Reserve a lease for the Plugin EP registrations used by these devices.
 * The reservation is made before Ort::Session creation and must be released
 * only after the native SessionState is destroyed.  Built-in devices produce
 * no entries.
 */
std::vector<std::string> acquirePluginSessions(
    const std::vector<EpDeviceInfo> &devices);

/** Decrement active-session counts; must run after the Ort::Session is gone. */
void releasePluginSessions(const std::vector<std::string> &registrations);

/** Unregister every plugin library tracked by this extension. */
void shutdownRuntime();

}  // namespace php_ort

#endif  // PHP_ORT_RUNTIME_H
