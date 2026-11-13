#include "ep_device.h"

#include <map>
#include <sstream>
#include <utility>

#include "onnx_tensor.h"

namespace php_ort {
namespace {

std::map<std::string, std::string> readKeyValueMap(
    const Ort::ConstKeyValuePairs &pairs) {
    std::map<std::string, std::string> result;
    if (!pairs) {
        return result;
    }
    for (const auto &entry : pairs.GetKeyValuePairs()) {
        result.emplace(entry.first, entry.second);
    }
    return result;
}

std::string encodeKeyPart(const std::string &value) {
    std::ostringstream out;
    out << value.size() << ':' << value;
    return out.str();
}

void appendMapKey(std::ostringstream &out,
                  const std::map<std::string, std::string> &map) {
    out << map.size() << ';';
    for (const auto &entry : map) {
        out << encodeKeyPart(entry.first) << encodeKeyPart(entry.second);
    }
}

}  // namespace

std::string EpDeviceInfo::baseKey() const {
    std::ostringstream out;
    out << encodeKeyPart(ep_name) << encodeKeyPart(ep_vendor)
        << encodeKeyPart(device_vendor) << static_cast<int>(device_type) << ':'
        << vendor_id << ':' << device_id << ';';
    appendMapKey(out, ep_metadata);
    out << ';';
    appendMapKey(out, ep_options);
    return out.str();
}

std::string EpDeviceInfo::identityKey() const {
    std::ostringstream out;
    out << baseKey() << '#' << instance_ordinal;
    return out.str();
}

std::string EpDeviceInfo::key() const {
    std::ostringstream out;
    out << identityKey() << '|' << encodeKeyPart(registration_name);
    return out.str();
}

bool EpDeviceInfo::matches(const Ort::ConstEpDevice &device) const {
    if (!device) {
        return false;
    }

    const char *name = device.EpName();
    const char *vendor = device.EpVendor();
    if (ep_name != (name ? name : "") || ep_vendor != (vendor ? vendor : "")) {
        return false;
    }

    auto hardware = device.Device();
    if (!hardware) {
        return false;
    }
    if (device_type != hardware.Type() || vendor_id != hardware.VendorId() ||
        device_id != hardware.DeviceId()) {
        return false;
    }
    const char *hardware_vendor = hardware.Vendor();
    if (device_vendor != (hardware_vendor ? hardware_vendor : "")) {
        return false;
    }

    return ep_metadata == readKeyValueMap(device.EpMetadata()) &&
           ep_options == readKeyValueMap(device.EpOptions());
}

EpDeviceInfo readEpDevice(const Ort::ConstEpDevice &device) {
    if (!device) {
        throw ConversionError("ORT returned a null EP device");
    }

    EpDeviceInfo info;
    const char *name = device.EpName();
    const char *vendor = device.EpVendor();
    info.ep_name = name ? name : "";
    info.ep_vendor = vendor ? vendor : "";

    auto hardware = device.Device();
    if (hardware) {
        info.device_type = hardware.Type();
        info.vendor_id = hardware.VendorId();
        info.device_id = hardware.DeviceId();
        const char *hardware_vendor = hardware.Vendor();
        info.device_vendor = hardware_vendor ? hardware_vendor : "";
    }

    info.ep_metadata = readKeyValueMap(device.EpMetadata());
    info.ep_options = readKeyValueMap(device.EpOptions());
    return info;
}

std::vector<EpDeviceInfo> readEpDevices(Ort::Env &env) {
    std::vector<EpDeviceInfo> result;
    std::map<std::string, uint32_t> ordinals;
    for (const auto &device : env.GetEpDevices()) {
        EpDeviceInfo info = readEpDevice(device);
        const std::string base = info.baseKey();
        info.instance_ordinal = ordinals[base]++;
        result.push_back(std::move(info));
    }
    return result;
}

std::vector<Ort::ConstEpDevice> getEpDevices(Ort::Env &env) {
    return env.GetEpDevices();
}

Ort::ConstEpDevice resolveEpDevice(Ort::Env &env, const EpDeviceInfo &info) {
    // Enumerate with ordinals so a snapshot for the second of two identical
    // devices cannot silently resolve to the first one.
    std::map<std::string, uint32_t> ordinals;
    for (const auto &device : env.GetEpDevices()) {
        EpDeviceInfo candidate = readEpDevice(device);
        const uint32_t ordinal = ordinals[candidate.baseKey()]++;
        if (candidate.matches(device) && info.matches(device) &&
            info.instance_ordinal == ordinal) {
            return device;
        }
    }
    throw ConversionError(
        "EP device is not available: " + info.ep_name +
        "; it may have been unregistered or the device may have disappeared");
}

}  // namespace php_ort
