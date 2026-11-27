# OpenVINO Plugin EP

OpenVINO is a **Plugin EP**: it is not compiled into the ONNX Runtime build.
The plugin is a separate library that you register at runtime, after which its
devices appear as `EpDevice` objects next to the built-in ones.

Related docs: [EP-CPU.md](EP-CPU.md), [EP-COREML.md](EP-COREML.md),
[EP-CUDA.md](EP-CUDA.md), and the end-to-end
[YOLO11 + PP-OCR example](EXAMPLE-OPENVINO-YOLO-OCR.md) (GD preprocessing,
per-stage timings, cold vs warm comparison against the CPU EP).

## What you need installed

| Requirement | Detail |
| --- | --- |
| Plugin package | `Intel.ML.OnnxRuntime.EP.OpenVINO`. Version **1.8.1** was tested against ORT 1.32.0. |
| OpenVINO runtime | Ships inside the same NuGet package under `runtimes/<rid>/native/`. Do not mix it with a system OpenVINO of a different major version. |
| TBB / hwloc | Also inside the package (`libtbb.so.12`, `libhwloc.so.15`). Keep them next to the plugin. |
| ORT version | The plugin is built against a specific ORT version. It must be loaded into a runtime of the same version; a mismatch fails at registration or at session creation. |
| CPU | An Intel CPU is expected. On other x86_64 CPUs OpenVINO's CPU plugin may refuse to load or silently fall back. |

Unpack the `.nupkg` (it is a ZIP) and keep one directory per platform:

```text
runtimes/linux-x64/native/
  libonnxruntime_providers_openvino_plugin.so   # the plugin
  libopenvino.so.2641                           # OpenVINO core
  libopenvino_intel_cpu_plugin.so               # CPU device
  libtbb.so.12
  libhwloc.so.15
```

`runtimes/win-x64/native/` has the same set as `*.dll`
(`onnxruntime_providers_openvino_plugin.dll`, `openvino.dll`, `tbb12.dll`, …).

## Build

The extension itself needs **no** special configure flag — the plugin is opened
at runtime, not linked:

```bash
phpize
./configure            # ordinary CPU NuGet build, ORT 1.32.0
make -j8
```

## Using it

```php
$runtime = OnnxRuntime\Runtime::default();
$runtime->registerExecutionProviderLibrary(
    'openvino',
    '/opt/openvino-ep/runtimes/linux-x64/native/libonnxruntime_providers_openvino_plugin.so'
);

$options = new OnnxRuntime\SessionOptions();
$selected = null;
foreach ($runtime->getEpDevices() as $device) {
    if ($device->epName() === 'OpenVINOExecutionProvider') {
        $selected = $device;
        break;
    }
}
if ($selected === null) {
    throw new RuntimeException('OpenVINO plugin registered but exposed no device');
}
$options->appendEpDevice($selected);

$session = new OnnxRuntime\Session('model.onnx', $options);
```

Search paths must contain **both** the ORT library and the plugin directory:

```bash
# Linux
export LD_LIBRARY_PATH=/opt/ort/lib:/opt/openvino-ep/runtimes/linux-x64/native
```

On Windows put the plugin and the OpenVINO/TBB DLLs in one directory and add it
to `PATH`, or copy them next to `php.exe` (the same DLL-search-order trap as the
CUDA runtime: `C:\Windows\System32` wins over `PATH`).

Devices exposed by 1.8.1 on a Linux x86_64 host:

```text
ep=OpenVINOExecutionProvider       type=cpu vendor=Intel reg='openvino' plugin=true
ep=OpenVINOExecutionProvider.AUTO  type=cpu vendor=Intel reg='openvino' plugin=true
```

`OpenVINOExecutionProvider` is the CPU device; `OpenVINOExecutionProvider.AUTO`
lets OpenVINO pick the underlying device itself (GPU or NPU on hosts that have
them). Both report `deviceType() === 'cpu'` in this release, so select by
`epName()`, not by device type.

## Threads: the option you want is not the one you would guess

Three knobs exist, and only the last one works:

| Knob | Effect |
| --- | --- |
| `SessionOptions::setIntraOpNumThreads()` (`--threads` in the bench scripts) | Ignored by the plugin's own pool. Wall time stays ~101 ms whether it is set to 1 or 4. |
| `appendEpDevice(..., ['num_of_threads' => '1'])` | Ignored. The plugin logs: `Provider option num_of_threads is ignored. Please set it via the load_config as: "INFERENCE_NUM_THREADS": "1"`. |
| `appendEpDevice(..., ['load_config' => '{"CPU":{"INFERENCE_NUM_THREADS":"1"}}'])` | Works. Device-keyed JSON, one nesting level, device key `CPU`, property `INFERENCE_NUM_THREADS`. |

```php
$options->appendEpDevice($selected, [
    // device key -> property map; the plugin accepts one level of nesting
    'load_config' => '{"CPU":{"INFERENCE_NUM_THREADS":"1"}}',
]);
```

`OMP_NUM_THREADS` has no effect either — the plugin is TBB-based, not
OpenMP-based.

Measured on the Linux benchmark host (2 physical cores / 4 logical CPUs, SMT;
BGE-small int8, batch 8 × 64, 20 runs, `benchmarks/php_ep_cpu_profile.php`):

| Session | p50 | Process CPU |
| --- | ---: | ---: |
| CPU EP, 2 intra-op threads | 110.0 ms | 187% |
| CPU EP, 4 intra-op threads | 106.5 ms | 354% |
| OpenVINO, `INFERENCE_NUM_THREADS=1` | 197.5 ms | 101% |
| OpenVINO, `INFERENCE_NUM_THREADS=2` | 101.6 ms | 175% |
| OpenVINO, default (no `load_config`) | 100.4 ms | 176% |
| OpenVINO, `--threads=4` only | 100.7 ms | 176% |

Read this as: OpenVINO's default footprint on this host is about two threads
(~1.7 cores), which is the physical core count. Against CPU EP at the same
budget (2 threads: 110 ms / 187%) it is ~8% faster. Against CPU EP at 4 threads
it is ~5% faster while using half the CPU. It does not scale beyond its default
width just because you raised ORT's intra-op setting.

The Windows 3.41× headline (71.9 ms vs 245.2 ms) in `benchmarks/README.md` is a
wall-clock ratio at unequal CPU budgets — OpenVINO at its default width against
CPU EP pinned to one thread — and should not be read as per-core throughput.

## Caveats

- Registration is process-wide and lives until module shutdown; destroying the
  PHP `Runtime` object does not unload the plugin. `unregisterExecutionProviderLibrary('openvino')`
  is rejected while a Session still holds a lease.
- A Session taking the plugin device reserves a lease; the plugin is only
  unloaded at zero leases.
- `DNNLExecutionProvider` is **not** available in Microsoft's prebuilt packages.
  oneDNN is used underneath OpenVINO's CPU plugin; an explicit DNNL EP would
  require an ORT source build with `--use_dnnl`.
- OpenVINO covers a subset of ONNX operators. Unsupported nodes fall back to the
  CPU EP, so a session can report `OpenVINOExecutionProvider` and still spend most
  of its time on CPU.
- On non-Intel x86_64 hosts, or hosts without the matching OpenVINO runtime
  version, registration can succeed while session creation fails. Check
  `getEpDevices()` before assuming the plugin is usable.
