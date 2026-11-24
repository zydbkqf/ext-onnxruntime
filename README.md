# PHP ONNX Runtime Extension

A native PHP extension that loads ONNX models and runs inference directly from PHP, using the Zend C API, C++, and the ONNX Runtime native library. It does not use FFI or Python.

Current status: synchronous inference, native async tasks, CPU I/O Binding, fixed-shape preallocated outputs, device tensors, explicit device copies, explicit multi-group batch inference, and async I/O Binding are implemented. The current feature set is verified on macOS arm64 ZTS and NTS, Windows x86_64 ZTS and NTS, Linux x86_64 ZTS, and Linux x86_64 NTS (33/36 to 35/38 PHPT, with Windows CUDA and no-CUDA/TensorRT platform skips; CoreML passes on macOS). The Windows CUDA Plugin EP smoke path has passed: register the provider DLL, enumerate the CUDA `EpDevice`, append it with `appendEpDevice()`, and run inference on the NVIDIA GPU. Windows CUDA and TensorRT inference through `appendExecutionProvider()` also runs on the GPU. This is a smoke result, not general Plugin EP acceptance across every provider, platform, unload sequence, and failure mode. The OpenVINO Plugin EP also runs on Windows and is benchmarked in `benchmarks/README.md`.

Execution providers have two distinct entry points:

- `appendExecutionProvider()` is the built-in EP path and calls ORT's provider-specific append API, including the dedicated CUDA and TensorRT appenders.
- `appendEpDevice()` is the device-based path. It resolves an `EpDevice` from `Runtime::getEpDevices()` and uses `SessionOptionsAppendExecutionProvider_V2`.

When no EP is appended, ORT defaults are preserved; this project does not promise "CPU only by default".

## Before You Use This

1. **This project is mainly for developers who do not want a Python runtime.** It is not automatically faster. Converting PHP zval data into a Tensor has real cost, and depending on payload size and call pattern it may not beat a separate Python process over IPC or HTTP. Benchmark the entire path, not only the ORT kernel time. See `benchmarks/README.md` for the exact workload and current CPU/CoreML/CUDA results.

2. **Do not run long synchronous inference inside a shared PHP-FPM pool.** A request occupies its FPM worker until inference returns; enough concurrent requests can exhaust the pool and stall unrelated traffic. Prefer a dedicated CLI worker, a dedicated process, or an isolated service. If you use threads, use a ZTS build and isolate thread usage, and validate it. ZTS alone does not make an arbitrary application thread-safe.

3. **Inference is compute-bound.** The OS can still preempt a process, but the PHP request does not cooperatively yield, so the current worker is unavailable for the whole run. Long CPU inference raises request latency and reduces concurrency. Use worker isolation, request limits, and capacity planning instead of assuming PHP will "yield" during the computation.

4. **Most PHP server deployments have CPU only.** GPU/NPU access is provided through ONNX Runtime execution providers, and availability depends on the ORT build and platform. EP offload is opt-in and must be benchmarked before production: extra data transfers and unsupported graph partitions can make a small model slower on an EP than on CPU.

5. **INT8-quantized models are often a good starting point for CPU, but not a universal rule.** Quantization can reduce accuracy and may not help every model or backend. Avoid very large models, measure latency, memory, and output quality, and test on the actual target hardware before production.

## Support Status

| Platform | PHP | Thread mode | Status |
| --- | --- | --- | --- |
| macOS arm64 | 8.5.6 / 8.5.12 / 8.6.0 | ZTS / NTS / ZTS | Verified: 34/36 (8.5) and 35/38 (8.6) PHPT; 1–3 platform skips (Windows CUDA `033`, no CUDA/TensorRT `036`, OpenVINO plugin `038`); CoreML passes; ASan, native sync/async stress; PHP 8.6 removed `XtOffsetOf`, handled by a version-guarded offset helper |
| Linux x86_64 | 8.6.0 | NTS | Verified: 33/37 PHPT, 4 platform skips (CoreML, Windows CUDA, no CUDA/TensorRT provider in the CPU SDK, OpenVINO plugin `038`); PHP built from source, ORT 1.28.0 CPU runtime |
| Linux x86_64 | 8.5.12 | NTS | Verified: 33/36 PHPT, 3 platform skips (CoreML, Windows CUDA, OpenVINO plugin `038`); CUDA built-in EP runs inference with the ORT 1.28.2 CUDA 13 GPU package; new `--with-onnxruntime=DIR` build path |
| Linux x86_64 | 8.5.12 | ZTS | Verified: 33/36 PHPT, 3 platform skips (CoreML, Windows CUDA, OpenVINO plugin `038`); CUDA built-in EP runs inference with the ORT 1.28.2 CUDA 13 GPU package; async ASan+LSan passed; Swoole 6.2.1 Fiber/coroutine/process/thread, `pcntl_fork`, `parallel` |
| Linux arm64 | 8.5.12 | NTS/ZTS | Verified: 33/36 PHPT, 3 platform skips (CoreML, Windows CUDA, OpenVINO plugin `038`); ORT 1.28.2 aarch64 CPU SDK; PHPT needs the cpuinfo-warning filter described below |
| FreeBSD 14.5 x86_64 | 8.5.12 | NTS | Verified: 34/38 PHPT, 4 platform skips (CoreML, Windows CUDA, no CUDA/TensorRT provider, OpenVINO plugin `038`); ORT 1.30.0 pkg; CI uses a FreeBSD VM and the pkg-provided PHP; PHPT needs the stderr filter in `docs/FREEBSD.md` |
| Windows x86_64 | 8.5.12 | NTS/ZTS | Verified: 35/37 PHPT, 2 platform skips (CoreML, OpenVINO plugin `038`); `034`/`035` CPU+CUDA async binding; native I/O Binding stress; CUDA Plugin EP smoke; CUDA and TensorRT inference via `appendExecutionProvider()` on the NVIDIA GPU |

The PHP code targets the Zend API in PHP 8.0–8.6 and rejects versions below 8.0 at compile time. Verified environments: macOS arm64 PHP 8.5.6 ZTS, PHP 8.5.12 NTS (Homebrew) and PHP 8.6.0 ZTS; Linux x86_64 PHP 8.5.12 NTS and ZTS (current feature set, Swoole 6.2.1 on ZTS), and the same two builds against the ORT 1.28.2 CUDA 13 GPU package, where the built-in CUDA EP runs inference; Linux arm64 PHP 8.5.12 NTS and ZTS (ORT 1.28.2 aarch64 CPU SDK, source-built PHP); Windows x86_64 PHP 8.5.12 ZTS and NTS with VS 2026 + MSVC 14.44 and ORT 1.28.2, including CUDA and TensorRT inference on an RTX 3050; FreeBSD 14.5 x86_64 PHP 8.5.12 NTS with the `onnxruntime` 1.30.0 package (CPU only; see `docs/FREEBSD.md`). On Linux arm64 and FreeBSD the ORT prebuilt library prints a `cpuid_info` warning on stderr, so PHPT must be run through a wrapper that filters that line — see `docs/LINUX-ARM64-BASELINE.md` and `docs/FREEBSD.md`; without it every test fails on output mismatch, not on behaviour. Debug builds and other unlisted combinations are not claimed as supported. `phpize`/`php-config` determine the ZTS/NTS and Debug/Release configuration; the resulting module must be installed into the extension_dir of the same PHP build and loaded by that same `php` binary.

## Requirements

- PHP 8.x; verified with PHP 8.5.6 CLI ZTS, PHP 8.5.12 CLI NTS (Zend Module API `20250925`) and PHP 8.6.0 CLI ZTS (Zend Module API `20261112`)
- `phpize` / `php-config` from the same PHP installation that will load the extension
- A C++17 compiler (Apple clang on macOS; g++/clang++ on Linux; base clang on FreeBSD)
- ONNX Runtime **1.18.0 or newer** for the core build (`ORT_API_VERSION >= 18`);
  1.32.0 is the current default. The core surface is verified against 1.18, 1.20,
  1.21, 1.23, 1.26 and 1.32. 1.17 and older are rejected by `config.m4` and
  `config.w32`; an exploratory 1.17.3 build compiled, but its bundled PHPT
  fixtures are ONNX IR 10, above ORT 1.17's IR 9 limit, so it is not supported.
- The `EpDevice`/`DeviceTensor`/Plugin-EP API requires **1.23.0 or newer**.
  Builds against 1.18–1.22 keep the same PHP class surface, but device-related
  methods throw `requires ONNX Runtime 1.23 or newer`. There is no separate
  "core" flavor: the same module adapts to whatever SDK version is supplied.

### Getting ONNX Runtime

> **Why NuGet instead of GitHub Releases:** since 2026-11-17 Microsoft has made
> `microsoft/onnxruntime`, `onnxruntime-genai`, `onnxruntime-extensions` and
> related repositories private while it investigates the `agent-framework`
> supply-chain incident. The official GitHub Release assets that this project
> previously downloaded (`onnxruntime-linux-x64-1.28.2.tgz`,
> `onnxruntime-osx-arm64-1.28.2.tgz`, …) now return **404**, and the GitHub
> API reports the repository as missing. Builds therefore use the public
> **NuGet flat-container** endpoint, which still serves versioned packages
> containing both the headers (`build/native/include` or
> `buildTransitive/native/include`) and the runtime/import libraries
> (`runtimes/<rid>/native`).

`./configure` downloads the SDK from NuGet by default:

```bash
phpize
./configure                      # CPU package: Microsoft.ML.OnnxRuntime 1.32.0
make -j8
```

Layout written to the NuGet cache (outside the source tree so `make clean`,
which removes every `*.so`, cannot delete the SDK):

```text
${XDG_CACHE_HOME:-$HOME/.cache}/ext-onnxruntime/nuget/<package>.<version>/
```

`third_party/nuget` is a convenience symlink to that directory. Set
`ONNXRUNTIME_NUGET_DIR` to relocate the cache.

Contents:

```text
include/onnxruntime_cxx_api.h    # all public headers
lib/libonnxruntime.*             # runtime library (+ provider libraries for GPU)
```

The NuGet package is cached; delete the directory to force a fresh download.

Package selection:

| Configure option | NuGet package | Default version |
| --- | --- | --- |
| *(none)* | `Microsoft.ML.OnnxRuntime` | `1.32.0` |
| `--with-cuda12` | `Microsoft.ML.OnnxRuntime.Gpu.Linux` | `1.26.0` |
| `--with-cuda13` | `Microsoft.ML.OnnxRuntime.Gpu.Linux` | `1.32.0` |
| `--with-coreml` | `Microsoft.ML.OnnxRuntime` (macOS arm64) | `1.32.0` |

Footnotes:

- `--with-cuda12` / `--with-cuda13` are Linux x86_64 only. configure checks
  `libcublas.so.<major>` and `libcudnn.so.9`; a missing runtime is a warning
  because build machines often have no GPU driver. Set
  `ONNXRUNTIME_STRICT_CUDA_CHECK=1` to make it fatal.
- CUDA 12 is pinned to **1.26.0** because that is the last
  `Microsoft.ML.OnnxRuntime.Gpu.Linux` NuGet package built against CUDA 12.
  The same package ID switched to CUDA 13 in **1.27.0**; the CUDA 12.8 builds
  that ONNX Runtime publishes for 1.27–1.30 are source builds, not NuGet
  binaries. Moving the default would silently require CUDA 13.0 or newer.
- `--with-coreml` is macOS arm64 only. NuGet's macOS dylib already embeds the
  CoreML EP; the flag asserts platform support and records the requirement.
- Override the version with `ONNXRUNTIME_VERSION`,
  `ONNXRUNTIME_CUDA12_VERSION` or `ONNXRUNTIME_CUDA13_VERSION`.
- macOS x86_64 is not published on NuGet and is not supported by this build path.

Three source modes:

```bash
./configure                                      # NuGet (default, CPU)
./configure --with-onnxruntime                   # system headers/libs
./configure --with-onnxruntime=/opt/ort-1.28.2   # explicit SDK directory
```

`--with-onnxruntime=DIR` accepts a directory with `include/` + `lib/`, or an
already-unpacked SDK root. The bare `--with-onnxruntime` searches
`pkg-config onnxruntime`, `/usr/include/onnxruntime`, `/usr/local/include`,
`/opt/homebrew/include` and the usual library directories.

FreeBSD has no NuGet RID, so the default NuGet mode intentionally fails with an
instructional error. Install `pkg install php85 onnxruntime` and use either
`./configure --with-onnxruntime` or
`./configure --with-onnxruntime=/usr/local`; details are in
`docs/FREEBSD.md`.

### OpenVINO Plugin EP (Windows/Linux)

OpenVINO is not built into the CPU NuGet runtime; it is a standalone plugin EP.
Install `Intel.ML.OnnxRuntime.EP.OpenVINO` (1.8.1 was tested against ORT 1.32),
keep the plugin and its OpenVINO/TBB DLLs together, then register the plugin on
`Runtime` and select its `EpDevice`:

```php
$runtime = OnnxRuntime\Runtime::default();
$runtime->registerExecutionProviderLibrary(
    'openvino',
    'D:\php\openvino-ep\onnxruntime_providers_openvino_plugin.dll'
);

$options = new OnnxRuntime\SessionOptions();
foreach ($runtime->getEpDevices() as $device) {
    if ($device->epName() === 'OpenVINOExecutionProvider') {
        $options->appendEpDevice($device);
        break;
    }
}
$session = new OnnxRuntime\Session('model.onnx', $options);
```

On Windows the plugin and the `openvino.dll` / `tbb*.dll` files must be in the
same directory or on `PATH`; on Linux add that directory to
`LD_LIBRARY_PATH`. The plugin exposes a CPU device and an
`OpenVINOExecutionProvider.AUTO` device. On an i5-12400F with BGE-small int8,
batch 8 × 64, OpenVINO's p50 was 71.9 ms versus 245.2 ms on CPU (3.41×); see
`benchmarks/README.md`.

`DNNLExecutionProvider` is not part of Microsoft's prebuilt NuGet packages:
`appendExecutionProvider('DNNL')` is rejected as an unknown provider. oneDNN is
used underneath OpenVINO's CPU plugin; an explicit DNNL EP requires building
ONNX Runtime from source with `--use_dnnl`.

### NuGet download integrity

`configure` downloads the `.nupkg` into a temporary directory, verifies its
SHA-256, and only then unpacks and atomically moves the SDK into the cache.
A checksum mismatch, download failure, or bad archive removes the temporary
tree and leaves no partially populated SDK directory. The Windows CI action
(`.github/actions/fetch-onnxruntime`) performs the same check before
`Expand-Archive`, and stages the output before replacing the destination
directory.

Pinned defaults:

| Package | Version | SHA-256 |
| --- | --- | --- |
| `Microsoft.ML.OnnxRuntime` (CPU, all RIDs) | 1.32.0 | `a9150bc2992cbbb6722070da9910e1ffa989a8bfc67bdd5bb8e745ca755641db` |
| `Microsoft.ML.OnnxRuntime.Gpu.Linux` (CUDA 13) | 1.32.0 | `3d01949bc846d41961b37988d1e63d782a6effcb93b650a6e0577fef982d38b5` |
| `Microsoft.ML.OnnxRuntime.Gpu.Linux` (CUDA 12) | 1.26.0 | `cf7943f58b9360fe1448c4baf40ac12ce4175978ecd3fe6f21a4c00976f63a59` |
| `Microsoft.ML.OnnxRuntime.Gpu.Windows` (CUDA 13) | 1.32.0 | `485c1af4e5719d7336c2f1cdf6d3342a3fcec88d3bad9621e33a283fff77c4c7` |
| `Microsoft.ML.OnnxRuntime.Gpu.Windows` (CUDA 12) | 1.26.0 | `2820b8cca24b7c2817f28aa6ea8774a5c519295caa7447b86d190c7079aabf0f` |

For a custom version on Unix, supply the digest with `ONNXRUNTIME_SHA256`,
`ONNXRUNTIME_CUDA12_SHA256` or `ONNXRUNTIME_CUDA13_SHA256`. On Windows, pass
`sha256` to the `fetch-onnxruntime` action (the release workflows already do).
If no digest is known, the build prints the actual SHA-256 and warns instead of
silently accepting the download.

The cache lives at
`${XDG_CACHE_HOME:-$HOME/.cache}/ext-onnxruntime/nuget/<package>.<version>/`,
outside the source tree, so `make clean` does not delete it. For a fully offline
build, point `--with-onnxruntime` at an unpacked SDK; that path never downloads
or verifies a NuGet archive.

Distribution development packages and their current versions:

| Source | Package | Version | Core build | EP devices / CoreML |
| --- | --- | --- | --- | --- |
| apt (Debian 13 trixie) | `libonnxruntime-dev` | **1.21.0** | Yes | No (needs 1.23) |
| dnf (Fedora 44) | `onnxruntime-devel` | **1.22.2** | Yes | No (needs 1.23) |
| pkg (FreeBSD 14.5) | `onnxruntime` | **1.30.0** | Yes | Yes / no CoreML |
| Homebrew (macOS arm64) | `onnxruntime` | **1.32.0** | Yes | Yes |
| NuGet (`Microsoft.ML.OnnxRuntime`) | flat-container package | 1.18.0 – 1.32.0 | Yes | 1.23.2+ only |

These can be used as the system source (`./configure --with-onnxruntime`); no
NuGet download is needed. apt's 1.21 and dnf's 1.22 are fully usable for the
core API (inference, IoBinding, async, batch), but `appendEpDevice`,
`DeviceTensor`, Plugin EP registration and CoreML are unavailable there and
throw a clear "requires ONNX Runtime 1.23" error at runtime. Debian sid / Ubuntu
`resolute` ship 1.23.2 and Fedora Rawhide 1.26.0, which unlock the full API.
Homebrew 1.32.0 includes CoreML and the full EP surface on macOS arm64.

The CoreML EP itself exists in older macOS dylibs (1.20 reports
`CoreMLExecutionProvider`), but those SDKs ship no CoreML appender in
`onnxruntime_c_api.h`, so the extension only exposes CoreML from an ORT 1.23+
SDK. `Runtime::supports('coreml')` reports this.

`config.m4` validates that `onnxruntime_cxx_api.h` and `libonnxruntime.*` exist
and refuses `ORT_API_VERSION < 18`. Capability code is guarded by
`ORT_API_VERSION`, not by a separate flavor: 1.23+ enables `OrtEpDevice`,
`OrtDeviceMemoryType`, Plugin EP registration and `CoreML`; older SDKs compile
the core code paths and expose `Runtime::supports('ep_device'|'device_tensor'|
'coreml'|'async')` so applications can branch instead of catching exceptions.

## Build

Standard build (NuGet CPU SDK):

```bash
phpize
./configure
make -j8
```

CUDA flavor:

```bash
./configure --with-cuda13
make -j8
```

Output: `modules/onnxruntime.so`

When switching between `--with-onnxruntime`, `--with-cuda12` and
`--with-cuda13`, run `make clean` after `./configure`: the generated config.h
changes `ORT_API_VERSION`, and stale object files would otherwise link against
the wrong API table. A mismatched module now aborts at startup with an explicit
API-version error instead of crashing.

### Windows native build

Windows uses the PHP SDK build system, not `phpize`/`configure` from the Unix
tree. The repository therefore ships `config.w32`.

Prerequisites:

- PHP 8.5/8.6 x64 **ZTS** runtime, plus the matching PHP SDK build environment
- Visual Studio C++ toolset and Windows SDK. PHP's Windows build scripts are
  normally validated with the VS 2022 toolchain; a VS 2026-only installation
  must still provide a compatible MSVC/Windows SDK setup.
- ONNX Runtime **Windows x64** SDK containing `include\onnxruntime_cxx_api.h`
  and `lib\onnxruntime.lib`; 1.18+ compiles, 1.23+ enables the EP-device API

The Windows SDK root is passed as the same configure option. `config.w32`
itself does not download NuGet: if you start from a `.nupkg`, copy
`buildTransitive/native/include/*.h` (or `build/native/include/*.h`) into
`include\` and `runtimes/win-x64/native/*` into `lib\` first. The release
workflows do this automatically through
`.github/actions/fetch-onnxruntime`, assembling `ort\sdk\include` and
`ort\sdk\lib` from `Microsoft.ML.OnnxRuntime` (CPU) or
`Microsoft.ML.OnnxRuntime.Gpu.Windows` (CUDA/TensorRT; 1.26.0 = CUDA 12 +
TensorRT 10, 1.32.0 = CUDA 13 + TensorRT 10).

`config.w32` checks those paths and links only `onnxruntime.lib`. GPU provider
DLLs (`onnxruntime_providers_cuda.dll`, `onnxruntime_providers_shared.dll`,
etc.) are runtime dependencies and must be deployed next to the ORT DLL; they
are not linked into the PHP extension.

Windows GPU prerequisites:

- `onnxruntime.dll`, `onnxruntime_providers_shared.dll`,
  `onnxruntime_providers_cuda.dll` and `onnxruntime_providers_tensorrt.dll`
  next to `php.exe`.
- **Keep them next to `php.exe`, not only on `PATH`.** Windows 10/11 ships its
  own `onnxruntime.dll` (ORT 1.17) in `C:\Windows\System32`, and `System32`
  comes before `PATH` in the DLL search order. When the runtime sits only on
  `PATH`, the extension picks up the system copy and fails with
  `The requested API version [28] is not available ... Current ORT Version is: 1.17.1`.
  Placing the 1.28.2 runtime in the `php.exe` directory wins because the
  application directory is searched first.
- CUDA 13: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` (CUDA
  toolkit `bin\x64`) and `cudnn64_9.dll` (cuDNN 9) on `PATH`. The CUDA
  provider refuses to load with `Error loading ... which depends on
  "cublasLt64_13.dll" which is missing`; the TensorRT provider additionally
  needs cuDNN.
- TensorRT 10.16 (`TensorRT-10.16.1.11`): its `bin` directory
  (`nvinfer_10.dll`, `nvonnxparser_10.dll`, plugin and builder-resource DLLs)
  on `PATH`, and the TensorRT **CUDA EP added after it** — the TensorRT EP
  falls back to CUDA, so append TensorRT first and CUDA second.

Windows verification against PHP 8.5.12 ZTS / NTS / ORT 1.28.2:
35/36 PHPT pass, 1 CoreML platform skip; `034`/`035` exercise both the CPU
path and, when a CUDA `EpDevice` is available, a CUDA async-binding branch
(`cuda=pass`, `cuda4=pass`); the Plugin EP path
(`registerExecutionProviderLibrary()` -> `getEpDevices()` ->
`appendEpDevice()`) registers the CUDA provider DLL and runs inference on an
NVIDIA GPU device. ORT's generic name-based appender deliberately rejects
CUDA and TensorRT, so `appendExecutionProvider('CUDA')` and
`appendExecutionProvider('TensorRT')` are routed to
`SessionOptionsAppendExecutionProvider_CUDA_V2` / `_TensorRT_V2` with the
options map applied through the matching `Update...ProviderOptionsWithValue`;
`036` guards that routing. On an RTX 3050 with
MatMul 512x2048x2048, ZTS ran 12.51 ms/run on CPU, 3.20 ms/run on CUDA and
2.95 ms/run on TensorRT (fp16, 4.3 s engine build, cached); NTS ran 11.25 /
3.24 / 2.96 ms/run. These are smoke numbers, not tuned baselines.

Notes on the Windows build: the extension must be linked with the same MSVC
toolset as the PHP core (`14.44.35207` here). Building with the newer
`14.52` toolset produces `Can't load module ... as it's linked with 14.52, but
the core is linked with 14.44`, so pass `-vcvars_ver=14.44.35207` to
`VsDevCmd.bat`. `VsDevCmd.bat`, not `vcvars64.bat`, is also what puts
`mc.exe`/`mt.exe` from the Windows SDK on `PATH`; without it `configure`
fails with `ERROR: mc is required`. NTS builds need the **NTS** devel pack
(`php-devel-pack-8.5.12-1-nts-Win32-vs17-x64.zip`) — the ZTS one cannot be
reused.

Linux verification against PHP 8.5.12 NTS and ZTS / ORT 1.28.2
`linux-x64-gpu_cuda13`: 33/35 PHPT pass with the same two platform skips.
CUDA is a **built-in** EP there — `getAvailableProviders()` lists
TensorRT/CUDA/CPU while `getEpDevices()` exposes only the CPU device, so
`appendEpDevice()` cannot reach CUDA and
`appendExecutionProvider('CUDA', ['device_id' => '0'])` is the entry point.
On WSL2 with an RTX 3050, a MatMul 512x2048x2048 ran 12.14 ms/run on CPU
(354 GFLOPS) versus 2.75 ms/run on CUDA (1564 GFLOPS) for ZTS, and
12.34 vs 2.99 ms for NTS. WSL2 has no `/sys/class/drm`, so the GPU package
logs a `device_discovery.cc` warning on startup; PHPT must be run through a
wrapper that filters it, otherwise every test fails on output mismatch.

The build now consumes ORT from NuGet by default and compiles against whichever
version is supplied. Verified builds: **1.18.0**, **1.20.0**, **1.21.0** and
**1.23.2** (macOS arm64 CPU, core API; EP APIs disabled below 1.23), **1.26.0**
(linux-x64 CUDA 12 package), **1.28.2** (explicit macOS SDK), and **1.32.0**
(NuGet CPU on macOS arm64 and linux-x64, NuGet CUDA 13 on linux-x64).
ORT 1.32 changes the CUDA/cuDNN requirements (CUDA 13 + cuDNN 9); record GPU
results per OS/ORT baseline.

`config.m4` link policy:

- macOS/BSD links `-lc++`; Linux links `-lstdc++` by default and switches to `-lc++` when `-stdlib=libc++` is explicitly present.
- C++17 is required, with `ZEND_ENABLE_STATIC_TSRMLS_CACHE=1`.
- NuGet builds embed an rpath to the downloaded SDK (`${XDG_CACHE_HOME:-$HOME/.cache}/ext-onnxruntime/nuget/<package>/lib`), so no `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` is needed. Explicit-SDK and system builds write no rpath by default; set `ONNXRUNTIME_RPATH` before `./configure` when a fixed runtime search path is required, otherwise use `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` as described below.

### Install and PHP configuration

```bash
sudo make install
```

Confirm that the module landed in the current PHP extension_dir:

```bash
php-config --extension-dir
php -d extension=onnxruntime.so -m | grep onnxruntime
```

Add it to `php.ini`:

```ini
extension=onnxruntime.so
```

Or load it for a single command:

```bash
php -d extension=onnxruntime.so --ri onnxruntime
```

`php --ri onnxruntime` should report extension version `0.4.12`. Use `php -n -d extension=...` to exclude unrelated ini settings while debugging.

### PECL

Install the extension with PECL:

```bash
pecl install onnxruntime
```

### PIE (Composer)

`composer.json` declares the package as `type: php-ext` with extension name
`onnxruntime`, ZTS/NTS support, and the optional `--with-onnxruntime` configure
option. After the repository is submitted to Packagist, PIE can install it
directly:

```bash
pie install zydbkqf/ext-onnxruntime              # NuGet CPU package
pie install zydbkqf/ext-onnxruntime --with-cuda13
pie install zydbkqf/ext-onnxruntime --with-onnxruntime=/opt/ort
```

The ORT SDK is not bundled. By default PIE downloads the NuGet CPU package;
pass `--with-onnxruntime=DIR` to use an existing SDK. Windows PIE does not
compile extensions on the fly:
it requires pre-built ZIP assets on the GitHub release for the tag. The
`release-windows` workflow builds one ZIP per PHP version and ZTS/NTS mode,
adds `onnxruntime.dll` to each ZIP, and uploads them to the release. The asset
name follows the PIE Windows convention, for example
`php_onnxruntime-0.4.9-8.5-nts-vs17-x86_64.zip`.

The default PIE asset is the CPU build. CUDA 12 and CUDA 13 builds are
published as additional assets with a variant suffix, because PIE has no
variant field in its Windows asset name:

```text
php_onnxruntime-0.4.9-8.5-nts-vs17-x86_64-cuda12.zip
php_onnxruntime-0.4.9-8.5-nts-vs17-x86_64-cuda13.zip
```

`pie install` always selects the CPU asset. If you need CUDA, download the
matching `-cuda12` or `-cuda13` ZIP from the GitHub release, extract it, and
copy `php_onnxruntime.dll` and the runtime DLLs (`onnxruntime.dll`,
`onnxruntime_providers_cuda.dll`, `onnxruntime_providers_shared.dll`, and any
other DLLs in the ZIP) into your PHP `extension_dir`. The CUDA build still
requires a matching NVIDIA driver plus the CUDA Toolkit and cuDNN runtime;
those are not bundled in the PIE ZIP.

Packagist reports `dev-main` until a version tag exists. Tag `0.4.9` (without a
`v` prefix), create the GitHub Release for that tag, and let the workflow
publish the Windows assets before advertising Windows PIE support.

### macOS signing

After every `make` on macOS, re-sign the module ad-hoc; otherwise the system may terminate it with `SIGKILL (Code Signature Invalid)`:

```bash
codesign --force --sign - modules/onnxruntime.so
```

## Usage

```bash
php -d extension=$PWD/modules/onnxruntime.so -r '
$t = OnnxRuntime\Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
var_dump($t->dtype());   // string(7) "float64"
var_dump($t->shape());   // [2, 3]
var_dump($t->toArray()); // PHP array with the same structure as the input

$o = $t->toOrtValue();
$copy = $o->toTensor();  // deep copy; OrtValue may be destroyed first
var_dump($copy->toArray());

// synchronous inference
$s = new OnnxRuntime\Session('/path/to/model.onnx');
var_dump($s->inputNames(), $s->inputShapes());
$out = $s->run(['X' => $t]);
var_dump($out['Y']->toArray());

// runtime configuration and version/EP queries
$options = new OnnxRuntime\SessionOptions();
$options->setIntraOpNumThreads(2);
$options->setGraphOptimizationLevel('all');
$options->appendExecutionProvider('CoreML'); // macOS; no EP is appended by default
$s2 = new OnnxRuntime\Session('/path/to/model.onnx', $options);
$options->setIntraOpNumThreads(4); // does not affect $s2

$runtime = OnnxRuntime\Runtime::default();
echo $runtime->getVersion();
var_dump($runtime->getAvailableProviders());
var_dump($runtime->getEpDevices());
'
```

### API

| Method | Description |
| --- | --- |
| `OnnxRuntime\Tensor::fromArray(mixed $value, ?string $dtype = null): Tensor` | Convert a PHP scalar or nested array to a Tensor; an explicit `$dtype` overrides inference |
| `dtype(): string` | ONNX type name: `float32` / `float64` / `int64` / `int32` / `int16` / `int8` / `uint8` / `bool` / `string` |
| `shape(): array` | Inferred shape (`int[]`) |
| `rank(): int` | Number of dimensions; a PHP scalar has rank 0 |
| `toArray(): mixed` | Rebuild a PHP array from the shape (rank 0 returns a scalar) |
| `toOrtValue(): OnnxRuntime\OrtValue` | Create an ORT tensor handle; the numeric path borrows the source Tensor buffer, while strings are copied by ORT |

The public Tensor API is `fromArray` / `dtype` / `shape` / `rank` / `toArray` / `toOrtValue`, plus `OrtValue::toTensor`. `dataTypeCode()`, `dataSize()`, `elementCount()`, and `ownership()` were removed after an earlier API review: the first two are internal debugging aids, and the last two expose implementation details or no longer have meaningful semantics.

`dtype` names are case-insensitive and accept the aliases `float` -> `float32` and `double` -> `float64`.

### OrtValue

| Method | Description |
| --- | --- |
| `toTensor(): OnnxRuntime\Tensor` | Deep-copy into an independent Tensor; no dependency on the OrtValue lifetime afterward |
| `__debugInfo()` | Returns only `dtype` / `rank` / `shape` |

`OnnxRuntime\Tensor` is final, supports deep-copy `clone`, and disallows dynamic properties. `OnnxRuntime\OrtValue` cannot be constructed with `new` and cannot be cloned; it is created only by `Tensor::toOrtValue()`.

### EpDevice

`OnnxRuntime\EpDevice` is an immutable snapshot returned by `Runtime::getEpDevices()`. It is final, cannot be constructed directly, and cannot be cloned. It stores EP/device identity and metadata, never a raw `OrtEpDevice*`; the pointer is resolved again from the process Env when the Session is created.

| Method | Description |
| --- | --- |
| `epName(): string` | ORT execution-provider name, for example `CPUExecutionProvider` |
| `epVendor(): string` | Execution-provider vendor |
| `deviceType(): string` | `cpu` / `gpu` / `npu` |
| `vendorId(): int` / `deviceId(): int` | Hardware vendor/device IDs |
| `deviceVendor(): string` | Hardware vendor string |
| `metadata(): array` / `options(): array` | EP metadata and EP options |
| `registrationName(): ?string` | Plugin registration name, or `null` for a built-in device |
| `isPlugin(): bool` | Whether this device belongs to a registered Plugin EP |
| `toArray(): array` / `__debugInfo()` | Snapshot as an array |

### Session

| Method | Description |
| --- | --- |
| `__construct(string $modelPath, ?OnnxRuntime\SessionOptions $options = null)` | Load a local ONNX model; failures throw `OnnxRuntime\Exception` |
| `inputNames(): string[]` / `outputNames(): string[]` | Model input/output names, in model order |
| `inputTypes(): array` / `outputTypes(): array` | Name -> ONNX dtype (for example `float32`) |
| `inputShapes(): array` / `outputShapes(): array` | Name -> shape; dynamic dimensions are `null`, never `0` |
| `run(array $inputs): array` | Run synchronous inference with `OnnxRuntime\Tensor` values keyed by input name; returns Tensors keyed by output name |
| `runBatch(array $batchInputs): array` | Run a list of independent, complete input maps sequentially; returns a list of output maps in the same order |
| `runAsync(array $inputs): OnnxRuntime\Task` | Validate and deep-copy inputs on the PHP thread, submit native async inference, and return a Task |
| `createIoBinding(): OnnxRuntime\IoBinding` | Create a CPU I/O Binding owned by this Session |
| `runWithBinding(OnnxRuntime\IoBinding $binding): array` | Run synchronously through the binding; rejects a binding from a different Session |
| `runWithBindingAsync(OnnxRuntime\IoBinding $binding): OnnxRuntime\Task` | Hand binding ownership to a native Task; the binding becomes single-use and outputs must be read with `Task::getResult()` |

`OnnxRuntime\Session` is final and cannot be cloned. `runBatch()` accepts a list (sequential integer keys starting at 0) of complete input maps; it executes each map as a separate `run()` and does not merge different groups into one model batch. Input/output names must match the ONNX model. Missing, unknown, non-Tensor, dtype-mismatched, or shape-mismatched inputs are rejected. Only non-numeric string names are currently supported: ONNX itself permits names such as `"0"`, but PHP converts numeric-string array keys to integers, and `run()` rejects numeric keys, so these models are not supported yet. Session I/O is Tensor-only; Sequence/Map/Optional values and dtypes that the current Tensor cannot represent (for example `float16`) fail at load time with `OnnxRuntime\ConversionException` instead of being reported as supported. Returned Tensors are deep copies and remain valid after the Session is destroyed.

The second constructor argument is optional. `null` uses ORT defaults, and the Session copies the `SessionOptions` snapshot at creation time. Mutating the options object later does not affect an existing Session.

Async inference uses the same input map as `run()`, but inputs are deep-copied into native Task storage before submission. The worker never touches PHP zvals, Tensor objects, or request-owned buffers. A Task holds `shared_ptr<SessionState>`, so destroying the PHP Session does not invalidate a queued or running task.

### IoBinding

| Method | Description |
| --- | --- |
| `bindInput(string $name, OnnxRuntime\Tensor $tensor): void` | Validate and deep-copy one model input; the PHP Tensor may be destroyed immediately afterward |
| `bindOutput(string $name): void` | Bind a model output to CPU allocation; every model output must be bound |
| `bindOutputBuffer(string $name, OnnxRuntime\Tensor $buffer): void` | Bind a fixed-shape output to a private preallocated numeric buffer; rejects dynamic shapes, dtype/shape mismatch and string buffers |
| `getOutputs(): array` | Return copies of the latest run's outputs; throws before the first run |

`OnnxRuntime\IoBinding` is final, cannot be constructed directly, and cannot be cloned. A binding holds `shared_ptr<SessionState>`, so it can outlive the PHP Session object; `getOutputs()` remains valid after the Session is destroyed. One binding is a single-owner mutable object and is not safe to share between PHP threads. A binding does not preallocate output buffers by default; every run lets ORT allocate fresh CPU outputs, which keeps dynamic output shapes correct but is not zero-copy. `bindOutputBuffer()` keeps a private copy of the buffer template and reuses it across runs. Dynamic outputs are rejected instead of silently reusing a stale shape. `bindOutput()` and `bindOutputBuffer()` are mutually exclusive per output and can be switched before a run. After `runWithBindingAsync()`, the binding is owned by the Task: `bindInput()`, `bindOutput()`, `bindOutputBuffer()`, `runWithBinding()`, and `getOutputs()` are all rejected; create a new binding for another run.

### Task

`OnnxRuntime\Task` is final, cannot be constructed directly, and cannot be cloned.

| Method | Description |
| --- | --- |
| `getState(): string` | `queued`, `running`, `completed`, `failed`, or `cancelled` |
| `isReady(): bool` | True for completed, failed, or cancelled |
| `wait(?float $timeout = null): bool` | Block until ready; `null` waits forever; timeout returns `false` without changing state |
| `getResult(): array` | Return output Tensors for a completed task; throws for failed or non-completed states; never waits implicitly |
| `getError(): ?Throwable` | Return the native failure object, or `null` when the task did not fail |
| `cancel(): bool` | Cancel only a `queued` task; a running ORT call is not interrupted |

Use `runAsync()` / `runWithBindingAsync()` only when the PHP process can wait for or poll the Task. In FPM this still occupies the worker while `wait()` blocks; the native pool does not create a PHP scheduler or Fiber adapter. A binding Task keeps the binding and all bound input/output buffers alive until the Task is destroyed. ORT `SynchronizeInputs()` runs before `Run()` and `SynchronizeOutputs()` runs before output copying; Task completion therefore means the bound transfer path has completed, not that a vendor stream was exposed as an independent async object.

### SessionOptions

| Method | Description |
| --- | --- |
| `setIntraOpNumThreads(int $threads)` | Set the intra-op thread count; `0` keeps the ORT default; negative values or values above the int limit are rejected |
| `setInterOpNumThreads(int $threads)` | Set the inter-op thread count; `0` keeps the ORT default |
| `setGraphOptimizationLevel(string $level)` | `disabled` / `basic` / `extended` / `all`, case-insensitive; default `all` |
| `setExecutionMode(string $mode)` | `sequential` / `parallel`; default `sequential` |
| `setLogSeverityLevel(int|string $level)` | `0..4` or `verbose` / `info` / `warning` / `error` / `fatal`; default `2` / `warning` |
| `setEnableCpuMemArena(bool $enable)` | Default `true` |
| `setEnableMemPattern(bool $enable)` | Default `true` |
| `appendExecutionProvider(string $name, array $options = [])` | Built-in EP path: append a provider such as `CoreML`, `CUDA` or `TensorRT`; on ORT 1.23+, a name that matches an `Runtime::getEpDevices()` entry (case-insensitive, `ExecutionProvider`/`EP` suffix optional) is routed through the V2 device API instead; options are string -> string |
| `appendEpDevice(OnnxRuntime\EpDevice $device, array $options = [])` | Device path: append a device from `Runtime::getEpDevices()` using `SessionOptionsAppendExecutionProvider_V2`; options are string -> string |
| `toArray(): array` | Return the current configuration snapshot, including `intraOpNumThreads`, `graphOptimizationLevel`, `executionProviders`, and `epDevices` |

`SessionOptions` is final, supports deep-copy clone, and disallows dynamic properties. Configuration applies only to Sessions created afterward; existing Sessions are unaffected by later setters. Built-in and device entries keep their append order. If neither method is called, no EP is appended and ORT defaults are preserved. On ORT 1.23+, `appendExecutionProvider()` falls back to the first matching `EpDevice` when ORT's case-sensitive name appender would reject the name; this makes `appendExecutionProvider('cpu')` and vendor short names work, and keeps plugin devices lease-protected. Provider names with multiple devices should use `appendEpDevice()` to select one explicitly. CoreML is available on macOS, but the extension does not select it automatically. `appendEpDevice()` rejects an object that is not an `EpDevice` and fails early if the device is no longer present in the Env. `toArray()` reports built-in entries under `executionProviders` and device entries under `epDevices` so the two mechanisms stay distinguishable; the fallback is applied to a Session's private copy and does not rewrite the user's `SessionOptions` snapshot.

### Runtime

| Method | Description |
| --- | --- |
| `Runtime::default(): Runtime` | Return a Runtime object; direct `new` and `clone` are forbidden |
| `getVersion(): string` | Return the ONNX Runtime version string |
| `getBuildInfo(): string` | Return ONNX Runtime build information |
| `getAvailableProviders(): string[]` | Return available EP names, such as `CPUExecutionProvider` |
| `supports(string $feature): bool` | Capability probe for the SDK this module was compiled against: `ep_device`, `device_tensor`, `coreml`, `async`; returns `false` when the ORT version is too old |
| `getEpDevices(): OnnxRuntime\EpDevice[]` | Return immutable device snapshots; use them with `SessionOptions::appendEpDevice()`; requires ORT 1.23+ |
| `registerExecutionProviderLibrary(string $name, string $path): void` | Register a Plugin EP library; empty names/paths, NUL bytes, and duplicate registrations are rejected |
| `unregisterExecutionProviderLibrary(string $name): void` | Unregister a Plugin EP library; an unknown name is rejected |
| `setAsyncThreads(int $threads): void` | Configure native async worker count (1..64); must be before the first `runAsync()` |
| `setAsyncQueueCapacity(int $capacity): void` | Configure bounded async queue capacity (1..1000000); immutable after engine start |
| `asyncInfo(): array` | Return `workers`, `queueCapacity`, `queued`, `running`, and `started` |

Plugin libraries are tracked by a process-level registry and live until module `MSHUTDOWN`; they are not unloaded when a PHP `Runtime` object is destroyed. The registry records each registration's generation, device keys, and active lease count under a mutex. `EpDevice` snapshots carry the registration generation, so an old snapshot cannot bind to a newly registered library with the same name; duplicate devices from one plugin are separated by an instance ordinal inside the current Env, and the registration name is part of the device key so two plugins with the same EP name do not collide. The native owner is `SessionState`, which holds the Env reference, plugin lease, and `Ort::Session` in destruction-safe order. A Session reserves the lease before `Ort::Session` creation, releases it immediately if construction fails, and releases it only after the native `Ort::Session` is destroyed; a future Task can retain `shared_ptr<SessionState>` independently of the PHP object. `unregisterExecutionProviderLibrary()` is rejected while any lease remains, and unload happens only at zero. At `MSHUTDOWN`, new Sessions/registrations are disabled first, then only zero-lease registrations are unloaded. The Windows CUDA Plugin EP smoke path has passed (registration, device enumeration, `appendEpDevice()`, inference); general Plugin EP acceptance across providers, platforms, unload sequences and failure modes is still not claimed.

### Type inference rules (when dtype is omitted)

| PHP element type | Inferred dtype |
| --- | --- |
| int only | `int64` |
| float only | `float64` |
| int + float mixed | `float64` |
| bool only | `bool` |
| string only | `string` |
| empty array (no elements to inspect) | `float32` (the ONNX empty-tensor default; specify another dtype explicitly if needed) |
| bool mixed with numeric values | Error; explicit dtype required |
| string mixed with numeric/bool values | Error; explicit dtype required |

Explicit dtype rules:

- Float -> integer: the value must be finite and exactly integral (`1.5` is rejected, never truncated silently), followed by a target-range check.
- Integer -> target integer: check the target range element by element (for example `[-128, 127]` for `int8`, `[0, 255]` for `uint8`).
- double -> `float32`: reject a finite input that overflows to `inf` after conversion to `float`; normal precision loss is allowed.
- Numeric -> `bool`: accept only `0/1` (or `0.0/1.0`); other values are rejected.
- No implicit conversion between strings and numeric values.

### Precision policy

- **Integer tensors:** precision loss is forbidden. Floating-point values must be exactly integral and must fall within the target integer range.
- **Floating-point tensors:** precision loss is allowed (common in ML workloads), but a finite input must not silently overflow to `inf`.
  - Example: `fromArray([16777217], 'float32')` -> `16777216.0`, no error.
  - Example: `fromArray([9007199254740993], 'float64')` -> `9007199254740992.0`, no error.
  - Example: `fromArray([1e39], 'float32')` still throws `ConversionException`.
- A future strict mode may add precision-loss checks; the current mode does not reject implicit precision loss.

### Error model

| Error source | PHP exception |
| --- | --- |
| PHP-level argument type error (for example dtype is not a string) | `TypeError` (engine) |
| Unsupported dtype name | `ValueError` |
| Array/element/range/shape error | `OnnxRuntime\ConversionException` (extends `OnnxRuntime\Exception` -> `\RuntimeException`) |
| Error returned by the ORT C/C++ API | `OnnxRuntime\Exception` (wraps `Ort::Exception`) |
| Model path, corrupt model, ORT load or run error | `OnnxRuntime\Exception` (from `Ort::Session` / `Run`) |
| Input/output dtype, rank, or static shape mismatch | `OnnxRuntime\ConversionException` (checked before entering ORT) |

The extension maps real ORT runtime errors. `ConversionException` means PHP-side input or conversion failed; `Exception` means ORT rejected the operation.

### Conversion constraints

- Accept only **lists** (contiguous integer keys `0..n-1`); associative arrays and sparse indexes are rejected.
- Ragged nested arrays are rejected, not padded.
- Mixing nested arrays and scalars is rejected.
- Empty dimensions are valid (for example `[[], []]` -> shape `[2, 0]`).
- Maximum nesting depth is 64, preventing malicious input from exhausting the C stack.
- Element errors include a path (for example `value[1][2]: ...`).
- The per-dimension limit for `toArray()` is `min(UINT32_MAX, ZEND_LONG_MAX)`. There is no separate limit on the total number of structural nodes. A zero-element Tensor such as shape `{100000000, 0}` still creates a very large number of empty subarrays even though its element count is zero; memory exhaustion is left to the PHP allocator. If structural cost needs a tighter bound, add a node budget as a separate policy rather than weakening Tensor shape validation.

## Boundaries with ONNX Runtime

- zvals are read only during `fromArray()` and copied immediately; `Tensor` / `OrtValue` never retain a PHP array pointer.
- `Tensor` always owns its `data` / `strings`. The numeric `Tensor -> OrtValue` path borrows the buffer without copying; `OrtValue` holds a reference to the source Tensor so the buffer remains valid for the OrtValue lifetime.
- String Tensor -> OrtValue is allocated and copied by ORT; the OrtValue does not depend on the source Tensor.
- `OrtValue -> Tensor` always deep-copies, so the returned Tensor does not depend on the OrtValue.
- Strings are written through ORT's `GetResizedStringTensorElementBuffer()` with exact byte lengths, supporting embedded NUL, empty strings, and consecutive empty strings. After writing, each element length is read back with `GetStringTensorElementLength()`. Reading requires the first element offset to be 0 and validates every `begin <= end <= total`. Re-run `012-ortvalue-string.phpt` whenever ORT is upgraded.

## Tests and build smoke

Full PHPT suite:

```bash
NO_INTERACTION=1 make test
```

macOS: 35/38 passed, 3 platform skips (Windows CUDA `033`, no CUDA/TensorRT `036`, OpenVINO plugin `038`). Windows PHP 8.5.12 ZTS: 34/36 passed, 2 platform skips (CoreML `025`, OpenVINO plugin `038`), including `034`/`035` on both CPU and CUDA. FreeBSD 14.5 PHP 8.5.12 NTS with ORT 1.30.0: 34/38 passed, 4 platform skips (CoreML `025`, Windows CUDA `033`, no CUDA/TensorRT `036`, OpenVINO plugin `038`); the `cpuid_info` line requires the filter documented in `docs/FREEBSD.md`. Linux x86_64 PHP 8.5.12 ZTS (Docker `php:8.5-zts`): 33/36 passed, 3 platform skips (CoreML `025`, Windows CUDA `033`, OpenVINO plugin `038`), including `034`/`035`; `tests/run-async-sanitizers.sh` passed with ASan+LSan and `native-async-stress-ok`. Linux x86_64 PHP 8.5.12 NTS (Docker `php:8.5-cli`): 33/36 passed, the same 3 platform skips, built with `--with-onnxruntime=/opt/ort`. CPU I/O Binding ASan+LSan also passed; Fiber, 8 coroutines, Swoole process, 4 Swoole threads, `pcntl_fork`, and 4 `parallel` runtimes passed. CPU I/O Binding has a native ASan stress gate:

```bash
bash tests/run-io-binding-sanitizers.sh
```

It exercises 150 identity runs with a preallocated output buffer, 30 dynamic-shape runs, input copies, output ownership and binding lifetime past SessionState destruction; macOS reports `native-io-binding-stress-ok` under ASan (LSan unavailable on Darwin). Coverage includes 1D/2D/3D conversion, empty arrays, empty inner dimensions, scalars and rank 0, mixed numeric types, bool/string, explicit dtype, overflow and precision, associative/sparse arrays, ragged arrays, nested scalars, recursion depth, empty/zero-dimensional tensors, int64/int32/int16/int8/uint8 boundaries, float32/float64 precision loss and overflow, large arrays (500x500), object API/clone/constructor protection, OrtValue round trips for every dtype, strings/NUL/empty strings, lifetime after the source Tensor is destroyed, Session metadata/dynamic dimensions, single and multi-input/output inference, model paths and corrupt models, missing/unknown/wrong dtype or shape inputs, repeated inference, outputs remaining valid after Session destruction, zero-element Tensor (`[0]` / `[2,0]`) inference, explicit rejection of non-Tensor and unsupported dtypes, input dtype/rank/static-shape contracts, `run()` reflection parameter count, `SessionOptions` defaults/setters/invalid values/clone/snapshot behavior, `Runtime` version and Provider/EP device queries, built-in `appendExecutionProvider` configuration, CoreML inference, `EpDevice` immutability, the V2 device append path validated with an ORT-provided built-in CPU device, native async Task submission/result/lifetime, CPU I/O Binding input/output binding, repeated runs, dynamic output shapes, zero-element output, binding/session mismatch, binding lifetime after Session destruction, and fixed-shape preallocated output with explicit dynamic-output rejection, and explicit multi-group batch inference with per-entry error context.

Compile/load/basic Tensor/inference smoke:

```bash
bash tests/build-smoke.sh                  # NuGet CPU package
bash tests/build-smoke.sh --with-cuda13   # NuGet CUDA 13 package
bash tests/build-smoke.sh /opt/ort-sdk    # explicit SDK root
```

With no directory the script lets `configure` download the NuGet package.

The script runs `phpize --clean`, `phpize`, `configure`, `make`, module loading, `Tensor::fromArray()`, `identity.onnx` inference, and the full PHPT suite. macOS adds an ad-hoc signature automatically.

Native memory gate:

```bash
bash tests/run-sanitizers.sh          # native Tensor ASan; Linux also enables LSan and appends Valgrind when available
bash tests/run-async-sanitizers.sh    # native AsyncEngine ASan/LSan (macOS + Linux verified)
```

macOS supports ASan only (LSan is unavailable); the leak gate lives on Linux ASan+LSan/Valgrind. The Linux x86_64 PHP 8.5.12 ZTS container has been verified with `native-async-stress-ok` and LSan enabled. The `008` `memory_get_usage()` check reflects only the Zend/PHP heap and cannot replace a native sanitizer. `tests/native_stress.cpp` also covers `elementCount()` for zero dimensions, scalars, negative dimensions, and multiplication overflow.

The GitHub Actions workflow is `.github/workflows/ci.yml`; it downloads the ORT SDK for Linux/macOS and calls the same smoke script. The workflow has not actually run on GitHub Actions yet, so it must not be used to claim that the full Linux/macOS matrix is supported.

## Benchmark

`benchmarks/fastembed_int8_bpe.py` compares FastEmbed against the PHP extension
on `Qdrant/bge-small-en-v1.5-onnx-Q` (BGE-small, int8 ONNX, WordPiece tokenizer).
It reports FastEmbed end-to-end, FastEmbed tokenizer + ORT, Python raw ORT,
PHP tokenization + ORT, and PHP ORT-only latency separately. The PHP
tokenization path is implemented in PHP (`benchmarks/php_wordpiece.php`), so it
does not call Python for tokenization. See `benchmarks/README.md` for the exact
workload and current CPU/CoreML/CUDA results.

## Directory Layout

```text
config.m4                  # phpize build configuration (external ORT SDK resolution and validation)
composer.json              # PIE/Packagist metadata (php-ext, ZTS/NTS, configure options)
package.xml                # PECL source-package metadata (no development SDK)
src/
  php_onnxruntime.h        # Module declaration and PHP version floor
  onnxruntime.cpp          # Zend bindings: classes, object lifecycle, methods
  onnx_tensor.h/.cpp       # Zend-independent Tensor representation and type utilities
  zval_converter.h/.cpp    # Two-pass scan: shape/type checks + flattened copy
  ort_value_converter.h/.cpp # Tensor <-> OrtValue
  ort_environment.h/.cpp   # Process-level Ort::Env (MINIT create / MSHUTDOWN destroy)
  session.h/.cpp           # shared SessionState ownership, metadata, synchronous Run
  session_options.h/.cpp   # Zend-independent SessionConfig snapshot and EP append
  async_engine.h/.cpp      # Bounded native worker pool and Task state
  io_binding.h/.cpp        # CPU I/O Binding state, input copies, output ownership
  ep_device.h/.cpp         # EpDevice identity snapshot and Env resolution
  runtime.h/.cpp           # Plugin EP registry, device ownership, and session leases
tests/
  build-smoke.sh           # Compile + load + Tensor + inference + PHPT
  run-sanitizers.sh        # Native Tensor ASan/LSan/Valgrind
  run-async-sanitizers.sh  # Native AsyncEngine ASan
  run-io-binding-sanitizers.sh # Native I/O Binding ASan/LSan
  *.phpt                   # 37 PHPT tests
  fixtures/                # Local ONNX models
benchmarks/
  fastembed_int8_bpe.py    # FastEmbed vs raw ORT vs PHP ORT/tokenize+ORT
  php_ort_bench.php        # PHP-side timing with pre-tokenized inputs
  php_tokenize_ort_bench.php # PHP tokenization + ORT timing
  php_wordpiece.php        # PHP WordPiece tokenizer for the benchmark
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

ONNX Runtime is a separate third-party dependency licensed under the MIT License (Copyright Microsoft Corporation). It is not bundled with this repository or the PECL source package. See [NOTICE](NOTICE) for the upstream project and license references.
