# PHP ONNX Runtime Extension

A native PHP extension that loads ONNX models and runs inference directly from PHP, using the Zend C API, C++, and the ONNX Runtime native library. It does not use FFI or Python.

Current status: synchronous inference, native async tasks, CPU I/O Binding, fixed-shape preallocated outputs, device tensors, explicit device copies, explicit multi-group batch inference, and async I/O Binding are implemented. The current feature set is verified on macOS arm64, Windows x86_64 ZTS, Linux x86_64 ZTS, and Linux x86_64 NTS (33/35 PHPT, with CoreML and Windows-CUDA platform skips). The Windows CUDA Plugin EP smoke path has passed: register the provider DLL, enumerate the CUDA `EpDevice`, append it with `appendEpDevice()`, and run inference on the NVIDIA GPU. This is a smoke result, not general Plugin EP acceptance across every provider, platform, unload sequence, and failure mode.

Execution providers have two distinct entry points:

- `appendExecutionProvider()` is the built-in EP path and calls ORT's provider-specific append API.
- `appendEpDevice()` is the device-based path. It resolves an `EpDevice` from `Runtime::getEpDevices()` and uses `SessionOptionsAppendExecutionProvider_V2`.

When no EP is appended, ORT defaults are preserved; this project does not promise "CPU only by default".

## Before You Use This

1. **This project is mainly for developers who do not want a Python runtime.** It is not automatically faster. Converting PHP zval data into a Tensor has real cost, and depending on payload size and call pattern it may not beat a separate Python process over IPC or HTTP. Benchmark the entire path, not only the ORT kernel time.

2. **Do not run long synchronous inference inside a shared PHP-FPM pool.** A request occupies its FPM worker until inference returns; enough concurrent requests can exhaust the pool and stall unrelated traffic. Prefer a dedicated CLI worker, a dedicated process, or an isolated service. If you use threads, use a ZTS build and isolate thread usage, and validate it. ZTS alone does not make an arbitrary application thread-safe.

3. **Inference is compute-bound.** The OS can still preempt a process, but the PHP request does not cooperatively yield, so the current worker is unavailable for the whole run. Long CPU inference raises request latency and reduces concurrency. Use worker isolation, request limits, and capacity planning instead of assuming PHP will "yield" during the computation.

4. **Most PHP server deployments have CPU only.** GPU/NPU access is provided through ONNX Runtime execution providers, and availability depends on the ORT build and platform. EP offload is opt-in and must be benchmarked before production: extra data transfers and unsupported graph partitions can make a small model slower on an EP than on CPU.

5. **INT8-quantized models are often a good starting point for CPU, but not a universal rule.** Quantization can reduce accuracy and may not help every model or backend. Avoid very large models, measure latency, memory, and output quality, and test on the actual target hardware before production.

## Support Status

| Platform | PHP | Thread mode | Build type | Status |
| --- | --- | --- | --- | --- |
| macOS arm64 | 8.5.6 | ZTS | Release | Verified: 34/35 PHPT, 1 Windows-CUDA skip (`033`), ASan, native sync/async stress |
| macOS arm64 | 8.x | NTS | Debug/Release | Not verified |
| Linux x86_64 | 8.5.12 | NTS | Release | Verified: 33/35 PHPT, 2 platform skips (CoreML, Windows CUDA); new `--with-onnxruntime=DIR` build path |
| Linux x86_64 | 8.5.12 | ZTS | Release | Verified: 33/35 PHPT, 2 platform skips (CoreML, Windows CUDA); async ASan+LSan passed; Swoole 6.2.1 Fiber/coroutine/process/thread, `pcntl_fork`, `parallel` |
| Linux arm64 | 8.x | ZTS/NTS | Debug/Release | Not verified |
| Linux x86_64 | 8.x | Any | Debug | Not verified |
| Windows x86_64 | 8.5.12 | ZTS | Release | Verified: 34/35 PHPT, 1 CoreML platform skip; `034`/`035` CPU+CUDA async binding; native I/O Binding stress; CUDA Plugin EP smoke |

The PHP code targets the Zend API in PHP 8.0–8.5 and rejects versions below 8.0 at compile time. Verified environments: macOS arm64 PHP 8.5.6 ZTS; Linux x86_64 PHP 8.5.12 NTS and ZTS (current feature set, Swoole 6.2.1 on ZTS); Windows x86_64 PHP 8.5.12 ZTS with VS 2026 + MSVC 14.44 and ORT 1.28.2, all Release builds. Debug builds, Linux arm64, macOS NTS, Windows NTS, and other unlisted combinations are not claimed as supported. `phpize`/`php-config` determine the ZTS/NTS and Debug/Release configuration; the resulting module must be installed into the extension_dir of the same PHP build and loaded by that same `php` binary.

## Requirements

- PHP 8.x; the currently verified version is PHP 8.5.6 CLI ZTS (Zend Module API `20250925`)
- `phpize` / `php-config` from the same PHP installation that will load the extension
- A C++17 compiler (Apple clang on macOS; g++/clang++ on Linux)
- An ONNX Runtime v1.28.2 (`ORT_API_VERSION 28`) **include + shared library** SDK

### Getting ONNX Runtime

Download the official prebuilt package for your platform and extract it:

```text
onnxruntime-<platform>-1.28.2/
  include/onnxruntime_cxx_api.h
  lib/libonnxruntime.*
```

The SDK root is passed to `configure`; `include/` and `lib/` are derived from
that directory. The extension **does not vendor the ONNX Runtime source tree**
and does not bake a platform-specific prebuilt library into the build scripts.
The `third_party/onnxruntime` directory in this repository is a local
development SDK, not a production dependency.

`config.m4` verifies that `onnxruntime_cxx_api.h` and `libonnxruntime.*` exist.
`--with-onnxruntime` is required; `./configure` fails instead of silently
selecting a bundled library.

## Build

Standard external-SDK build:

```bash
phpize
./configure --with-onnxruntime=/opt/onnxruntime-linux-x64-1.28.2
make -j8
```

For local development against the repository SDK:

```bash
phpize
./configure --with-onnxruntime="$PWD/third_party/onnxruntime"
make -j8
```

Output: `modules/onnxruntime.so`

### Windows native build

Windows uses the PHP SDK build system, not `phpize`/`configure` from the Unix
tree. The repository therefore ships `config.w32`.

Prerequisites:

- PHP 8.5 x64 **ZTS** runtime, plus the matching PHP SDK build environment
- Visual Studio C++ toolset and Windows SDK. PHP's Windows build scripts are
  normally validated with the VS 2022 toolchain; a VS 2026-only installation
  must still provide a compatible MSVC/Windows SDK setup.
- ONNX Runtime **Windows x64** SDK containing `include\onnxruntime_cxx_api.h`
  and `lib\onnxruntime.lib`

The Windows SDK root is passed as the same configure option:

```bat
configure --with-onnxruntime=E:\ort\onnxruntime-win-x64-1.28.2 --with-prefix=D:\php
```

`config.w32` checks those paths and links only `onnxruntime.lib`. GPU provider
DLLs (`onnxruntime_providers_cuda.dll`, `onnxruntime_providers_shared.dll`,
etc.) are runtime dependencies and must be deployed next to the ORT DLL; they
are not linked into the PHP extension.

Windows verification against PHP 8.5.12 ZTS / ORT 1.28.2:
34/35 PHPT pass, 1 CoreML platform skip; `034`/`035` exercise both the CPU
path and, when a CUDA `EpDevice` is available, a CUDA async-binding branch
(`cuda=pass`, `cuda4=pass`); the Plugin EP path
(`registerExecutionProviderLibrary()` -> `getEpDevices()` ->
`appendEpDevice()`) registers the CUDA provider DLL and runs inference on an
NVIDIA GPU device. CUDA is a Plugin EP in this ORT build and is not accepted
by `appendExecutionProvider('CUDA')`.

The repository's verified ORT baseline is **1.28.2**. ORT 1.32 can be used for
Windows GPU experiments, but its provider/CUDA/cuDNN requirements differ and
its results must be recorded as a separate OS/ORT baseline.

`config.m4` link policy:

- macOS/BSD links `-lc++`; Linux links `-lstdc++` by default and switches to `-lc++` when `-stdlib=libc++` is explicitly present.
- C++17 is required, with `ZEND_ENABLE_STATIC_TSRMLS_CACHE=1`.
- No ORT rpath is written by default. Set `ONNXRUNTIME_RPATH` before `./configure` when a fixed runtime search path is required; otherwise use `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` as described below.

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

`php --ri onnxruntime` should report extension version `0.4.0`. Use `php -n -d extension=...` to exclude unrelated ini settings while debugging.

### Runtime library loading

The extension needs `libonnxruntime` at runtime. If the ORT library is not in the system loader's default path, choose one of:

```bash
# macOS
export DYLD_LIBRARY_PATH=/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
```

Or write an rpath at build time:

```bash
ONNXRUNTIME_RPATH=/opt/onnxruntime/lib ./configure --with-onnxruntime=/opt/onnxruntime
```

An rpath is a fixed path inside the binary. Release packages must ensure that the target machine uses the same ORT location. For relocatable installs, prefer a system library path or runtime environment variables.

### PECL source package

Build the PECL source package:

```bash
pecl package package.xml
```

The output is `onnxruntime-0.4.7.tgz`. `package.xml` contains extension sources, build files, tests, and documentation, but not the `third_party/` development SDK. Builds from that package still need an external ONNX Runtime SDK as described above.

### PIE (Composer)

`composer.json` declares the package as `type: php-ext` with extension name
`onnxruntime`, ZTS/NTS support, and the required `--with-onnxruntime=DIR` configure option.
After the repository is submitted to Packagist, PIE can install it directly:

```bash
pie install zydbkqf/ext-onnxruntime --with-onnxruntime=/opt/ort
```

The ORT SDK is not bundled, so a source install needs the same external SDK as
a manual `phpize` build. The PIE package currently targets Linux and macOS
source builds. Windows PIE installation is not enabled until a prebuilt DLL ZIP
is published as a GitHub release asset; use the PECL Windows DLL package for now.

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
| `appendExecutionProvider(string $name, array $options = [])` | Built-in EP path: append a provider such as `CoreML`; options are string -> string |
| `appendEpDevice(OnnxRuntime\EpDevice $device, array $options = [])` | Device path: append a device from `Runtime::getEpDevices()` using `SessionOptionsAppendExecutionProvider_V2`; options are string -> string |
| `toArray(): array` | Return the current configuration snapshot, including `intraOpNumThreads`, `graphOptimizationLevel`, `executionProviders`, and `epDevices` |

`SessionOptions` is final, supports deep-copy clone, and disallows dynamic properties. Configuration applies only to Sessions created afterward; existing Sessions are unaffected by later setters. Built-in and device entries keep their append order. If neither method is called, no EP is appended and ORT defaults are preserved. CoreML is available on macOS, but the extension does not select it automatically. `appendEpDevice()` rejects an object that is not an `EpDevice` and fails early if the device is no longer present in the Env. `toArray()` reports built-in entries under `executionProviders` and device entries under `epDevices` so the two mechanisms stay distinguishable.

### Runtime

| Method | Description |
| --- | --- |
| `Runtime::default(): Runtime` | Return a Runtime object; direct `new` and `clone` are forbidden |
| `getVersion(): string` | Return the ONNX Runtime version string |
| `getBuildInfo(): string` | Return ONNX Runtime build information |
| `getAvailableProviders(): string[]` | Return available EP names, such as `CPUExecutionProvider` |
| `getEpDevices(): OnnxRuntime\EpDevice[]` | Return immutable device snapshots; use them with `SessionOptions::appendEpDevice()` |
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

macOS: 34/35 passed, 1 Windows-CUDA skip (`033`). Windows PHP 8.5.12 ZTS: 34/35 passed, 1 CoreML skip (`025`), including `034`/`035` on both CPU and CUDA. Linux x86_64 PHP 8.5.12 ZTS (Docker `php:8.5-zts`): 33/35 passed, 2 platform skips (CoreML `025`, Windows CUDA `033`), including `034`/`035`; `tests/run-async-sanitizers.sh` passed with ASan+LSan and `native-async-stress-ok`. Linux x86_64 PHP 8.5.12 NTS (Docker `php:8.5-cli`): 33/35 passed, the same 2 platform skips, built with `--with-onnxruntime=/opt/ort`. CPU I/O Binding ASan+LSan also passed; Fiber, 8 coroutines, Swoole process, 4 Swoole threads, `pcntl_fork`, and 4 `parallel` runtimes passed. CPU I/O Binding has a native ASan stress gate:

```bash
bash tests/run-io-binding-sanitizers.sh
```

It exercises 150 identity runs with a preallocated output buffer, 30 dynamic-shape runs, input copies, output ownership and binding lifetime past SessionState destruction; macOS reports `native-io-binding-stress-ok` under ASan (LSan unavailable on Darwin). Coverage includes 1D/2D/3D conversion, empty arrays, empty inner dimensions, scalars and rank 0, mixed numeric types, bool/string, explicit dtype, overflow and precision, associative/sparse arrays, ragged arrays, nested scalars, recursion depth, empty/zero-dimensional tensors, int64/int32/int16/int8/uint8 boundaries, float32/float64 precision loss and overflow, large arrays (500x500), object API/clone/constructor protection, OrtValue round trips for every dtype, strings/NUL/empty strings, lifetime after the source Tensor is destroyed, Session metadata/dynamic dimensions, single and multi-input/output inference, model paths and corrupt models, missing/unknown/wrong dtype or shape inputs, repeated inference, outputs remaining valid after Session destruction, zero-element Tensor (`[0]` / `[2,0]`) inference, explicit rejection of non-Tensor and unsupported dtypes, input dtype/rank/static-shape contracts, `run()` reflection parameter count, `SessionOptions` defaults/setters/invalid values/clone/snapshot behavior, `Runtime` version and Provider/EP device queries, built-in `appendExecutionProvider` configuration, CoreML inference, `EpDevice` immutability, the V2 device append path validated with an ORT-provided built-in CPU device, native async Task submission/result/lifetime, CPU I/O Binding input/output binding, repeated runs, dynamic output shapes, zero-element output, binding/session mismatch, binding lifetime after Session destruction, and fixed-shape preallocated output with explicit dynamic-output rejection, and explicit multi-group batch inference with per-entry error context.

Compile/load/basic Tensor/inference smoke:

```bash
bash tests/build-smoke.sh /opt/onnxruntime-linux-x64-1.28.2
```

Pass the extracted ONNX Runtime SDK root. For the repository development SDK
use `bash tests/build-smoke.sh "$PWD/third_party/onnxruntime"`.

The script runs `phpize --clean`, `phpize`, `configure --with-onnxruntime=DIR`, `make`, module loading, `Tensor::fromArray()`, `identity.onnx` inference, and the full PHPT suite. macOS adds an ad-hoc signature automatically.

Native memory gate:

```bash
bash tests/run-sanitizers.sh          # native Tensor ASan; Linux also enables LSan and appends Valgrind when available
bash tests/run-async-sanitizers.sh    # native AsyncEngine ASan/LSan (macOS + Linux verified)
```

macOS supports ASan only (LSan is unavailable); the leak gate lives on Linux ASan+LSan/Valgrind. The Linux x86_64 PHP 8.5.12 ZTS container has been verified with `native-async-stress-ok` and LSan enabled. The `008` `memory_get_usage()` check reflects only the Zend/PHP heap and cannot replace a native sanitizer. `tests/native_stress.cpp` also covers `elementCount()` for zero dimensions, scalars, negative dimensions, and multiplication overflow.

The GitHub Actions workflow is `.github/workflows/ci.yml`; it downloads the ORT SDK for Linux/macOS and calls the same smoke script. The workflow has not actually run on GitHub Actions yet, so it must not be used to claim that the full Linux/macOS matrix is supported.

## Benchmark

`benchmarks/fastembed_int8_bpe.py` compares FastEmbed against the PHP extension
on `Qdrant/bge-small-en-v1.5-onnx-Q` (BGE-small, int8 ONNX, BPE tokenizer).
It reports FastEmbed end-to-end, FastEmbed tokenizer + ORT, and PHP ORT-only
latency separately; the PHP extension has no BPE tokenizer, so PHP numbers are
inference-only. See `benchmarks/README.md` for the exact workload and current
CPU/CoreML results.

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
  *.phpt                   # 35 PHPT tests
  fixtures/                # Local ONNX models
benchmarks/
  fastembed_int8_bpe.py    # FastEmbed vs raw ORT vs PHP ORT
  php_ort_bench.php        # PHP-side timing with pre-tokenized inputs
third_party/onnxruntime/   # macOS arm64 development SDK only (not a production dependency)
docs/                      # design documents (DOCX)
```

Stable baseline: [BASELINE.md](docs/BASELINE.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

ONNX Runtime is a separate third-party dependency licensed under the MIT License (Copyright Microsoft Corporation). It is not bundled with this repository or the PECL source package. See [NOTICE](NOTICE) for the upstream project and license references.
