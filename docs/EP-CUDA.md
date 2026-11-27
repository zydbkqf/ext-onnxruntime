# CUDA and TensorRT (built-in provider and CUDA Plugin EP)

CUDA reaches ONNX Runtime through two different mechanisms, and both are
available here:

| Mechanism | How you select it | What must be installed |
| --- | --- | --- |
| Built-in CUDA / TensorRT EP | `SessionOptions::appendExecutionProvider('CUDA')` / `('TensorRT')` | A GPU-flavored ORT package (`Microsoft.ML.OnnxRuntime.Gpu.*`), CUDA runtime, cuDNN, driver |
| CUDA Plugin EP | `Runtime::registerExecutionProviderLibrary()` → `getEpDevices()` → `appendEpDevice()` | The same runtime pieces, plus `onnxruntime_providers_cuda` as a **plugin** whose version matches the ORT you load it into |

Related docs: [EP-CPU.md](EP-CPU.md), [EP-COREML.md](EP-COREML.md),
[EP-OPENVINO.md](EP-OPENVINO.md).

## What you need installed

### Linux x86_64

```bash
phpize
./configure --with-cuda13     # NuGet Microsoft.ML.OnnxRuntime.Gpu.Linux 1.32.0 = CUDA 13 + TensorRT 10
make -j8
# or
./configure --with-cuda12     # same package ID, 1.26.0 = CUDA 12 + TensorRT 10
```

- `--with-cuda12` / `--with-cuda13` are **Linux x86_64 only**; the CUDA
  packages are not offered for macOS or FreeBSD.
- CUDA 12 is pinned to ORT **1.26.0** because that is the last
  `Gpu.Linux` NuGet package built against CUDA 12; the same package ID moved to
  CUDA 13 in 1.27.0.
- `configure` checks for `libcublas.so.<major>` and `libcudnn.so.9`. A missing
  runtime is only a warning (build hosts often have no driver); set
  `ONNXRUNTIME_STRICT_CUDA_CHECK=1` to make it fatal.
- Switching between `--with-onnxruntime`, `--with-cuda12` and `--with-cuda13`
  requires `make clean` — `config.h` changes `ORT_API_VERSION` and stale
  objects would link against the wrong API table.

Runtime side, at load time:

- NVIDIA driver matching the toolkit major.
- `cudart`, `cublas`, `cublasLt`, `cufft`, `curand` and cuDNN 9 on
  `LD_LIBRARY_PATH`. The CUDA provider refuses to load if `libcublasLt` is
  missing; TensorRT additionally needs cuDNN.

### Windows x86_64

The Windows build path is `config.w32`, not `phpize`, and the SDK must be
`Microsoft.ML.OnnxRuntime.Gpu.Windows` (1.26.0 = CUDA 12 + TensorRT 10,
1.32.0 = CUDA 13 + TensorRT 10). Copy
`buildTransitive/native/include/*.h` → `include\` and
`runtimes/win-x64/native/*` → `lib\`, or let the `fetch-onnxruntime` action do it.

Runtime side — put these **next to `php.exe`**, not only on `PATH`:

```text
onnxruntime.dll
onnxruntime_providers_shared.dll
onnxruntime_providers_cuda.dll
onnxruntime_providers_tensorrt.dll   # only if you use TensorRT
```

Windows ships its own ORT 1.17 `onnxruntime.dll` in `C:\Windows\System32`, and
`System32` is searched before `PATH`. With the runtime only on `PATH` the
extension loads the system copy and fails with
`The requested API version [28] is not available ... Current ORT Version is: 1.17.1`.

Plus, on `PATH`:

- CUDA 13: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` (CUDA toolkit
  `bin\x64`) and `cudnn64_9.dll` (cuDNN 9).
- CUDA 12: the `*64_12.dll` equivalents plus `cudnn64_9.dll`.
- TensorRT 10.16: its `bin` directory (`nvinfer_10.dll`, `nvonnxparser_10.dll`,
  builder-resource DLLs).

## Using the built-in provider

```php
$options = new OnnxRuntime\SessionOptions();
$options->appendExecutionProvider('CUDA', ['device_id' => '0']);
$session = new OnnxRuntime\Session('model.onnx', $options);
```

`appendExecutionProvider('CUDA')` is routed to
`SessionOptionsAppendExecutionProvider_CUDA_V2` (and `('TensorRT')` to
`_TensorRT_V2`), with the options map applied through the matching
`Update...ProviderOptionsWithValue`. ORT's generic name-based appender rejects
both names, so this routing is required and is guarded by `tests/036`.

Common CUDA options (string → string, forwarded to ORT):

| Option | Meaning |
| --- | --- |
| `device_id` | CUDA device ordinal, default `0` |
| `arena_extend_strategy` | `kNextPowerOfTwo` (default) or `kSameAsRequested` |
| `gpu_mem_limit` | Arena size cap in bytes; empty means unlimited |
| `cudnn_conv_algo_search` | `EXHAUSTIVE` (default), `HEURISTIC`, `DEFAULT` |
| `do_copy_in_default_stream` | `1`/`0`; `0` only when you manage streams yourself |
| `enable_cuda_graph` | `1` to capture the graph for repeated identical shapes |
| `use_tf32` | `0`/`1` on Ampere and newer |

TensorRT ordering matters: the TensorRT EP falls back to CUDA, so append
**TensorRT first and CUDA second**:

```php
$options->appendExecutionProvider('TensorRT', ['trt_fp16_enable' => '1']);
$options->appendExecutionProvider('CUDA', ['device_id' => '0']);
```

Smoke numbers on an RTX 3050, MatMul 512×2048×2048 (PHP 8.5.12 ZTS, ORT 1.28.2):
CPU 12.51 ms/run, CUDA 3.20 ms/run, TensorRT fp16 2.95 ms/run (4.3 s engine
build, cached). NTS: 11.25 / 3.24 / 2.96 ms/run. These are smoke results, not
tuned baselines.

## Using the CUDA Plugin EP

Same runtime libraries, but the CUDA provider is loaded as a plugin instead of
being appended by name. Requires ORT 1.23.0+ (`Runtime::supports('ep_device')`).

```php
$runtime = OnnxRuntime\Runtime::default();
$runtime->registerExecutionProviderLibrary(
    'cuda',
    'D:\php\onnxruntime_providers_cuda.dll'
);

$options = new OnnxRuntime\SessionOptions();
foreach ($runtime->getEpDevices() as $device) {
    if ($device->epName() === 'CUDAExecutionProvider') {
        $options->appendEpDevice($device);
        break;
    }
}
$session = new OnnxRuntime\Session('model.onnx', $options);
```

Device tensors come from the same device object:

```php
$device = /* the CUDA EpDevice from above */;
$t = OnnxRuntime\Tensor::allocateOnDevice([2, 2], 'float32', $device);   // DeviceTensor

$host = OnnxRuntime\Tensor::fromArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
$t->copyFrom($host);          // CPU Tensor -> device
$copy = $t->copyTo();         // no arguments; returns a new CPU Tensor
$t->copyFromDevice($other);   // DeviceTensor -> DeviceTensor
```

`DeviceTensor` exposes `dtype()`, `shape()`, `rank()`, `memoryType()`,
`device()`, `copyFrom(Tensor)`, `copyTo(): Tensor`, `copyFromDevice(DeviceTensor)`
and `toArray()`. It is not constructible and cannot be cloned.

`tests/033-device-tensor-cuda.phpt` exercises exactly this path; it is skipped
unless `ONNXRUNTIME_CUDA_PLUGIN` points at the plugin DLL (defaults to
`D:\php\onnxruntime_providers_cuda.dll`). Registration is process-wide and lives
until module shutdown, and `unregisterExecutionProviderLibrary()` is refused
while any Session or DeviceTensor still holds a lease — the test asserts that.

The plugin must be built against the ORT version it is loaded into. A plugin
from a different ORT release can register successfully and then fail at
`Ort::Session` construction or produce wrong results; there is no compatibility
negotiation here.

## Limits that apply to both paths

- I/O Binding in this extension is **CPU binding**: `bindOutput()` allocates on
  the CPU and ORT copies across the PCIe boundary. The transfer is synchronized
  (`SynchronizeInputs()` before `Run()`, `SynchronizeOutputs()` before output
  copying), so a completed Task means the copies finished — not that a vendor
  stream was exposed as an async object.
- Unsupported nodes fall back to the CPU EP, so a CUDA session can be mostly
  CPU-bound on a graph CUDA does not cover.
- `appendExecutionProvider('CUDA')` and `appendEpDevice(cudaDevice)` are not
  interchangeable: the first uses the V1/V2 name appender without a device
  lease, the second reserves a plugin lease for the Session's lifetime.
- `enable_cuda_graph` requires stable shapes; dynamic inputs break or silently
  widen the capture. Prefer fixed-shape models, or `bindOutputBuffer()` with
  fixed shapes.
