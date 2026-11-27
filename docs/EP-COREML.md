# CoreML execution provider (macOS, built-in)

CoreML is a **built-in** provider on macOS arm64: the ONNX Runtime dylib that
NuGet ships for `osx-arm64` already contains it, so there is nothing extra to
install and no plugin library to register.

Related docs: [EP-CPU.md](EP-CPU.md), [EP-OPENVINO.md](EP-OPENVINO.md),
[EP-CUDA.md](EP-CUDA.md).

## What you need installed

| Requirement | Detail |
| --- | --- |
| OS / arch | macOS on Apple silicon (arm64). macOS x86_64 is not published on NuGet and is not supported by the default build path. |
| ONNX Runtime | `Microsoft.ML.OnnxRuntime` 1.23.0 or newer for the EP-device API (`getEpDevices()`, `appendEpDevice()`); the name-based `appendExecutionProvider('CoreML')` path works on older SDKs too. |
| Apple frameworks | Provided by the OS. CoreML and the ANE need a reasonably current macOS; the extension itself adds no framework dependency. |

No CoreML-specific package exists on NuGet, and nothing is downloaded beyond the
normal CPU package:

```bash
phpize
./configure                    # CPU NuGet package also carries CoreML on macOS arm64
# or: ./configure --with-coreml # asserts the platform and records the requirement
make -j8
codesign --force --sign - modules/onnxruntime.so
```

`--with-coreml` is a build-time assertion, not a switch that adds code: it
rejects non-macOS hosts and records the requirement, while
`Runtime::supports('coreml')` is computed as "Apple platform and
`ORT_API_VERSION >= 23`". Building without the flag on macOS still gives you a
working CoreML EP; building with it is the documented, verifiable configuration.

## Check that it is there

```php
$runtime = OnnxRuntime\Runtime::default();
var_dump($runtime->getAvailableProviders());
var_dump($runtime->supports('coreml'));
```

On macOS arm64 with ORT 1.32.0 this prints:

```text
array(3) { [0]=> string(22) "CoreMLExecutionProvider" [1]=> string(24) "WebGpuExecutionProvider" [2]=> string(22) "CPUExecutionProvider" }
bool(true)
```

`WebGpuExecutionProvider` is listed by the same dylib but is not covered by this
project's tests.

## Using it: name path

The simplest form, and the one exercised by `tests/025-coreml.phpt`:

```php
$options = new OnnxRuntime\SessionOptions();
$options->appendExecutionProvider('CoreML', ['ModelFormat' => 'MLProgram']);
$session = new OnnxRuntime\Session('model.onnx', $options);
```

Built-in entries keep their append order. Because the CPU EP is always present
as the terminal fallback, nodes CoreML cannot execute still run on CPU, so a
partially covered graph does not fail — it just runs slower than expected.

Options are passed through to ORT's CoreML appender and validated when the
Session is created: an unknown key fails the constructor with
`Unknown option: <key>`. On ORT 1.32 these keys are accepted
(`['ModelFormat' => 'MLProgram']` selects the newer ML Program format instead of
the legacy `NeuralNetwork` one):

| Option | Verified on 1.32 | Meaning |
| --- | --- | --- |
| `ModelFormat` | yes (`MLProgram`, `NeuralNetwork`) | CoreML model format |
| `MLComputeUnits` | yes | which CoreML compute units may be used |
| `EnableOnSubgraphs` | yes | allow running graph partitions on CoreML |
| `RequireStaticInputShapes` | yes | reject dynamic shapes instead of recompiling |
| `ProfileComputePlan` | yes | dump the CoreML compute plan |
| `MLProgram` | **no** | rejected: `Unknown option: MLProgram`; it is not a key, it is a `ModelFormat` value |
| `ModelCacheDir`, `EnableLowPrecisionAccumulationOnGPU` | **no** | rejected as unknown on 1.32 |

A rejected option throws at `new OnnxRuntime\Session(...)`, not at
`appendExecutionProvider()`: the options are only parsed when the session is
created, which is why a snapshot-only test can store an invalid key without
noticing.

## Using it: device path

`getEpDevices()` exposes two CoreML devices on Apple silicon, both with the same
`epName()`; they differ by reported device type:

```text
device ep=CoreMLExecutionProvider type=npu vendor=Microsoft reg=NULL plugin=false
device ep=CoreMLExecutionProvider type=gpu vendor=Microsoft reg=NULL plugin=false
```

Select one explicitly when you care which backend runs:

```php
$runtime = OnnxRuntime\Runtime::default();
$options = new OnnxRuntime\SessionOptions();
foreach ($runtime->getEpDevices() as $device) {
    if ($device->epName() === 'CoreMLExecutionProvider' && $device->deviceType() === 'gpu') {
        $options->appendEpDevice($device);
        break;
    }
}
$session = new OnnxRuntime\Session('model.onnx', $options);
```

Do not select by `epName()` alone here: both devices share it, and the first
match is the `npu` entry.

## Caveats

- **Do not append CoreML on non-macOS hosts.** `appendExecutionProvider('CoreML')`
  fails there because the provider is not compiled into the library.
- CoreML compiles subgraphs into CoreML models; the first run of a new shape
  includes compilation cost. Time the second and later runs, or use
  `Runtime::supports('coreml')`-guarded benchmarks rather than a single cold call.
- The ANE may execute parts of the graph in reduced precision. Numerical output
  can differ from CPU in the last bits — verify tolerance per model instead of
  assuming bitwise equality.
- Dynamic shapes are supported but force recompilation per new shape; a model
  with a fully dynamic batch dimension pays that cost repeatedly.
- `appendExecutionProvider('CoreML')` and `appendEpDevice()` are different
  mechanisms: the first calls ORT's name-based appender, the second uses
  `SessionOptionsAppendExecutionProvider_V2` with a device lease.
