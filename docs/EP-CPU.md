# CPU execution provider

The CPU EP is a **built-in** provider: it ships inside every ONNX Runtime build,
including the plain `Microsoft.ML.OnnxRuntime` NuGet package that `./configure`
downloads by default. There is nothing extra to install, and a Session uses it
without any configuration.

Related docs: [EP-COREML.md](EP-COREML.md), [EP-OPENVINO.md](EP-OPENVINO.md),
[EP-CUDA.md](EP-CUDA.md).

## What you need installed

Nothing beyond ONNX Runtime itself. Pick one source:

| Source | Version seen here | How to select it |
| --- | --- | --- |
| NuGet `Microsoft.ML.OnnxRuntime` | 1.18.0 – 1.32.0 (default 1.32.0) | `./configure` (default) |
| System package | apt 1.21.0, dnf 1.22.2, FreeBSD pkg 1.30.0, Homebrew 1.32.0 | `./configure --with-onnxruntime` |
| Unpacked SDK directory | any | `./configure --with-onnxruntime=/opt/ort` |

The core API (inference, I/O Binding, async, batch) works from ORT 1.18.0.
`appendEpDevice()`, `DeviceTensor` and Plugin EP registration need 1.23.0+.
apt 1.21 / dnf 1.22 are therefore fully usable for CPU inference, but the
device API throws `requires ONNX Runtime 1.23 or newer` there.

## Build and load

```bash
phpize
./configure            # CPU NuGet package, 1.32.0
make -j8
php -d extension=$PWD/modules/onnxruntime.so -r 'var_dump(OnnxRuntime\Runtime::default()->getAvailableProviders());'
```

macOS always needs a re-signature after every `make`:

```bash
codesign --force --sign - modules/onnxruntime.so
```

Without it the module is killed with `Code Signature Invalid`. Linux and
FreeBSD need no extra step; on FreeBSD use the `pkg` ONNX Runtime, since NuGet
has no FreeBSD RID — see [FREEBSD.md](FREEBSD.md).

## Using it

By default a Session appends no provider and ORT keeps its own default order,
where `CPUExecutionProvider` is always last and always present. So this is
enough:

```php
$session = new OnnxRuntime\Session('model.onnx');
```

Append it explicitly only when you want a defined order, for example to keep a
GPU EP first and CPU as the guaranteed fallback:

```php
$options = new OnnxRuntime\SessionOptions();
$options->appendExecutionProvider('CPUExecutionProvider');   // built-in path
$session = new OnnxRuntime\Session('model.onnx', $options);
```

`appendExecutionProvider('cpu')`, `('CPU')` and `('CPUExecutionProvider')` all
resolve to the same provider: the name is matched case-insensitively, and the
`ExecutionProvider` / `EP` suffix is optional.

## Threads and memory

`setIntraOpNumThreads()` is the only knob that bound CPU EP work:

```php
$options = new OnnxRuntime\SessionOptions();
$options->setIntraOpNumThreads(2);
```

Measured on the Linux benchmark host (2 physical cores / 4 logical CPUs, SMT;
BGE-small int8, batch 8 × 64, 20 runs, `benchmarks/php_ep_cpu_profile.php`):

| intra-op threads | p50 | Process CPU |
| ---: | ---: | ---: |
| 1 | 214.2 ms | 99% |
| 2 | 110.0 ms | 187% |
| 3 | 135.5 ms | 278% |
| 4 | 106.5 ms | 354% |

Going from 2 to 4 threads buys ~3% wall time for roughly double the CPU, because
the extra two are hyperthreads. **Set this to the physical core count, not the
logical one**, unless the process is genuinely the only load on the machine.

Other knobs, both default-on:

- `setEnableCpuMemArena(bool)` — arena allocator for intermediate buffers.
  Disable it when several long-lived Sessions should not hold pooled memory.
- `setEnableMemPattern(bool)` — memory pattern planning for fixed shapes.
- `setExecutionMode('parallel')` — parallel execution of independent graph
  branches; `sequential` is the default. `parallel` only pays off on graphs with
  several wide independent branches, and it interacts badly with the intra-op
  thread count (both pools compete for the same cores).

## Caveats

- The CPU EP is the terminal fallback of every other EP. When a GPU EP supports
  only part of the graph, the remaining nodes run on CPU, which is why a
  "GPU" session can still be mostly CPU-bound.
- `appendExecutionProvider('CPU')` appends a second, explicit entry; it does not
  replace the implicit fallback. Do not append it in a loop.
- CPU EP is correct on every dtype the extension exposes; `int8`/`uint8` models
  run quantized kernels, but the observed speed depends on the model's quantization
  layout, not on the extension.
