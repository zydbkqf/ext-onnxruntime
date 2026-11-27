# fastembed vs PHP ONNX Runtime benchmark

Model: [`Qdrant/bge-small-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q)
— BGE-small English, int8-quantized ONNX, 384 dimensions, WordPiece tokenizer.

## What is measured

The benchmark separates tokenization and inference so the PHP-side tokenizer
cost is visible instead of being hidden inside a single end-to-end number:

| Column | Includes | Purpose |
| --- | --- | --- |
| `fastembed_e2e` | FastEmbed tokenizer + ORT run + CLS pooling + normalization | What a FastEmbed user experiences |
| `fastembed_ort` | FastEmbed tokenizer + raw ORT run | Tokenizer cost plus inference |
| `php_tokenize_ort` | PHP WordPiece tokenizer + tensor build + ORT run | PHP-side text-to-inference path |
| `php_ort` | PHP extension ORT run with the same pre-tokenized `int64` inputs | Inference only |

`php_tokenize_ort` uses the PHP WordPiece implementation in
`benchmarks/php_wordpiece.php`; it does not call Python for tokenization. The
token IDs are checked against the FastEmbed tokenizer for the fixed text.
`php_ort` is still the cleanest ORT-layer comparison against
`raw_ort_python`.

Fixed workload: batch 8, sequence length 64, 3 warm-up runs, 100 measured runs,
one intra-op thread. Session creation is excluded from the measured section;
tokenization is included in `fastembed_e2e`, `fastembed_ort`, and
`php_tokenize_ort`, and excluded from `php_ort`.

## Run

Python environment:

```bash
python3 -m venv /tmp/ort-bench-venv
/tmp/ort-bench-venv/bin/pip install fastembed "onnxruntime==1.28.0"
```

Download the model files into a directory containing
`model_optimized.onnx`, `tokenizer.json`, `tokenizer_config.json`, `config.json`,
and `special_tokens_map.json`.

```bash
python3 benchmarks/fastembed_int8_bpe.py \
  --model-dir /path/to/bge-small-en-v1.5-onnx-Q \
  --php /usr/local/php/bin/php \
  --extension modules/onnxruntime.so \
  --ort-lib-dir /path/to/onnxruntime/lib \
  --provider cpu \
  --iterations 100
```

Provider values:

- `cpu`: `CPUExecutionProvider`
- `coreml`: `CoreMLExecutionProvider` (macOS)
- `cuda`: `CUDAExecutionProvider` (Windows/Linux with `onnxruntime-gpu`,
  plus `--cuda-lib` pointing at the PHP CUDA provider DLL on Windows)
- `openvino`: `OpenVINOExecutionProvider` (the standalone plugin package
  `Intel.ML.OnnxRuntime.EP.OpenVINO`; pass `--openvino-lib` pointing at
  `onnxruntime_providers_openvino_plugin.dll` on Windows, or the plugin `.so`
  on Linux). The FastEmbed/Python side additionally needs
  `pip install onnxruntime-ep-openvino`.
- `tensorrt`: `TensorrtExecutionProvider` with `CUDAExecutionProvider` as its
  fallback. TensorRT builds an engine on first run; `--trt-cache-dir` points at
  a cache directory so reruns reuse it instead of rebuilding. The engine build
  happens before the measured section, so the numbers are steady-state.

`--output FILE` writes the full result JSON for every provider column
(fastembed e2e, fastembed tokenizer+ORT, Python raw ORT, PHP tokenize+ORT,
PHP ORT); the committed samples live in `benchmarks/results/`.

Windows CUDA example:

```bat
python benchmarks\fastembed_int8_bpe.py ^
  --model-dir C:\models\bge-small-en-v1.5-onnx-Q ^
  --php D:\php\php.exe ^
  --extension D:\php\ext\php_onnxruntime.dll ^
  --provider cuda ^
  --cuda-lib D:\php\onnxruntime_providers_cuda.dll ^
  --ort-lib-dir E:\ort\onnxruntime-win-x64-1.28.2\lib
```

On Windows, install `fastembed-gpu` (which brings `onnxruntime-gpu`) and make
sure the CUDA/cuDNN DLLs are on `PATH` before running the benchmark. With
CUDA 13 and the pip-provided cuDNN, add these directories:

```text
<venv>\Lib\site-packages\nvidia\cudnn\bin
<venv>\Lib\site-packages\nvidia\cu13\bin\x86_64
```

## Execution-provider benchmark

`php_ep_bench.php` measures the same model on CPU and on a device-backed EP
without needing the Python FastEmbed stack. It reads static shapes from the
model; for a 2D dynamic input it uses `--batch-size` and `--sequence-length`.
The model inputs are deterministic dummy values, so this compares execution
speed, not accuracy.

```bat
php -n -d extension=D:\php\ext\php_onnxruntime.dll benchmarks\php_ep_bench.php ^
  --model D:\models\bge-small-en-v1.5-onnx-Q\model_optimized.onnx ^
  --provider openvino ^
  --openvino-lib D:\php\openvino-ep\onnxruntime_providers_openvino_plugin.dll ^
  --iterations 50 --warmup 5 --threads 1 ^
  --output benchmarks\results\windows-zts-openvino-ep.json
```

Windows 11, i5-12400F, PHP 8.6.0 ZTS, extension 0.4.12 / ORT 1.32.0,
OpenVINO plugin package `Intel.ML.OnnxRuntime.EP.OpenVINO` 1.8.1, BGE-small
int8, batch 8 × 64, 1 intra-op thread, 50 measured runs:

| Provider | Mean | p50 | p95 | Speedup (p50) |
| --- | ---: | ---: | ---: | ---: |
| CPU | 248.4 ms | 245.2 ms | 259.1 ms | 1.00× |
| OpenVINO (CPU device) | 75.8 ms | 71.9 ms | 100.9 ms | 3.41× |

The OpenVINO plugin registered `OpenVINOExecutionProvider` and an
`.AUTO` device; this run used the explicit CPU device. The Microsoft
`Microsoft.ML.OnnxRuntime` NuGet package does not expose a `DNNLExecutionProvider`
(`appendExecutionProvider('DNNL')` is rejected as an unknown provider); oneDNN
is used underneath OpenVINO's CPU plugin. An explicit DNNL EP requires building
ONNX Runtime from source with `--use_dnnl`.

### Linux

Same model and workload, PHP 8.5.12 NTS, extension 0.4.13, ORT 1.32.0,
OpenVINO plugin `Intel.ML.OnnxRuntime.EP.OpenVINO` 1.8.1 (`linux-x64`),
BGE-small int8, batch 8 x 64. Host: KVM VM on an Intel Xeon Platinum with
**2 physical cores / 4 logical CPUs** (SMT), 7.1 GiB RAM. This is not
comparable to the Windows numbers above: different CPU, different core count.

`php_ep_bench.php`, CPU column then OpenVINO column in the same process, 20
measured runs:

| Provider | `--threads` | Mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 213.1 ms | 213.2 ms | 214.5 ms |
| CPU | 4 | 106.8 ms | 106.8 ms | 107.7 ms |
| OpenVINO | 1 | 101.4 ms | 101.2 ms | 102.5 ms |
| OpenVINO | 4 | 100.7 ms | 100.6 ms | 101.8 ms |

CPU usage, measured on isolated single-provider runs
(`benchmarks/php_ep_cpu_profile.php` wrapped in `/usr/bin/time -f "cpu%=%P"`), 20
measured runs:

| Provider | intra-op threads | p50 | Process CPU |
| --- | ---: | ---: | ---: |
| CPU | 1 | 214.2 ms | 99% |
| CPU | 4 | 106.5 ms | 354% |
| OpenVINO | 1 | 101.2 ms | 173% |
| OpenVINO | 4 | 100.7 ms | 176% |

OpenVINO thread bounds, same isolated harness. `--threads` is ignored by the
plugin, and so is its `num_of_threads` provider option (it logs
`Provider option num_of_threads is ignored. Please set it via the load_config as:
"INFERENCE_NUM_THREADS": "1"`). The working knob is the device-keyed
`load_config` JSON:

| OpenVINO setting | p50 | Process CPU |
| --- | ---: | ---: |
| `{"CPU":{"INFERENCE_NUM_THREADS":"1"}}` | 197.5 ms | 101% |
| `{"CPU":{"INFERENCE_NUM_THREADS":"2"}}` | 101.6 ms | 175% |
| default, no `load_config` | 100.4 ms | 176% |
| `--threads=4`, no `load_config` | 100.7 ms | 176% |

What these tables mean:

- `--threads` only sets ORT's intra-op thread count. The OpenVINO CPU plugin
  runs its own TBB pool, so the option does not bound it: wall time stays at
  ~101 ms whether it is 1 or 4, while the process keeps consuming ~1.7 cores.
  `OMP_NUM_THREADS=1` does not change that either — the plugin is TBB-based, not
  OpenMP-based. Bounding it *is* possible, just not through that option:
  `load_config` with `INFERENCE_NUM_THREADS=1` pins it to one core (197.5 ms /
  101%), which proves the earlier "cannot be pinned" reading was a knob problem,
  not a plugin limitation.
- The Windows 3.41x figure above is therefore a wall-clock ratio at unequal CPU
  budgets: OpenVINO runs at its default width while CPU EP is pinned to one
  thread on a 6-core i5. That run did not record CPU usage, so it cannot be read
  as per-core efficiency.
- OpenVINO's default footprint here is ~2 threads (~1.7 cores), i.e. the
  physical core count of this host, not the 4 logical CPUs. The honest
  same-budget comparison is therefore OpenVINO default (100.4 ms / 176%) against
  CPU EP at 2 intra-op threads (110.0 ms / 187%): ~8% faster for slightly less
  CPU. Against CPU EP at 4 threads it is ~5% faster while using half the CPU,
  because the extra two CPU threads are hyperthreads on this host.

The same caveat applies to the DNNL notes above: a provider that manages its own
thread pool will look fast whenever the baseline is thread-limited.

## Results

macOS arm64, PHP 8.5.12 NTS (Homebrew), extension ORT 1.28.2, Python ORT
1.28.0, 100 measured runs:

| Provider | fastembed e2e | fastembed tokenizer+ORT | Python raw ORT | PHP tokenize+ORT | PHP ORT |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 252.2 ms | 252.5 ms | 251.2 ms | 253.2 ms | 251.3 ms |
| CoreML | 253.2 ms | 254.5 ms | 255.6 ms | 253.3 ms | 255.1 ms |

Windows 11 x86_64, RTX 3050, PHP 8.5.12 ZTS, extension ORT 1.28.2,
Python ORT 1.32.0 (`fastembed-gpu` 0.9.1), CUDA 13.4, cuDNN 9.28.1.4,
100 measured runs:

| Provider | fastembed e2e | fastembed tokenizer+ORT | Python raw ORT | PHP tokenize+ORT | PHP ORT |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 240.3 ms | 239.3 ms | 237.8 ms | 222.8 ms | 223.0 ms |
| CUDA | 3.64 ms | 3.72 ms | 3.05 ms | 3.19 ms | 3.16 ms |
| CUDA + TensorRT | 5.26 ms | 5.25 ms | 3.69 ms | 6.45 ms | 6.37 ms |

Windows CUDA is dominated by the GPU path: PHP tokenize+ORT is about 0.18 ms
slower than PHP ORT-only for this fixed batch, while the FastEmbed Python
wrapper is about 0.69 ms slower than raw ORT.

The TensorRT row is not a win here: TensorRT only takes the subgraphs it
supports and hands the rest back to the CUDA EP, so on a batch-8 × 64-token
BGE-small model its per-run overhead dominates and it lands ~2x slower than
plain CUDA. It is reported because it proves the path runs, not as a
recommendation for small models.

TensorRT also requires shape inference to be present on the model before
session creation — this int8 QDQ export has no inferred intermediate shapes,
so ORT fails with `TensorRT input: <node> has no shape specified`. The row
above was measured on a byte-identical copy of `model_optimized.onnx` that
went through `onnx.shape_inference.infer_shapes()` first
(`E:\models\bge-small-en-v1.5-onnx-Q-trt`); same weights, same graph, only
shape annotations added. The CPU and CUDA rows use the original file.

These are latency measurements for the fixed batch above, not throughput
claims. They are single-machine samples and should be rerun on the deployment
hardware. When the Python ORT and PHP extension ORT versions differ (for
example, a Windows environment with Python ORT 1.32 and extension ORT 1.28),
compare the columns as end-to-end samples, not as a controlled ORT-version
study.

## Result JSON

Committed samples from the Windows run above (100 iterations each, PHP 8.5.12
ZTS extension 0.4.0, Python ORT 1.32.0), plus the Linux ep-bench runs (20
iterations, PHP 8.5.12 NTS extension 0.4.13):

- `benchmarks/results/windows-zts-cpu.json`
- `benchmarks/results/windows-zts-cuda.json`
- `benchmarks/results/windows-zts-tensorrt.json`
- `benchmarks/results/windows-zts-openvino-ep.json`
- `benchmarks/results/linux-nts-openvino-ep-t1.json`
- `benchmarks/results/linux-nts-openvino-ep-t4.json`

Each file records `platform`, `machine`, `provider`, `model_file`, and the
five percentile sets (mean/p50/p95/min/max) for every column, so a table row
can be rebuilt without rerunning. Regenerate:

```bat
python benchmarks\fastembed_int8_bpe.py ^
  --model-dir E:\models\bge-small-en-v1.5-onnx-Q ^
  --php D:\php\php.exe ^
  --extension D:\php\ext\php_onnxruntime.dll ^
  --provider cuda --cuda-lib D:\php\onnxruntime_providers_cuda.dll ^
  --iterations 100 --output benchmarks\results\windows-zts-cuda.json
```

For the TensorRT row, pass `--provider tensorrt`, point `--model-dir` at the
shape-inferred copy (`E:\models\bge-small-en-v1.5-onnx-Q-trt`), and add
`--trt-cache-dir E:\trt-cache` so the engine is cached instead of rebuilt.
