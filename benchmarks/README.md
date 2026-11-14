# fastembed vs PHP ONNX Runtime benchmark

Model: [`Qdrant/bge-small-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q)
— BGE-small English, int8-quantized ONNX, 384 dimensions, BPE tokenizer.

## What is measured

The benchmark deliberately separates three layers:

| Column | Includes | Purpose |
| --- | --- | --- |
| `fastembed_e2e` | FastEmbed tokenizer + ORT run + CLS pooling + normalization | What a FastEmbed user experiences |
| `fastembed_ort` | FastEmbed tokenizer + raw ORT run | Tokenizer cost plus inference |
| `php_ort` | PHP extension ORT run with the same pre-tokenized `int64` inputs | Inference only |

The PHP extension does not ship a BPE tokenizer, so `php_ort` must not be
compared with `fastembed_e2e` as if tokenization were free. Compare
`php_ort` with `raw_ort_python` / `fastembed_ort` for the ORT layer.

Fixed workload: batch 8, sequence length 64, 3 warm-up runs, 20 measured runs,
one intra-op thread. Session creation is excluded from the measured section.

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
  --iterations 20
```

Provider values:

- `cpu`: `CPUExecutionProvider`
- `coreml`: `CoreMLExecutionProvider` (macOS)
- `cuda`: `CUDAExecutionProvider` (Windows/Linux with `onnxruntime-gpu`,
  plus `--cuda-lib` pointing at the PHP CUDA provider DLL on Windows)

Windows CUDA example:

```bat
python benchmarks\fastembed_int8_bpe.py ^
  --model-dir C:\models\bge-small-en-v1.5-onnx-Q ^
  --php D:\php\php.exe ^
  --extension D:\php\ext\php_onnxruntime.dll ^
  --provider cuda ^
  --cuda-lib D:\php\onnxruntime_providers_cuda.dll ^
  --ort-lib-dir E:\www\ext-onnxruntime\third_party\onnxruntime\lib
```

On Windows, install `fastembed-gpu` (which brings `onnxruntime-gpu`) and make
sure the CUDA/cuDNN DLLs are on `PATH` before running the benchmark. With
CUDA 13 and the pip-provided cuDNN, add these directories:

```text
<venv>\Lib\site-packages\nvidia\cudnn\bin
<venv>\Lib\site-packages\nvidia\cu13\bin\x86_64
```

## Results

macOS arm64, PHP 8.5.6 ZTS, extension ORT 1.28.2, Python ORT 1.28.0:

| Provider | fastembed e2e | fastembed tokenizer+ORT | Python raw ORT | PHP ORT |
| --- | ---: | ---: | ---: | ---: |
| CPU | 259.8 ms | 248.5 ms | 257.8 ms | 261.3 ms |
| CoreML | 260.7 ms | 258.1 ms | 261.0 ms | 248.0 ms |

Windows 11 x86_64, RTX 3050, PHP 8.5.12 ZTS, extension ORT 1.28.2,
Python ORT 1.32.0 (`fastembed-gpu` 0.9.1), CUDA 13.4, cuDNN 9.28.1.4:

| Provider | fastembed e2e | fastembed tokenizer+ORT | Python raw ORT | PHP ORT |
| --- | ---: | ---: | ---: | ---: |
| CPU | 241.7 ms | 238.8 ms | 239.6 ms | 224.3 ms |
| CUDA | 3.62 ms | 3.68 ms | 3.15 ms | 3.04 ms |

These are latency measurements for the fixed batch above, not throughput
claims. They are single-machine samples and should be rerun on the deployment
hardware. The Python and PHP runs use different ORT builds (1.32.0 vs 1.28.2),
so compare the columns as end-to-end samples, not as a controlled ORT-version
study.
