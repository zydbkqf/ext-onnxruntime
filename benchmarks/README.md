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
| CPU | 241.4 ms | 240.7 ms | 239.6 ms | 226.2 ms | 223.2 ms |
| CUDA | 3.73 ms | 3.67 ms | 3.04 ms | 3.27 ms | 3.10 ms |

Windows CUDA is dominated by the GPU path: PHP tokenize+ORT is about 0.18 ms
slower than PHP ORT-only for this fixed batch, while the FastEmbed Python
wrapper is about 0.69 ms slower than raw ORT.

These are latency measurements for the fixed batch above, not throughput
claims. They are single-machine samples and should be rerun on the deployment
hardware. When the Python ORT and PHP extension ORT versions differ (for
example, a Windows environment with Python ORT 1.32 and extension ORT 1.28),
compare the columns as end-to-end samples, not as a controlled ORT-version
study.
