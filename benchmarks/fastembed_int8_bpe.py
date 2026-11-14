#!/usr/bin/env python3
"""Compare FastEmbed, raw ONNX Runtime, and the PHP extension.

Model: Qdrant/bge-small-en-v1.5-onnx-Q (int8 ONNX, BGE-small, WordPiece tokenizer).

The benchmark separates three measurements:

* ``fastembed_e2e``: FastEmbed tokenize + ORT run + pool/normalize.
* ``fastembed_ort``: FastEmbed tokenizer + raw ORT run, no pooling.
* ``php_ort``: PHP extension ORT run with the same pre-tokenized inputs.

The PHP extension does not ship a tokenizer, so ``php_ort`` is intentionally
an inference-only number. Do not compare it directly with ``fastembed_e2e``
without accounting for the tokenizer and pooling work.

Example:
  python3 benchmarks/fastembed_int8_bpe.py \
    --model-dir /tmp/fastembed-cache/bge-small-en-v1.5-onnx-Q \
    --php /usr/local/bin/php \
    --extension modules/onnxruntime.so \
    --provider coreml \
    --iterations 100
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="directory containing model_optimized.onnx and tokenizer files")
    parser.add_argument("--php", required=True, help="PHP executable")
    parser.add_argument("--extension", required=True, help="onnxruntime extension path")
    parser.add_argument("--php-script", default=str(Path(__file__).with_name("php_ort_bench.php")))
    parser.add_argument("--php-tokenize-script",
                        default=str(Path(__file__).with_name("php_tokenize_ort_bench.php")))
    parser.add_argument("--provider", choices=["cpu", "coreml", "cuda"], default="cpu")
    parser.add_argument("--cuda-lib", default=None, help="CUDA provider DLL for PHP (Windows)")
    parser.add_argument("--ort-lib-dir", default=None, help="ORT shared-library directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default=None, help="write the result JSON here")
    return parser.parse_args()


def percentiles(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    n = len(ordered)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[int(n * 0.50)],
        "p95_ms": ordered[min(n - 1, int(n * 0.95))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def measure(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return percentiles(times)


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    model_path = model_dir / "model_optimized.onnx"
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    provider = {
        "cpu": "CPUExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[args.provider]

    import onnxruntime as ort
    available = ort.get_available_providers()
    if provider not in available:
        raise SystemExit(f"{provider} unavailable; available={available}")

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        specific_model_path=str(model_dir),
        providers=[provider],
        threads=args.threads,
    )
    tokenizer = model.model.tokenizer
    tokenizer.enable_truncation(max_length=args.sequence_length)
    tokenizer.enable_padding(length=args.sequence_length, pad_id=0, pad_token="[PAD]")

    text = "The quick brown fox jumps over the lazy dog."
    texts = [text] * args.batch_size

    encoded = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)
    onnx_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    # FastEmbed's tokenizer + raw ORT path (no pool/normalize).
    fastembed_ort = measure(lambda: model.model.onnx_embed(texts), args.warmup, args.iterations)
    # FastEmbed public API: tokenize + ORT + pool/normalize.
    fastembed_e2e = measure(
        lambda: list(model.embed(texts, batch_size=args.batch_size)), args.warmup, args.iterations)
    # Raw ORT only, with the exact arrays also given to PHP.
    raw_ort = measure(lambda: model.model._run_model(onnx_input), args.warmup, args.iterations)

    spec = {
        "model": str(model_path),
        "provider": args.provider,
        "cuda_lib": args.cuda_lib,
        "threads": args.threads,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "inputs": {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "token_type_ids": token_type_ids.tolist(),
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        spec_path = handle.name

    env = os.environ.copy()
    if args.ort_lib_dir:
        lib_dir = str(Path(args.ort_lib_dir).resolve())
        if platform.system() == "Darwin":
            env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
        elif platform.system() == "Windows":
            env["PATH"] = lib_dir + ";" + env.get("PATH", "")
        else:
            env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    command = [args.php, "-n", "-d", f"extension={Path(args.extension).resolve()}",
               str(Path(args.php_script).resolve()), spec_path]
    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    os.unlink(spec_path)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"PHP benchmark failed with exit code {proc.returncode}")

    php_result = json.loads(proc.stdout.strip().splitlines()[-1])

    tokenize_spec = {
        "model": str(model_path),
        "provider": args.provider,
        "cuda_lib": args.cuda_lib,
        "threads": args.threads,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "tokenizer_path": str(model_dir / "tokenizer.json"),
        "text": text,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(tokenize_spec, handle)
        tokenize_spec_path = handle.name

    tokenize_proc = subprocess.run(
        [args.php, "-n", "-d", f"extension={Path(args.extension).resolve()}",
         str(Path(args.php_tokenize_script).resolve()), tokenize_spec_path],
        env=env, text=True, capture_output=True)
    os.unlink(tokenize_spec_path)
    if tokenize_proc.returncode != 0:
        sys.stderr.write(tokenize_proc.stdout)
        sys.stderr.write(tokenize_proc.stderr)
        raise SystemExit(f"PHP tokenize benchmark failed with exit code {tokenize_proc.returncode}")

    php_tokenize_result = json.loads(tokenize_proc.stdout.strip().splitlines()[-1])

    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "provider": provider,
        "model": "Qdrant/bge-small-en-v1.5-onnx-Q",
        "model_file": str(model_path),
        "onnxruntime_python": ort.__version__,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "threads": args.threads,
        "fastembed_e2e": fastembed_e2e,
        "fastembed_ort": fastembed_ort,
        "raw_ort_python": raw_ort,
        "php_ort": php_result,
        "php_tokenize_ort": php_tokenize_result,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
