import argparse
import time
from pathlib import Path

import torch


def benchmark_model(model_path: str, depth: int, height: int, width: int, warmup: int, runs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    x = torch.randn(1, 1, depth, height, width, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    total_time = end - start
    avg_latency = total_time / runs
    fps = 1.0 / avg_latency
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

    print("Inference Benchmark")
    print("-------------------")
    print(f"Device: {device}")
    print(f"Input shape: (1, 1, {depth}, {height}, {width})")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Warmup runs: {warmup}")
    print(f"Measured runs: {runs}")
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)

    args = parser.parse_args()

    benchmark_model(
        model_path=args.model_path,
        depth=args.depth,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
