# benchmark.py

import time
import torch
import numpy as np

from simulation import (
    create_base_layout,
    init_state,
    step_cpu,
    step_gpu_many,
    to_torch
)

H = 512   # IMPORTANT: big grid for GPU
W = 512
STEPS = 5000


def cpu_benchmark():
    layout = create_base_layout(H, W)
    gas, fans, leaks = init_state(layout)

    t0 = time.time()

    for _ in range(STEPS):
        gas = step_cpu(layout, gas, fans, leaks)

    return time.time() - t0


def gpu_benchmark():
    layout = create_base_layout(H, W)
    gas, fans, leaks = init_state(layout)

    # convert FIRST (fixes your bug)
    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks, "cuda")

    # warmup (VERY important)
    step_gpu_many(layout_t, gas_t, fans_t, leaks_t, 50)
    torch.cuda.synchronize()

    t0 = time.time()

    step_gpu_many(layout_t, gas_t, fans_t, leaks_t, STEPS)
    torch.cuda.synchronize()

    return time.time() - t0


if __name__ == "__main__":

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    cpu_time = cpu_benchmark()
    gpu_time = gpu_benchmark()

    print(f"\nCPU time: {cpu_time:.3f}s")
    print(f"GPU time: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster 🚀")