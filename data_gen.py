# data_gen.py
import argparse
import time
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch

from utils import set_seed
from simulation import create_base_layout, init_state, add_random_openings, to_torch, step_gpu


# config
DEFAULT_GRID = 128
DEFAULT_STEPS = 200
SAMPLES_TO_SAVE = 8
RESULTS_CSV = "results.csv"
SAMPLES_DIR = Path("dataset_samples")
OUT_DIR = Path("dataset")

SAVE_SEQUENCE = False


def sample_random_config(H=DEFAULT_GRID, W=DEFAULT_GRID):
    layout = create_base_layout(H, W)

    if np.random.rand() < 0.6:
        layout = add_random_openings(layout, n_open=np.random.randint(1, 4))

    gas, fans, leaks = init_state(layout)

    tunnel_indices = list(zip(*np.where(layout == 1)))

    leaks[:] = 0.0
    fans[:] = 0.0

    for _ in range(np.random.randint(1, 5)):
        i, j = tunnel_indices[np.random.randint(len(tunnel_indices))]
        leaks[i, j] = float(0.6 + 0.8 * np.random.rand())

    for _ in range(np.random.randint(0, 5)):
        i, j = tunnel_indices[np.random.randint(len(tunnel_indices))]
        fans[i, j] = float(0.6 + 0.8 * np.random.rand())

    diffusion = float(0.12 + 0.18 * np.random.rand())
    leak_strength = float(0.03 + 0.07 * np.random.rand())
    fan_strength = float(0.05 + 0.12 * np.random.rand())

    return layout, gas, fans, leaks, diffusion, leak_strength, fan_strength


def run_scenario(T=DEFAULT_STEPS, device="cuda"):
    layout, gas_np, fans_np, leaks_np, diffusion, leak_strength, fan_strength = sample_random_config()

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    layout_t, gas_t, fans_t, leaks_t = to_torch(
        layout, gas_np, fans_np, leaks_np, device=dev.type
    )

    for _ in range(T):
        gas_t = step_gpu(
            layout_t,
            gas_t,
            fans_t,
            leaks_t,
            leak_strength=leak_strength,
            fan_strength=fan_strength,
            diffusion=diffusion,
        )

    final_gas = gas_t.detach().cpu().numpy()

    # improved labeling
    max_gas = final_gas.max()
    mean_gas = final_gas.mean()
    danger_ratio = (final_gas > 0.6).mean()

    label = 1 if (
        max_gas > 0.7 or
        mean_gas > 0.05 or
        danger_ratio > 0.01
    ) else 0

    return final_gas, layout, fans_np.astype(np.float32), label


def generate_dataset(N=1000, out_path="dataset.npz", device="cuda", seed=42):
    set_seed(seed)

    xs = []
    ys = []

    OUT_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(exist_ok=True)

    unsafe = 0

    for i in tqdm(range(N), desc="Generating dataset"):
        gas, layout, fans, label = run_scenario(device=device)

        x = np.stack(
            [
                gas.astype(np.float32),
                fans.astype(np.float32),
                layout.astype(np.float32),
            ],
            axis=0,
        )

        xs.append(x)
        ys.append(label)

        if label == 1:
            unsafe += 1

    xs = np.stack(xs)
    ys = np.array(ys)

    np.savez_compressed(out_path, x=xs, y=ys)

    print(f"Saved -> {out_path}")
    print(f"Shape: {xs.shape}")
    print(f"Unsafe ratio: {unsafe/N:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", default="dataset/dataset.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_dataset(args.n, args.out, args.device)