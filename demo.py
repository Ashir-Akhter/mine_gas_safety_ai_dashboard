# demo.py
from utils import time_function, log_results
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation import create_base_layout, init_state, to_torch, step_gpu, step_cpu, add_random_openings
from model import SafetyCNN
from data_gen import sample_random_config
from simulation import step_cpu as sim_step_cpu
import argparse

def greedy_place_fans(layout_np, warmup_steps=40, k=3, device='cuda'):
    """
    Simple policy: run a short warmup (leaks only) to find high-gas locations then place k fans there.
    """
    layout, gas_np, fans_np, leaks_np, diffusion, leak_strength, fan_strength = sample_random_config()
    # run warmup on CPU (fast) to get hotspots
    gas = gas_np.copy()
    for _ in range(warmup_steps):
        gas = sim_step_cpu(layout, gas, np.zeros_like(fans_np), leaks_np,
                           leak_strength=leak_strength, fan_strength=fan_strength, diffusion=diffusion)
    # pick top-k indices
    flat = gas.flatten()
    topk_idx = np.argsort(flat)[-k:]
    fans = np.zeros_like(fans_np)
    H, W = gas.shape
    for idx in topk_idx:
        i, j = divmod(int(idx), W)
        fans[i, j] = 1.0
    return layout, gas_np, fans, leaks_np, diffusion, leak_strength, fan_strength

def benchmark(layout, gas_np, fans_np, leaks_np, steps, device):
    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas_np, fans_np, leaks_np, device)

    def cpu_fn():
        g = gas_np.copy()
        for _ in range(steps):
            g = step_cpu(layout, g, fans_np, leaks_np)

    def gpu_fn():
        g = gas_t
        for _ in range(steps):
            g = step_gpu(layout_t, g, fans_t, leaks_t)

    cpu_mean, _ = time_function(cpu_fn)
    gpu_mean, _ = time_function(gpu_fn)

    return cpu_mean, gpu_mean

def run_single_demo(device='cuda', use_greedy=False):
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    if use_greedy:
        layout, gas_np, fans_np, leaks_np, diffusion, leak_strength, fan_strength = greedy_place_fans(device=device)
    else:
        layout, gas_np, fans_np, leaks_np, diffusion, leak_strength, fan_strength = sample_random_config()

    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas_np, fans_np, leaks_np, device)

    # GPU run
    start = time.perf_counter()
    g = gas_t
    max_g = 0.0
    steps = 200
    for _ in range(steps):
        g = step_gpu(layout_t, g, fans_t, leaks_t,
                     leak_strength=leak_strength,
                     fan_strength=fan_strength,
                     diffusion=diffusion)
        max_g = max(max_g, float(g.max().item()))
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    # CPU run for comparison
    gcpu = gas_np.copy()
    start = time.perf_counter()
    for _ in range(steps):
        gcpu = sim_step_cpu(layout, gcpu, fans_np, leaks_np,
                            leak_strength=leak_strength,
                            fan_strength=fan_strength,
                            diffusion=diffusion)
    cpu_time = time.perf_counter() - start

    label = "UNSAFE" if max_g > 0.7 else "SAFE"
    print(f"Max gas: {max_g:.3f} → {label}")
    print(f"CPU time: {cpu_time:.3f}s, GPU time: {gpu_time:.3f}s, speedup: {cpu_time / max(gpu_time, 1e-6):.2f}x")

    # model prediction
    model = SafetyCNN().to(dev)
    try:
        model.load_state_dict(torch.load('safety_cnn.pt', map_location=dev))
    except Exception:
        print("Model checkpoint not found. Skipping model prediction.")
        model = None

    if model is not None:
        model.eval()
        with torch.no_grad():
            x = np.stack([g.detach().cpu().numpy(), fans_np], axis=0)[None]
            xt = torch.tensor(x, dtype=torch.float32, device=dev)
            logit = model(xt)
            prob = torch.sigmoid(logit)[0, 0].item()
        print(f"Model predicted unsafe probability: {prob:.3f}")

    # heatmap
    plt.figure(figsize=(5,5))
    plt.imshow(g.detach().cpu().numpy(), cmap='hot', origin='lower')
    plt.title(f"Gas concentration ({label})")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--greedy', action='store_true', help='Use greedy fan placement')
    args = parser.parse_args()
    run_single_demo(device=args.device, use_greedy=args.greedy)