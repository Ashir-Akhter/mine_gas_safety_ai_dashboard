# simulation.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import random


# =========================================================
# CONFIG
# =========================================================
DEFAULT_H = 64
DEFAULT_W = 64

# Use:
#   SEED = 42    -> deterministic
#   SEED = None  -> random every run (recommended for dataset gen)
SEED = None


# =========================================================
# SAFE SEEDING (FIXED)
# =========================================================
# Only seed if integer. Prevents crash when SEED=None.
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# LAYOUT CREATION
# =========================================================
def create_base_layout(H=DEFAULT_H, W=DEFAULT_W):
    layout = np.zeros((H, W), dtype=np.int8)

    mid = H // 2

    # main corridor
    layout[mid - 3:mid + 4, 5:W - 5] = 1

    # side rooms
    layout[mid - 5:mid + 6, W // 4 - 3:W // 4 + 3] = 1
    layout[mid - 5:mid + 6, 3 * W // 4 - 3:3 * W // 4 + 3] = 1

    return layout


def add_random_openings(layout: np.ndarray, n_open=2):
    """
    Random small branches for layout variety
    """
    H, W = layout.shape

    for _ in range(n_open):
        r = np.random.randint(26, 34)
        c = np.random.randint(8, 56)
        length = np.random.randint(2, 8)
        direction = np.random.choice([-1, 1])

        for k in range(length):
            cc = c + k * direction
            if 0 <= cc < W:
                layout[r, cc] = 1

    return layout


# =========================================================
# STATE INIT
# =========================================================
def init_state(layout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gas = np.zeros_like(layout, dtype=np.float32)
    fans = np.zeros_like(layout, dtype=np.float32)
    leaks = np.zeros_like(layout, dtype=np.float32)

    # default demo placement (data_gen overrides anyway)
    mid = layout.shape[0] // 2
    leaks[mid, layout.shape[1] // 4] = 1.0
    fans[mid, 3 * layout.shape[1] // 4] = 1.0

    return gas, fans, leaks


# =========================================================
# CPU DIFFUSION
# =========================================================
def step_cpu(
        layout: np.ndarray,
        gas: np.ndarray,
        fans: np.ndarray,
        leaks: np.ndarray,
        leak_strength=1.0,
        fan_strength=0.0,
        diffusion=0.5
) -> np.ndarray:
    mask = (layout == 1)

    up = np.roll(gas, -1, axis=0)
    down = np.roll(gas, 1, axis=0)
    left = np.roll(gas, -1, axis=1)
    right = np.roll(gas, 1, axis=1)

    neighborhood = (gas + up + down + left + right) / 5.0

    val = (1 - diffusion) * gas + diffusion * neighborhood
    val += leak_strength * leaks
    val -= fan_strength * fans
    val = np.clip(val, 0.0, None)

    val *= mask.astype(np.float32)
    return val


# =========================================================
# GPU DIFFUSION
# =========================================================
_KERNELS = {}


def get_kernel(device):
    """
    4-neighbour averaging kernel
    """
    key = device.type

    if key not in _KERNELS:
        k = torch.tensor(
            [[0.0, 0.25, 0.0],
             [0.25, 0.0, 0.25],
             [0.0, 0.25, 0.0]],
            dtype=torch.float32,
            device=device
        ).view(1, 1, 3, 3)

        _KERNELS[key] = k

    return _KERNELS[key]


def to_torch(layout_np, gas_np, fans_np, leaks_np, device='cuda'):
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    return (
        torch.from_numpy(layout_np).to(dev),
        torch.from_numpy(gas_np).to(dev),
        torch.from_numpy(fans_np).to(dev),
        torch.from_numpy(leaks_np).to(dev)
    )


def step_gpu(
        layout: torch.Tensor,
        gas: torch.Tensor,
        fans: torch.Tensor,
        leaks: torch.Tensor,
        leak_strength=1.0,
        fan_strength=0.0,
        diffusion=0.5
) -> torch.Tensor:
    device = gas.device
    kernel = get_kernel(device)

    gas_4d = gas.unsqueeze(0).unsqueeze(0)
    neighborhood = F.conv2d(gas_4d, kernel, padding=1).squeeze(0).squeeze(0)

    val = (1 - diffusion) * gas + diffusion * neighborhood
    val += leak_strength * leaks
    val -= fan_strength * fans
    val = torch.clamp(val, min=0.0)

    return torch.where(layout == 1, val, torch.zeros_like(val))


def step_gpu_many(layout, gas, fans, leaks, steps):
    g = gas
    for _ in range(steps):
        g = step_gpu(layout, g, fans, leaks)
    return g