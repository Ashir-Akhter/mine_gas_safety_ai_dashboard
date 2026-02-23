# demo_predict_video.py
# =========================================
# Mining Gas AI Safety Demo (CORRECTED)
# Uses GasCNN + gas_model_best.pt
# Heatmap + SAFE/UNSAFE + probability
# =========================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

from simulation import create_base_layout, init_state, to_torch, step_gpu


# ======================================================
# SAME MODEL AS train_model.py  (IMPORTANT)
# ======================================================

class GasCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# SETTINGS
# ======================================================

H = 128
W = 128

STEPS = 400
FPS = 30

MODEL_PATH = "gas_model_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_VIDEO = "prediction_demo.mp4"

THRESHOLD = 0.5


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():
    model = GasCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ======================================================
# MAIN
# ======================================================

def run_demo():

    print("Loading trained model...")
    model = load_model()

    print("Creating layout...")
    layout = create_base_layout(H, W)
    gas, fans, leaks = init_state(layout)

    # stronger random leaks for visual variety
    for _ in range(6):
        i = np.random.randint(0, H)
        j = np.random.randint(0, W)
        if layout[i, j] == 1:
            leaks[i, j] = 2.0

    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks, DEVICE)

    # ==================================================
    # Plot
    # ==================================================

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(layout, cmap="gray", alpha=0.25, origin="lower")

    heat = ax.imshow(
        np.zeros_like(layout),
        cmap="inferno",
        norm=LogNorm(vmin=1e-4, vmax=1),
        origin="lower"
    )

    title = ax.set_title("Initializing...")
    plt.colorbar(heat, ax=ax)

    print("Rendering prediction video...")

    # ==================================================
    # UPDATE LOOP
    # ==================================================

    def update(frame):
        nonlocal gas_t

        # ---- simulate ----
        gas_t = step_gpu(
            layout_t,
            gas_t,
            fans_t,
            leaks_t,
            leak_strength=0.6,
            fan_strength=0.04,
            diffusion=0.8
        )

        gas_np = gas_t.detach().cpu().numpy()

        # ---- NORMALIZE (same as training) ----
        gas_norm = gas_np / (gas_np.max() + 1e-6)

        x = np.stack([gas_norm, fans, layout], axis=0).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        label = "UNSAFE" if prob > THRESHOLD else "SAFE"

        # ---- visuals ----
        heat.set_data(gas_np)

        title.set_text(f"{label}  |  risk={prob:.2f}  |  step={frame}")

        if label == "UNSAFE":
            title.set_color("red")
        else:
            title.set_color("lime")

        return [heat, title]


    ani = animation.FuncAnimation(
        fig,
        update,
        frames=STEPS,
        interval=1000 // FPS,
        blit=False
    )

    print("Saving video...")
    ani.save(OUT_VIDEO, fps=FPS)

    print(f"\nSaved -> {OUT_VIDEO}")


# ======================================================

if __name__ == "__main__":
    run_demo()