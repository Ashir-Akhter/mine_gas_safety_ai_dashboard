# demo_video.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm # this is to show more gas with better colours and more

from simulation import create_base_layout, init_state, to_torch, step_gpu


# =========================
# SETTINGS (edit freely)
# =========================
H = 512
W = 512
STEPS = 1200
FPS = 60
OUT_VIDEO = "mine_gas_demo.mp4"
DEVICE = "cuda"


# =========================
# MAIN
# =========================
def make_animation():

    print("Creating layout...")
    layout = create_base_layout(H, W)

    gas, fans, leaks = init_state(layout)

    import numpy as np

    # multi visible leaks for demo
    for _ in range(6):
        i = np.random.randint(0, H)
        j = np.random.randint(0, W)
        if layout[i, j] == 1:
            leaks[i, j] = 2.0   # VERY strong

    layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks, DEVICE)

    layout_np = layout
    fans_np = fans
    leaks_np = leaks

    # find fan/leak coordinates for markers
    fan_pts = np.argwhere(fans_np > 0)
    leak_pts = np.argwhere(leaks_np > 0)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xticks([])
    ax.set_yticks([])

    # background tunnels
    ax.imshow(layout_np, cmap="gray", alpha=0.25, origin="lower")

    gas_img = ax.imshow(
        np.zeros_like(layout_np),
        cmap="inferno",
        norm=LogNorm(vmin=1e-4, vmax=1),
        origin="lower",
        alpha=0.9
    )

    if len(fan_pts):
        ax.scatter(fan_pts[:,1], fan_pts[:,0], c="cyan", s=50, label="Fans")

    if len(leak_pts):
        ax.scatter(leak_pts[:,1], leak_pts[:,0], c="red", s=50, label="Leaks")

    title = ax.set_title("Gas Diffusion Simulation")
    ax.legend(loc="upper right")

    plt.colorbar(gas_img, ax=ax, label="Gas concentration")

    print("Rendering frames...")

    def update(frame):

        # Decision System --> shows when to evacuate and when not
        ax.set_title(f"Max gas {gas_np.max():.2f} | Unsafe cells {(gas_np>0.4).sum()}")

        nonlocal gas_t

        gas_t = step_gpu(
            layout_t,
            gas_t,
            fans_t,
            leaks_t,
            leak_strength=0.6,   # BIGGER leak
            fan_strength=0.04,    # weaker fan
            diffusion=0.8        # spreads fast
            )

        gas_np = gas_t.cpu().numpy()

        # for danger areas to glow bright white/red
        danger = gas_np > 0.4
        gas_np[danger] = 1.0

        # swowing motion trails:
        prev = gas_img.get_array()
        gas_img.set_data(0.7 * prev + 0.3 * gas_np)

        title.set_text(f"Gas Diffusion Step {frame}")

        # danger warning
        if gas_np.max() > 0.7:
            title.set_color("red")
        else:
            title.set_color("white")

        return [gas_img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=STEPS,
        interval=1000//FPS,
        blit=False
    )

    print("Saving video...")
    ani.save(OUT_VIDEO, fps=FPS)

    print(f"\nSaved -> {OUT_VIDEO}")


if __name__ == "__main__":
    make_animation()