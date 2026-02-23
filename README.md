# Mining Safety AI Digital Twin (v1)

**Short:** GPU-accelerated 2D mining digital twin that simulates gas diffusion in a 64×64 grid (tunnels vs rock), supports domain randomization, generates synthetic datasets and trains a small CNN to classify unsafe scenarios. Includes CPU vs GPU timing for benchmarking and a simple greedy control policy for fan placement.

## Highlights
- 2D grid simulation with leaks, ventilation fans, and diffusion physics.
- GPU-accelerated inner loop using `torch.conv2d`.
- Domain randomization (layouts, leak/fan counts, strengths, diffusion) to generate robust synthetic datasets.
- Trains a compact CNN (BatchNorm + global average pooling) — fast to train on an RTX laptop.
- Demo script that compares CPU vs GPU speed, prints max gas and model predictions, and plots a heatmap.
- Simple greedy fan placement policy (baseline for RL).

## Quick start
1. Create a venv and install:
```bash
pip install -r requirements.txt