# Mining Gas Safety AI Digital Twin

GPU-accelerated mining safety simulator + AI risk detector built with PyTorch CUDA.

A real-time 2D digital twin that simulates gas diffusion inside underground tunnels, generates synthetic training data via domain randomization, and uses a CNN to classify unsafe gas conditions. Includes a live Streamlit dashboard with automatic alarm alerts.

---

## Live Demo
Streamlit App:
https://aimine.streamlit.app/

Demo video:
Video on my LinkedIn under projects: https://www.linkedin.com/in/ashir-akhter/details/projects/

---

## Features

### Simulation
- 2D mine tunnel layout generation
- Gas diffusion physics
- Leaks + ventilation fans
- Domain randomization for synthetic data
- CPU (NumPy) + GPU (CUDA) implementations

### AI
- Compact CNN (PyTorch)
- Binary SAFE vs UNSAFE classification
- Real-time inference
- Risk threshold tuning

### Dashboard
- Live heatmap visualization
- Probability meter
- History plot
- Automatic alarm sound when unsafe
- Start/Stop/Reset controls

---

## GPU Acceleration

Benchmark on RTX 3050:

| Mode | Time |
|-------|---------|
| CPU (NumPy) | 33.3s |
| GPU (CUDA) | 1.09s |

**30.6× speedup**

The diffusion solver uses:
- torch.conv2d
- CUDA tensors
- cuDNN acceleration

---

## Tech Stack

- Python
- PyTorch (CUDA/cuDNN)
- Streamlit
- NumPy
- Matplotlib

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

---

## Project Structure

```
simulation.py        # CPU + GPU diffusion solvers
train_model.py       # CNN training
data_gen.py          # synthetic dataset generation
streamlit_dashboard.py  # real-time demo
benchmark.py         # CPU vs GPU speed test
```
## Deployment

Deployed using Docker containers on Oracle Kubernetes Engine with LoadBalancer service.

---

## Why This Project?

Designed to demonstrate:

- GPU computing (CUDA)
- Simulation pipelines
- Synthetic data generation
- ML inference systems
- Real-time visualization

Skills aligned with robotics, simulation, and accelerated computing roles.

---

## Author
Ashir Akhter
Computer Science @ York University
