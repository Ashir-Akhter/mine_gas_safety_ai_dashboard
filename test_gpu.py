import time
from simulation import create_base_layout, init_state, step_cpu, to_torch, step_gpu
from simulation import step_gpu_many
import torch

STEPS = 8000

layout = create_base_layout(512, 512)
gas, fans, leaks = init_state(layout)

# -------------------------
# CPU
# -------------------------
start = time.time()
g = gas.copy()
for _ in range(STEPS):
    g = step_cpu(layout, g, fans, leaks)
cpu_time = time.time() - start


# -------------------------
# GPU
# -------------------------
layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks)

# warmup
for _ in range(50):
    step_gpu(layout_t, gas_t, fans_t, leaks_t)
torch.cuda.synchronize()

start = time.time()

g = step_gpu_many(layout_t, gas_t, fans_t, leaks_t, STEPS)

torch.cuda.synchronize()
gpu_time = time.time() - start


print("CPU time:", cpu_time)
print("GPU time:", gpu_time)
print("Speedup:", cpu_time/gpu_time)