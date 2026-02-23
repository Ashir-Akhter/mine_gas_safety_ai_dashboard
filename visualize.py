import matplotlib.pyplot as plt
from simulation import create_base_layout, init_state, to_torch, step_gpu
import torch

layout = create_base_layout(256, 256)
gas, fans, leaks = init_state(layout)

layout_t, gas_t, fans_t, leaks_t = to_torch(layout, gas, fans, leaks)

plt.ion()  # interactive mode

fig, ax = plt.subplots()
img = ax.imshow(gas_t.cpu().numpy(), origin="lower", vmin=0, vmax=1)
plt.colorbar(img)

for step in range(5000):

    gas_t = step_gpu(layout_t, gas_t, fans_t, leaks_t)

    if step % 5 == 0:
        img.set_data(gas_t.cpu().numpy())
        ax.set_title(f"Step {step}")
        plt.pause(0.001)

plt.ioff()
plt.show()