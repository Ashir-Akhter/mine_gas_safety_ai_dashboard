from simulation import create_base_layout, init_state, step_cpu

layout = create_base_layout()
gas, fans, leaks = init_state(layout)

for _ in range(20):
    gas = step_cpu(layout, gas, fans, leaks)

print("max gas after 20 steps:", gas.max())