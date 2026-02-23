import numpy as np

files = [
    "dataset/dataset_part1.npz",
    "dataset/dataset_part2.npz",
    "dataset/dataset_part3.npz"
]

xs, ys = [], []

for f in files:
    d = np.load(f, allow_pickle=True)
    xs.append(d["x"])
    ys.append(d["y"])

x = np.concatenate(xs)
y = np.concatenate(ys)

np.savez_compressed("dataset/dataset.npz", x=x, y=y)

print("Merged shape:", x.shape)