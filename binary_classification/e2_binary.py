import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Spiral3d.csv', delimiter=',', skip_header=1)

X = data[:, :3]
y = data[:, 3].astype(int)

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X[idx, 0], X[idx, 1], label=f'Class {label}', alpha=0.6)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Scatter Plot of Spiral3D Dataset (x1 vs x2)')
plt.legend()
plt.grid(True)
plt.show()