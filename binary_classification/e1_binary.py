import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Spiral3d.csv', delimiter=',', skip_header=1)

X = data[:, :3]
y = data[:, 3].astype(int)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(y):
    indices = y == label
    ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], label=f'Class {label}', alpha=0.6)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Spiral3D Dataset Visualization by Class')
ax.legend()
plt.show()