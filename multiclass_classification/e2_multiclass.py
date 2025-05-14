import numpy as np

data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', dtype=str, skip_header=1)

classes = data[:, -1]

Y = np.zeros((3, len(classes)))

for i, label in enumerate(classes):
    if label == 'NO':
        Y[:, i] = [1, -1, -1]
    elif label == 'DH':
        Y[:, i] = [-1, 1, -1]
    elif label == 'SL':
        Y[:, i] = [-1, -1, 1]

print("Formato de Y com dimensão ℝ^{C x N}:", Y.shape)