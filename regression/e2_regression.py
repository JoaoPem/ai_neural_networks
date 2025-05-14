import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('aerogerador.dat')
X = data[:, 0]
y = data[:, 1]

plt.scatter(X, y, s=1)
plt.xlabel('Wind Speed')
plt.ylabel('Generated Power')
plt.title('Scatter Plot of Wind Speed vs Power')
plt.show()
