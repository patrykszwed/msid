import matplotlib.pyplot as plt
import numpy as np

Npoints = 1000
x_min, x_max, y_min, y_max = -3, 3, -3, 3

x_u = np.random.uniform(x_min, x_max, size=Npoints)
y_u = np.random.uniform(y_min, y_max, size=Npoints)

x_n = np.random.normal(0, 1, size=Npoints)
y_n = np.random.normal(0, 1, size=Npoints)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(x_u, y_u, '.', markersize=5)
ax[0].set_xlim([x_min - 1, x_max + 1])
ax[0].set_ylim([y_min - 1, y_max + 1])

ax[1].plot(x_n, y_n, '.', markersize=5)
ax[1].set_xlim([x_min - 1, x_max + 1])
ax[1].set_ylim([y_min - 1, y_max + 1])

plt.show()
