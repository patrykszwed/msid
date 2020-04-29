import matplotlib.pyplot as plt
import numpy as np

N = 10
a, b = 1, 0.5  # parameters of the model
x_data = np.random.uniform(0, 1, size=N)

"""
Perfect data
"""
# y_data = a * x_data + b # the perfect data


"""
Data with noise
"""
standard_deviation = 0.1
noise = np.random.normal(0, standard_deviation, size=N)
y_data = a * x_data + b + noise

plt.plot(x_data, y_data, '.', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
