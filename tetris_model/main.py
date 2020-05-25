import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


def get_scaled_value(list_to_scale, value_to_scale):
    min_value = min(list_to_scale)
    max_value = max(list_to_scale)
    return (value_to_scale - min_value) / (max_value - min_value)


def get_model(x, a):
    y = 0
    index = 0
    for a_single in a:
        y += a_single * (x ** index)
        index += 1
    return y


age = np.asarray([21, 12, 17, 22, 46, 34, 45, 44, 19, 14, 43, 47])
points = np.asarray([6720, 744, 3328, 2000, 7236, 3896, 5676, 4796, 2016, 744, 5144, 6452])

scaled_age = np.asarray([get_scaled_value(age, value) for value in age])
scaled_points = np.asarray([get_scaled_value(points, value) for value in points])

plt.plot(scaled_age, scaled_points, '.', markersize=10)


X = np.vstack([np.ones_like(scaled_age), scaled_age, scaled_age ** 2, scaled_age ** 3, scaled_age ** 4,
               scaled_age ** 5, scaled_age ** 6, scaled_age ** 7])
Y = np.vstack([scaled_points])


a_opt = inv(X @ X.T) @ X @ Y.T
a_opt_regularized = inv(X @ X.T + 5 * np.identity(len(X))) @ X @ Y.T


xx = np.linspace(min(scaled_age), max(scaled_age), 100)
calculated_model = get_model(xx, a_opt)
calculated_model_regularized = get_model(xx, a_opt_regularized)


plt.plot(scaled_age, scaled_points, '.', markersize=10, label='Measurements')
plt.plot(xx, calculated_model, label='Calculated model')
plt.plot(xx, calculated_model_regularized, label='Calculated model with regularization')

plt.title('Points and age of players in a Tetris-like game')
plt.xlabel('Age')
plt.ylabel('Points')
plt.legend()
plt.show()
