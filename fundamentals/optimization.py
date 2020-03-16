import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

sunglasses = np.array([110, 126, 150, 152, 190, 173, 183, 175, 221, 234, 217, 235])
ice_cream = np.array([185, 215, 325, 332, 406, 408, 412, 421, 445, 522, 544, 614])
dataset = sunglasses, ice_cream

x = np.linspace(start=min(sunglasses), stop=max(sunglasses), num=100)


def linear_model(x, pars):
    a, b = pars   # unpack model parameters
    return a*x + b    # any equation you dream about


"""
Prediction error calculation
"""


def Q(params_tuple, model, dataset):
    xdata, ydata = dataset     # unpack dataset
    ypred = model(xdata, params_tuple)  # calculate model predictions for all data
    return sum(abs(ydata - ypred))


def Q_squares(params_tuple, model, dataset):
    xdata, ydata = dataset     # unpack dataset
    ypred = model(xdata, params_tuple)  # calculate model predictions for all data
    return sum((ydata - ypred)**2)


params_init = np.random.uniform(-1, 1, size=2)
params_best = fmin(Q, params_init, args=(linear_model, dataset))
params_best_squares = fmin(Q_squares, params_init, args=(linear_model, dataset))

plt.plot(sunglasses, ice_cream, 'o', label='data')
plt.ylim(0, max(ice_cream) + 50)
plt.plot(x, linear_model(x, params_best), label='model mean')
plt.plot(x, linear_model(x, params_best_squares), 'r', label='model squares')

plt.xlabel('Sunglasses sold [$]', fontsize=14)
plt.ylabel('Ice cream sales', fontsize=14)
plt.legend(loc='lower right')
# see the file in the current directory!
plt.savefig('my_fig.png', dpi=300)
plt.show()


print('The best solution for mean is: ', params_best,
      ',\nand corresponding value of the objective function is: ',
      Q(params_best, linear_model, dataset))

print('The best solution for squares is: ', params_best_squares,
      ',\nand corresponding value of the objective function is: ',
      Q(params_best_squares, linear_model, dataset))
