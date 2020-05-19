import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


def model(x, a):
    y = 0
    print('xx', x)
    print('a', a)
    index = 0
    for a_single in a:
        print('a_single', a_single)
        print('index', index)
        y += a_single * (x ** index)
        index += 1
    return y


# yy = a_opt[1] * xx + a_opt[0]

N = 500
a, b = 1, 0.5  # parametry modelu (nie pomyl tego a z symbolem we wzorze powyżej)
r = 0.1
z = np.random.normal(0, r, size=N)
xdata = np.random.uniform(0, 1, size=N)
ydata = a * xdata + b + z  # model idealny z nałożonym szumem
plt.plot(xdata, ydata, '.', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

X = np.vstack([np.ones_like(xdata), xdata])
print('X', X)
print(X.shape)
Y = np.vstack([ydata])
# Y = np.asarray([ydata])
print(Y.shape)
print('Y', Y)
a_opt = inv(X @ X.T) @ X @ Y.T
print('a_opt', a_opt)
xx = np.linspace(min(xdata), max(xdata), 100)
yy = a_opt[1] * xx + a_opt[0]
print('yy', yy)
yy2 = model(xx, a_opt)
print('yy2', yy2)

plt.plot(xdata, ydata, '.', markersize=10, label='pomiary')
plt.plot(xx, yy, label='model wyliczony')
plt.plot(xx, yy2, label='model wyliczony v2')
plt.plot(xx, a * xx + b, label='model idealny')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
