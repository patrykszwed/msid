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

N = 10
# a, b = 1, 0.5  # parametry modelu (nie pomyl tego a z symbolem we wzorze powyżej)
a = [1, 0.5, 0.75, 0.31]
r = 0.1
z = np.random.normal(0, r, size=N)
xdata = np.random.uniform(0, 1, size=N)
# ydata = a * xdata + b + z  # model idealny z nałożonym szumem
ydata = a[3] * xdata ** 3 + a[2] * xdata ** 2 + a[1] * xdata ** 1 + a[
    0] * xdata ** 0  # model idealny z nałożonym szumem
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
# yy = a_opt[1] * xx + a_opt[0]
# print('yy', yy)
yy2 = model(xx, a_opt)
print('yy2', yy2)

plt.plot(xdata, ydata, '.', markersize=10, label='pomiary')
# plt.plot(xx, yy, label='model wyliczony')
plt.plot(xx, yy2, label='model wyliczony v2')
plt.plot(xx, a[3] * xx ** 3 + a[2] * xx ** 2 + a[1] * xx ** 1 + a[
    0] * xx ** 0, label='model idealny')
# a[3] * xdata ** 3 + a[2] * xdata ** 2 + a[1] * xdata ** 1 + a[
#     0] * xdata ** 0

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
