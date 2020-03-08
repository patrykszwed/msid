import numpy as np
n = 10
x = np.zeros(n)
x[0] = 1
x[1] = 1
for i in range(1, n - 1):
    x[i + 1] = x[i] + x[i - 1]
print(x)
