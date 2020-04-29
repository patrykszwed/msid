import numpy as np
from numpy.linalg import inv

"""
Watch out for assignment operator in Python!
"""
A = np.array([[1, 2, 3], [4, 5, 6]])
print('A', A)

B = A.T
print('B', B)

B[1, 1] = 100

print('B', B)
print('A', A)

"""
Correct way of assignment
"""
print('CORRECT WAY OF ASSIGNMENT')
A = np.array([[1, 2, 3], [4, 5, 6]])
print('A', A)

B = A.T.copy()
print('B', B)

B[1, 1] = 100

print('B', B)
print('A', A)

"""
Matrix multiplication
"""
print('Matrix multiplication')
A = np.array([[1, 2, 3], [4, 5, 6]])
print('A', A)

B = A.T.copy()
print('B', B)

C = np.dot(B, A)
print('C', C)
D = B @ A  # this is the same as np.dot()
print('D', D)

E = A * A  # this is element-wise operation. Each row is squared, so before (0,1) was 2, after A * A it is 4
print('E', E)

"""
Matrix inversion
"""
print('Matrix inversion')
A = np.array([[1, 2, 3], [4, 5, 6]])
print('A', A)

B = A.T.copy()
print('B', B)

C = A @ B
print('C', C)
D = inv(C)
print('D', D)

# Todo HOMEWORK PART!

"""
Rearranging matrices
"""
print('Rearranging matrices')
tab = np.arange(start=0, stop=25, step=1).reshape(5, 5)
print('tab', tab, end='\n\n')
print('tab.flatten()', tab.flatten(), end='\n\n')
print('tab.reshape(1, -1)', tab.reshape(1, -1), end='\n\n')
print('tab.reshape(-1, 1)', tab.reshape(-1, 1), end='\n\n')

x = np.arange(0, 5)
print('x', x)
x_vstack = np.vstack([x, x])
print('x_vstack', x_vstack)
x_hstack = np.hstack([x, x])
print('x_hstack', x_hstack)
print('x', x)
print('x.T', x.T)  # one dimensional arrays are not affected by transpose. You have to use reshape to get 2-D matrix.

x = np.arange(1, 13)
print('x', x)
x_reshaped = x.reshape(3, 4, order='F')
print('x_reshaped', x_reshaped)

"""
Array dimensions
"""
# Todo
