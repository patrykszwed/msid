import numpy as np

my_list = [56, 72, 86, 63, 45, 90, 12]
min_value = min(my_list)
max_value = max(my_list)


def get_scaled_value(value_to_scale):
    return (value_to_scale - min_value) / (max_value - min_value)


print('Scaled value =', get_scaled_value(14))


a = np.arange(5)
b = a

print(a)
print(b)
a[0] = 5
print(a)
print(b)
b[1] = 7
print(a)
print(b)
