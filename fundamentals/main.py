import numpy as np
import matplotlib.pyplot as plt

sunglasses = np.array([110, 126, 150, 152, 190, 173, 183, 175, 221, 234, 217, 235])
ice_cream = np.array([185, 215, 325, 332, 406, 408, 412, 421, 445, 522, 544, 614])

# determine the range for x axis
x = np.linspace(start=min(sunglasses), stop=max(sunglasses), num=100)
print(x)

a, b = 3, 1  # model parameters
params_tuple = a, b  # pack model parameters into the tuple params_tuple

print(a)
print(b)
print(params_tuple)


def linear_model(x, params_tuple):
    a, b = params_tuple  # unpack model parameters
    return a * x + b  # any equation you dream about xD


plt.plot(sunglasses, ice_cream, 'o', label='data')
plt.ylim(0, max(ice_cream) + 50)

plt.plot(x, linear_model(x, params_tuple), label='model')
plt.xlabel('Sunglasses sold [$]', fontsize=14)
plt.ylabel('Ice cream sales', fontsize=14)
plt.legend(loc='lower right')

print(type(sunglasses))
print(type(ice_cream))
dataset = sunglasses, ice_cream  # pack data (type tuple)
print(type(dataset))
print(type(dataset[0]))
print(type(dataset[1]))

"""
Function calculates prediction error.
Prediction error may be calculated as the differences between predictions of the model
and the measurements from the dataset, added up together
"""


def Q(params_tuple, model, dataset):
    xdata, ydata = dataset  # unpack dataset
    ypred = model(xdata, params_tuple)  # calculate model predictions
    return sum(abs(ydata - ypred))


def my_func(params_tuple, linear_model, dataset):
    q_final = Q(params_tuple, linear_model, dataset)
    a, b = params_tuple
    a_temp = 0
    b_temp = 0
    for i in range(100):
        a_temp += 0.1
        params_temp = a_temp, b_temp
        q_temp = Q(params_temp, linear_model, dataset)
        if q_final > q_temp:
            q_final = q_temp
            a = a_temp

    a_temp = 0
    b_temp = 0
    for i in range(100):
        b_temp += 0.1
        params_temp = a_temp, b_temp
        q_temp = Q(params_temp, linear_model, dataset)
        if q_final > q_temp:
            q_final = q_temp
            a = a_temp

    a_temp = 0
    b_temp = 0
    for i in range(100):
        a_temp += 0.1
        b_temp += 0.1
        params_temp = a_temp, b_temp
        q_temp = Q(params_temp, linear_model, dataset)
        if q_final > q_temp:
            q_final = q_temp
            a = a_temp

    print('The sum of errors is:', q_final)
    params_tuple = a, b
    return params_tuple


params_tuple = my_func(params_tuple, linear_model, dataset)
plt.plot(x, linear_model(x, params_tuple), label='model', color='r')
plt.show()
