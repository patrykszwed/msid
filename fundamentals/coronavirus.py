import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
import math
from coronavirus_response import response

"""
Sum of squared errors (function for genetic algorithm to minimize)
"""


def sum_of_squared_errors(params_tuple, model, x_data, y_data):
    y_pred = model(x_data, params_tuple)  # calculate model predictions for all data
    # print('y_pred', y_pred)
    # print('y_data', y_data)
    return np.sum((y_data - y_pred) ** 2.0)


def sigmoid(x, params):
    a, b, c = params
    y = a / (1 + np.exp(-b * (x - c)))
    return y


def main():
    total_cases = np.asarray([day['Cases'] for day in response])
    days = np.asarray([i for i in range(1, len(total_cases) + 1)])
    # logs_x = np.asarray([np.log(x) for x in days])
    # logs_y = np.asarray([np.log(y) for y in total_cases])
    logs_x = days
    logs_y = total_cases
    print('logs_y', logs_y)
    # print('logs_x', logs_x)
    plt.figure(figsize=(15, 5))
    # plt.plot(days, np.exp(logs_y), '-o', label='measured data')
    plt.plot(days, logs_y, '-o', label='measured data')
    # plt.ylim([0, 13000])
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Total Coronavirus Cases', fontsize=14)
    plt.title('Total Cases in Poland', fontsize=16)

    # determine the range for x axis
    params_init = np.random.uniform(-1, 1, size=3)
    # print('params_init', params_init)
    params_best = fmin(sum_of_squared_errors, params_init, args=(sigmoid, logs_x, logs_y))
    print('1st sum_of_squared_errors = ', sum_of_squared_errors(params_best, sigmoid, logs_x, logs_y))
    params_best = [11.04935003, 0.32165335, 4.97391684]
    print('2nd sum_of_squared_errors = ', sum_of_squared_errors(params_best, sigmoid, logs_x, logs_y))
    print('params_best', params_best)
    # x = np.linspace(start=min(logs_y), stop=max(logs_y), num=len(total_cases))
    x = np.linspace(start=min(logs_y), stop=max(logs_y), num=len(total_cases))
    # print('x', x)
    y = sigmoid(x, params_best)
    print('sigmoid', y)
    
    y_copy = sigmoid(x, params_best)
    # print('y_copy', np.exp(y_copy))
    print('after y_copy', y_copy)
    # plt.plot(days, np.exp(sigmoid(x, params_best)), '-o', label='naive model')
    plt.plot(days, sigmoid(x, params_best), '-o', label='naive model')

    plt.legend()
    plt.show()


main()
