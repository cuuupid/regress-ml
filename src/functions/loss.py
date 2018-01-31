import math

def sum_squared_error(p, e):
    err = 0
    for prediction, expected in zip(p, e):
        err += abs(prediction - expected)**2
    return err

def sum_error(p, e):
    err = 0
    for prediction, expected in zip(p, e):
        err += abs(prediction - expected)
    return err
