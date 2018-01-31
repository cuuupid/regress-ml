import math

def sum_squared_error(p, e):
    err = 0
    for prediction, expected in zip(p, e):
        _err = 0
        for _p, _e in zip(prediction, expected):
            _err += abs(_p - _e)
        err += _err**2
    return err

def sum_error(p, e):
    err = 0
    for prediction, expected in zip(p, e):
        for _p, _e in zip(prediction, expected):
            err += abs(_p - _e)
    return err
