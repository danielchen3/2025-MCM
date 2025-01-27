# Test for optimization

import math

from scipy.optimize import differential_evolution, dual_annealing

import numpy as np

from config import P


def heuristic(rate, t=0):
    if t == 0:
        return math.tanh(rate)
    else:
        return math.tanh(rate)


def satisfaction_factor(N, P, ratio, t=0):
    get_heruistic = heuristic(ratio, t)
    S_nrt = (1 - N / P) * get_heruistic * (1 - get_heruistic)
    return S_nrt


def get_obj1(params, t=0):
    import config
    tourist_ratio, m1, m2, rate = params
    get_heruistic = heuristic(rate, t)
    # global P
    # print("------优化在get_obj1函数中------")
    # print(f"P is {config.P}")
    N = float(config.P) * tourist_ratio
    S_nrt = satisfaction_factor(N, config.P, tourist_ratio, t)
    obj1 = N * (m1 * get_heruistic + m2 * (1 - get_heruistic)) * (1 + S_nrt)
    return obj1


def neg_get_obj1(params, t=0):
    import config
    tourist_ratio, m1, m2, rate = params
    get_heruistic = heuristic(rate, t)
    N = float(config.P) * tourist_ratio
    S_nrt = satisfaction_factor(N, config.P, tourist_ratio, t)
    obj1 = N * (m1 * get_heruistic + m2 * (1 - get_heruistic)) * (1 + S_nrt)
    return -obj1

