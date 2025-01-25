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
    tourist_ratio, m1, m2, rate = params
    get_heruistic = heuristic(rate, t)
    N = float(P) * tourist_ratio
    S_nrt = satisfaction_factor(N, P, tourist_ratio, t)
    obj1 = N * (m1 * get_heruistic + m2 * (1 - get_heruistic)) * (1 + S_nrt)
    return obj1

def neg_get_obj1(params, t=0):
    tourist_ratio, m1, m2, rate = params
    get_heruistic = heuristic(rate, t)
    N = float(P) * tourist_ratio
    S_nrt = satisfaction_factor(N, P, tourist_ratio, t)
    obj1 = N * (m1 * get_heruistic + m2 * (1 - get_heruistic)) * (1 + S_nrt)
    return -obj1


def gradient_descent(learning_rate=0.01, max_iters=1000, tol=1e-6):
    # 初始值
    params = np.array([1.2, 500, 10, 0.1])  # tourist_ratio, m1, m2, rate
    prev_obj1 = float("-inf")

    for i in range(max_iters):
        # 计算目标函数的梯度（近似）
        grad = np.zeros_like(params)
        for j in range(len(params)):
            # 计算目标函数对每个参数的偏导数（数值微分）
            params_plus = params.copy()
            params_plus[j] += tol
            grad[j] = (get_obj1(params_plus) - get_obj1(params)) / tol

        # 更新参数
        params += learning_rate * grad

        # 打印每次迭代的目标函数值
        obj1 = get_obj1(params)
        print(f"Iteration {i+1}: obj1 = {obj1}, params = {params}")

        # 如果目标函数值变化小于 tol，则停止迭代
        if abs(obj1 - prev_obj1) < tol:
            print(f"Converged after {i+1} iterations.")
            break

        prev_obj1 = obj1

    return params
