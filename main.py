from income import *

from config import *

from environment import (
    get_obj2,
    neg_get_obj2,
    get_obj2_2,
    get_obj2_get1,
    get_obj2_get2,
    get_obj2_get3,
)

from res_satisfaction import get_obj3, neg_get_obj3

from scipy.optimize import NonlinearConstraint, LinearConstraint
from obj4 import get_obj4, neg_get_obj4


def max_constraint(params, t=0):
    (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        B3,
        B4,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    # 返回一个数组，表示x应该大于等于其他所有值
    return np.array(
        [
            alpha_g - alpha_w,  # x >= a
            alpha_g - alpha_r,  # x >= b
            alpha_g - alpha_infra,  # x >= b
        ]
    )


max_nlc = NonlinearConstraint(max_constraint, 0, np.inf)


def max_constraint_obj2(params, t=0):
    (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    # 返回一个数组，表示x应该大于等于其他所有值
    return np.array(
        [
            alpha_g - alpha_w,  # x >= a
            alpha_g - alpha_r,  # x >= b
            alpha_g - alpha_infra,  # x >= b
        ]
    )


max_nlc_obj2 = NonlinearConstraint(max_constraint_obj2, 0, np.inf)


# 非线性约束
def nonlinear_constraint(params, t=0):
    (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        B3,
        B4,
        # k1,
        # k2,
        # k3,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    N = P * tourist_ratio
    K = 4 * beta * heuristic(rate, t) * (1 - heuristic(rate, t))

    left = N / np.exp(K * (1 - N / P))
    right = N0 / np.exp(beta * (rate + alpha_g + 1))
    return left - right  # 返回0表示等式约束


nlc = NonlinearConstraint(nonlinear_constraint, 0.0, 0.0)  # 等式约束


bounds_1 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
]  # tourist_ratio, m1, m2, rate

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution

bounds_2 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
]

A = np.array(
    [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
)  # B11 + B12 + B13 + B2 + B3 <= 1

linear_constraint = LinearConstraint(A, -np.inf, 1.0)

alpha_cons = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

alpha_linear_constraint = LinearConstraint(alpha_cons, -np.inf, 1.0)

alpha_cons_obj2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

alpha_linear_constraint_obj2 = LinearConstraint(alpha_cons_obj2, -np.inf, 1.0)


bounds_3 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B3_min, B3_max),
    # (k_min, k_max),
    # (k_min, k_max),
    # (k_min, k_max),
]
bounds_4 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B4_min, B4_max),
]


def get_objs_min_max():

    result = differential_evolution(get_obj1, bounds_1, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj1 = result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最小化的 obj1 值: {optimal_obj1}")

    obj1_max = -optimal_obj1

    result = differential_evolution(neg_get_obj1, bounds_1, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj1 = -result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最大化的 obj1 值: {optimal_obj1}")

    obj1_min = -optimal_obj1

    result = differential_evolution(
        get_obj2,
        bounds_2,
        constraints=[alpha_linear_constraint_obj2, max_nlc_obj2],  # 设置约束
        args=(0,),
        maxiter=10000,
    )

    optimal_params = result.x
    optimal_obj2 = result.fun

    print(f"Optimized Parameters: {optimal_params}")
    print(f"Minimized OBJ2: {optimal_obj2}")

    obj2_min = optimal_obj2

    result = differential_evolution(
        neg_get_obj2,
        bounds_2,
        constraints=[alpha_linear_constraint_obj2, max_nlc_obj2],  # 设置约束
        args=(0,),
        maxiter=10000,
    )

    optimal_params = result.x
    optimal_obj2 = -result.fun

    print(f"Optimized Parameters: {optimal_params}")
    print(f"Maximized OBJ2: {optimal_obj2}")

    obj2_max = optimal_obj2

    result = differential_evolution(get_obj3, bounds_3, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj3 = result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最小化的 obj3 值: {optimal_obj3}")

    obj3_max = -optimal_obj3

    result = differential_evolution(neg_get_obj3, bounds_3, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj3 = -result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最大化的 obj3 值: {optimal_obj3}")

    obj3_min = -optimal_obj3

    ## Obj4

    result = differential_evolution(get_obj4, bounds_4, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj4 = result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最小化的 obj4 值: {optimal_obj4}")

    obj4_max = -optimal_obj4

    result = differential_evolution(neg_get_obj4, bounds_4, args=(0,), maxiter=10000)

    # output the result
    optimal_params = result.x
    optimal_obj4 = -result.fun

    print(f"最优参数: {optimal_params}")
    print(f"最大化的 obj4 值: {optimal_obj4}")

    obj4_min = -optimal_obj4

    return (
        obj1_min,
        obj1_max,
        obj2_min,
        obj2_max,
        obj3_min,
        obj3_max,
        obj4_min,
        obj4_max,
    )


bounds_final = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B3_min, B3_max),
    (k_min, k_max),
    (k_min, k_max),
    (k_min, k_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
]

bounds_final_sensitivity = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B3_min, B3_max),
    (B4_min, B4_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
]

# 用于记录每代的最优目标值
obj1_values = []
obj2_values = []
obj3_values = []


# 用于绘制目标值的变化
def record_objective_values(xk, convergence):
    # 获取当前最优目标值
    params_obj1 = xk[:4]  # 假设前4个参数属于obj1
    params_obj2 = np.concatenate((xk[:8], xk[12:16]))
    params_obj3 = np.concatenate((xk[:4], xk[8:12]))  # 假设后4个参数属于obj3

    # 获取每个目标的值
    obj1_raw = -get_obj1(params_obj1, 0)  # 最大化转最小化
    obj2_raw = get_obj2(params_obj2, 0)  # 最小化
    obj3_raw = -get_obj3(params_obj3, 0)  # 最大化转最小化

    # 设定每个目标的最大最小值（需要根据你的实际情况调整）
    # obj1_min, obj1_max = -17492684.54, -17824.19
    # obj2_min, obj2_max = 0.0, 17835375.23
    # obj3_min, obj3_max = -30000.94, -1062.22

    obj1_min, obj1_max = -35795272.07, -99940.09
    obj2_min, obj2_max = 0.0, 35795272.08
    obj3_min, obj3_max = -30000.94, -21000.85

    # 归一化
    obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
    obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
    obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

    # 记录当前代的目标值
    obj1_values.append(obj1_norm)
    obj2_values.append(obj2_norm)
    obj3_values.append(obj3_norm)


# 用回调记录每代最优解的目标值
# callback = record_objective_values

# B1i 数量
# alpha 之间的限制条件


def get_obj_final(params, t=0):
    import config

    # print(f"P is {P}")
    (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        B3,
        B4,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    params_obj1 = tourist_ratio, m1, m2, rate
    params_obj2 = (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    )
    params_obj3 = tourist_ratio, m1, m2, rate, B3
    params_obj4 = tourist_ratio, m1, m2, rate, B4
    return get_normalized_objectives(
        params_obj1, params_obj2, params_obj3, params_obj4, t
    )


def normalize_objective(value, min_val, max_val):
    """归一化函数，将值映射到[0,1]区间"""
    from config import P, R, A, N0

    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


obj4_datas = []
obj4_real_datas = []

t1 = []
t2 = []
t3 = []
t4 = []

iteration_nums = []
iter_count = 0


def get_normalized_objectives(params_obj1, params_obj2, params_obj3, params_obj4, t):
    # 获取原始值
    import config

    global iter_count

    # print(f"P is {P}")
    obj1_raw = -get_obj1(params_obj1, t)  # 最大化转最小化
    obj2_raw = get_obj2(params_obj2, t)  # 最小化
    obj3_raw = -get_obj3(params_obj3, t)  # 最大化转最小化
    obj4_raw = -get_obj4(params_obj4, t)

    print("------------raw_data----------------")
    # print(f"P is {config.P}")
    print(f"obj1 is {obj1_raw}")
    print(f"obj2 is {obj2_raw}")
    print(f"obj3 is {obj3_raw}")
    print(f"obj4 is {obj4_raw}")

    import config

    # print("------------new min max----------------")
    # print(f"obj1_min is {obj1_min}, obj1_max is {obj1_max}")
    # print(f"obj2_min is {obj2_min}, obj2_max is {obj2_max}")
    # print(f"obj3_min is {obj3_min}, obj3_max is {obj3_max}")

    obj1_min_local, obj1_max_local = config.obj1_min, config.obj1_max
    obj2_min_local, obj2_max_local = config.obj2_min, config.obj2_max
    obj3_min_local, obj3_max_local = config.obj3_min, config.obj3_max
    obj4_min_local, obj4_max_local = config.obj4_min, config.obj4_max

    print("------------new min max----------------")
    print(f"obj1_min is {obj1_min_local}, obj1_max is {obj1_max_local}")
    print(f"obj2_min is {obj2_min_local}, obj2_max is {obj2_max_local}")
    print(f"obj3_min is {obj3_min_local}, obj3_max is {obj3_max_local}")
    print(f"obj4_min is {obj4_min_local}, obj4_max is {obj4_max_local}")

    # 归一化
    obj1_norm = normalize_objective(obj1_raw, obj1_min_local, obj1_max_local)
    obj2_norm = normalize_objective(obj2_raw, obj2_min_local, obj2_max_local)
    obj3_norm = normalize_objective(obj3_raw, obj3_min_local, obj3_max_local)
    obj4_norm = normalize_objective(obj4_raw, obj4_min_local, obj4_max_local)

    print("------------normalized_data----------------")
    # print(f"P is {config.P}")
    print(f"obj1 is {obj1_norm}")
    print(f"obj2 is {obj2_norm}")
    print(f"obj3 is {obj3_norm}")
    print(f"obj4 is {obj4_norm}")

    # obj4_datas.append(obj4_norm)
    # obj4_real_datas.append(-obj4_raw)
    # tt1 = -obj4_raw / (-obj4_raw + get_obj2_2(params_obj2, t))
    # tt2 = get_obj2_get1(params_obj2, t) / (-obj4_raw + get_obj2_2(params_obj2, t))
    # tt3 = get_obj2_get2(params_obj2, t) / (-obj4_raw + get_obj2_2(params_obj2, t))
    # tt4 = get_obj2_get3(params_obj2, t) / (-obj4_raw + get_obj2_2(params_obj2, t))

    # if tt1 > 1:
    #     t1.append(1)
    # elif tt1 < 0:
    #     t1.append(0)
    # else:
    #     t1.append(tt1 * 0.)

    # if tt2 > 1:
    #     t2.append(1)
    # elif tt2 < 0:
    #     t2.append(0)
    # else:
    #     t2.append(tt2)

    # if tt3 > 1:
    #     t1.append(1)
    # elif tt3 < 0:
    #     t3.append(0)
    # else:
    #     t3.append(tt3)

    # if tt4 > 1:
    #     t4.append(1)
    # elif tt4 < 0:
    #     t4.append(0)
    # else:
    #     t4.append(tt4)
    # iteration_nums.append(iter_count)
    # iter_count += 1

    # 添加权重（可选）
    w1, w2, w3, w4 = 1 / 4, 1 / 4, 1 / 4, 1 / 4  # 权重系数

    # print(
    #     f"obj1_norm is {obj1_norm}, obj2_norm is {obj2_norm}, obj3_norm is {obj3_norm}"
    # )
    final_result = w1 * obj1_norm + w2 * obj2_norm + w3 * obj3_norm + w4 * obj4_norm
    # print(f"final_result is {final_result}")

    return final_result


param_history = {
    "B11": [],
    "B12": [],
    "B13": [],
    "B4": [],
}

B11_history = []
B12_history = []
B13_history = []
B4_history = []


def callback(xk, convergence):
    """优化过程的回调函数，记录每次迭代的目标函数值"""
    # 解包参数
    global iter_count
    (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        B3,
        B4,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = xk
    params_obj2 = (
        tourist_ratio,
        m1,
        m2,
        rate,
        B11,
        B12,
        B13,
        B2,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    )
    params_obj4 = tourist_ratio, m1, m2, rate, B4
    
    obj4_raw = -get_obj4(params_obj4, 0)

    tt1 = -obj4_raw / (-obj4_raw + get_obj2_2(params_obj2, 0))
    tt2 = get_obj2_get1(params_obj2, 0) / (-obj4_raw + get_obj2_2(params_obj2, 0))
    tt3 = get_obj2_get2(params_obj2, 0) / (-obj4_raw + get_obj2_2(params_obj2, 0))
    tt4 = get_obj2_get3(params_obj2, 0) / (-obj4_raw + get_obj2_2(params_obj2, 0))

    if tt1 > 1:
        t1.append(1)
    elif tt1 < 0:
        t1.append(0)
    else:
        t1.append(tt1)

    if tt2 > 1:
        t2.append(1)
    elif tt2 < 0:
        t2.append(0)
    else:
        t2.append(tt2)

    if tt3 > 1:
        t1.append(1)
    elif tt3 < 0:
        t3.append(0)
    else:
        t3.append(tt3)

    if tt4 > 1:
        t4.append(1)
    elif tt4 < 0:
        t4.append(0)
    else:
        t4.append(tt4)
    iteration_nums.append(iter_count)
    iter_count += 1


def get_final():

    import config
    import scipy.optimize

    result = scipy.optimize.differential_evolution(
        get_obj_final,
        bounds_final_sensitivity,
        constraints=[
            linear_constraint,
            nlc,
            alpha_linear_constraint,
            max_nlc,
        ],  # 设置约束
        args=(0,),
        maxiter=5000,
        # tol=1e-6,
        callback=callback,
    )

    optimal_params = result.x
    optimal_final = result.fun

    print(f"Optimized Parameters: {optimal_params}")
    print(f"Minimized Final: {optimal_final}")

    return optimal_params, optimal_final


# get_final()

# get_final()

# import numpy as np
# from scipy.signal import savgol_filter


# # 平滑化函数
# def smooth_data(data, window=5):
#     return savgol_filter(data, window, 3)  # window需要是奇数


# plt.figure(figsize=(12, 6))

# # 原始数据（半透明）
# # plt.plot(obj1_values, 'b--', alpha=0.3, label='Original Obj1')
# # plt.plot(obj2_values, 'g--', alpha=0.3, label='Original Obj2')
# # plt.plot(obj3_values, 'r--', alpha=0.3, label='Original Obj3')

# # 平滑后的数据
# plt.plot(smooth_data(obj1_values), "b-", label="Obj1 value")
# plt.plot(smooth_data(obj2_values), "g-", label="Obj2 value")
# plt.plot(smooth_data(obj3_values), "r-", label="Obj3 value")

# plt.xlabel("Generation")
# plt.ylabel("Normalized Objective Value")
# plt.legend()
# plt.title("Normalized Objective Values Over Generations")
# plt.grid(True)
# plt.show()


# epsilon = 10
# param_bounds = [
#     (
#         max(
#             optimal_params[0] - ((tourist_ratio_max - tourist_ratio_min) / epsilon),
#             tourist_ratio_min,
#         ),
#         min(
#             optimal_params[0] + ((tourist_ratio_max - tourist_ratio_min) / epsilon),
#             tourist_ratio_max,
#         ),
#     ),
#     (
#         max(optimal_params[1] - ((m1_max - m1_min) / epsilon), m1_min),
#         min(optimal_params[1] + ((m1_max - m1_min) / epsilon), m1_max),
#     ),
#     (
#         max(optimal_params[2] - ((m2_max - m2_min) / epsilon), m2_min),
#         min(optimal_params[2] + ((m2_max - m2_min) / epsilon), m2_max),
#     ),
#     (
#         max(optimal_params[3] - ((rate_max - rate_min) / epsilon), rate_min),
#         min(optimal_params[3] + ((rate_max - rate_min) / epsilon), rate_max),
#     ),
#     (
#         max(optimal_params[4] - ((B_max - B_min) / epsilon), B_min),
#         min(optimal_params[4] + ((B_max - B_min) / epsilon), B_max),
#     ),
#     (
#         max(optimal_params[5] - ((B_max - B_min) / epsilon), B_min),
#         min(optimal_params[5] + ((B_max - B_min) / epsilon), B_max),
#     ),
#     (
#         max(optimal_params[6] - ((B_max - B_min) / epsilon), B_min),
#         min(optimal_params[6] + ((B_max - B_min) / epsilon), B_max),
#     ),
#     (
#         max(optimal_params[7] - ((B_max - B_min) / epsilon), B_min),
#         min(optimal_params[7] + ((B_max - B_min) / epsilon), B_max),
#     ),
#     (
#         max(optimal_params[8] - ((B3_max - B3_min) / epsilon), B3_min),
#         min(optimal_params[8] + ((B3_max - B3_min) / epsilon), B3_max),
#     ),
#     (
#         max(optimal_params[9] - ((k_max - k_min) / epsilon), k_min),
#         min(optimal_params[9] + ((k_max - k_min) / epsilon), k_max),
#     ),
#     (
#         max(optimal_params[10] - ((k_max - k_min) / epsilon), k_min),
#         min(optimal_params[10] + ((k_max - k_min) / epsilon), k_max),
#     ),
#     (
#         max(optimal_params[11] - ((k_max - k_min) / epsilon), k_min),
#         min(optimal_params[11] + ((k_max - k_min) / epsilon), k_max),
#     ),
#     (
#         max(optimal_params[12] - ((alpha_max - alpha_min) / epsilon), alpha_min),
#         min(optimal_params[12] + ((alpha_max - alpha_min) / epsilon), alpha_max),
#     ),
#     (
#         max(optimal_params[13] - ((alpha_max - alpha_min) / epsilon), alpha_min),
#         min(optimal_params[13] + ((alpha_max - alpha_min) / epsilon), alpha_max),
#     ),
#     (
#         max(optimal_params[14] - ((alpha_max - alpha_min) / epsilon), alpha_min),
#         min(optimal_params[14] + ((alpha_max - alpha_min) / epsilon), alpha_max),
#     ),
#     (
#         max(optimal_params[15] - ((alpha_max - alpha_min) / epsilon), alpha_min),
#         min(optimal_params[15] + ((alpha_max - alpha_min) / epsilon), alpha_max),
#     ),
# ]

# 定义灵敏度分析问题
# problem = {
#     "num_vars": 13,  # 假设你有3个输入变量
#     "names": [
#         "tourist_ratio",
#         "m1",
#         "m2",
#         "rate",
#         "B11",
#         "B12",
#         "B13",
#         "B2",
#         "B3",
#         "alpha_g",
#         "alpha_w",
#         "alpha_r",
#         "alpha_infra",
#     ],
#     "bounds": bounds_final_sensitivity,  # 设置每个参数的范围
# }
