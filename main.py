from income import *

from config import *

from environment import get_obj2, neg_get_obj2

from res_satisfaction import get_obj3, neg_get_obj3

from scipy.optimize import NonlinearConstraint, LinearConstraint


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
        k1,
        k2,
        k3,
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
        k1,
        k2,
        k3,
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    N = P * tourist_ratio
    K = 4 * beta * heuristic(rate, t) * (1 - heuristic(rate, t))  # 假设h(r,t)函数已定义

    left = N / np.exp(K * (1 - N / P))
    right = N0 / np.exp(beta * (rate + alpha_g + 1))
    return left - right  # 返回0表示等式约束


nlc = NonlinearConstraint(nonlinear_constraint, 0, 0)  # 等式约束


bounds_1 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
]  # tourist_ratio, m1, m2, rate

# result = differential_evolution(get_obj1, bounds_1, args=(0,), maxiter=10000)

# # output the result
# optimal_params = result.x
# optimal_obj1 = result.fun

# print(f"最优参数: {optimal_params}")
# print(f"最小化的 obj1 值: {optimal_obj1}")

# result = differential_evolution(neg_get_obj1, bounds_1, args=(0,), maxiter=10000)

# # output the result
# optimal_params = result.x
# optimal_obj1 = -result.fun

# print(f"最优参数: {optimal_params}")
# print(f"最大化的 obj1 值: {optimal_obj1}")

# 使用模拟退火进行优化
# result = dual_annealing(get_obj1, bounds, args=(0,))

# 输出最优结果
# optimal_params = result.x
# optimal_obj1 = -result.fun  # 同样需要取负值

# print(f"最优参数: {optimal_params}")
# print(f"最大化的 obj1 值: {optimal_obj1}")

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
    [[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
)  # B11 + B12 + B13 + B2 + B3 <= 1

linear_constraint = LinearConstraint(A, -np.inf, 1.0)

alpha_cons = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

alpha_linear_constraint = LinearConstraint(alpha_cons, 1.0, 1.0)

alpha_cons_obj2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

alpha_linear_constraint_obj2 = LinearConstraint(alpha_cons_obj2, 1.0, 1.0)

# result = differential_evolution(
#     get_obj2,
#     bounds_2,
#     constraints=[alpha_linear_constraint_obj2, max_nlc_obj2],  # 设置约束
#     args=(0,),
#     maxiter=10000,
# )

# optimal_params = result.x
# optimal_obj2 = result.fun

# print(f"Optimized Parameters: {optimal_params}")
# print(f"Minimized OBJ2: {optimal_obj2}")

# result = differential_evolution(
#     neg_get_obj2,
#     bounds_2,
#     constraints=[alpha_linear_constraint_obj2, max_nlc_obj2],  # 设置约束
#     args=(0,),
#     maxiter=10000,
# )

# optimal_params = result.x
# optimal_obj2 = -result.fun

# print(f"Optimized Parameters: {optimal_params}")
# print(f"Maximized OBJ2: {optimal_obj2}")

bounds_3 = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (k_min, k_max),
    (k_min, k_max),
    (k_min, k_max),
]

# result = differential_evolution(get_obj3, bounds_3, args=(0,), maxiter=10000)

# # output the result
# optimal_params = result.x
# optimal_obj3 = result.fun

# print(f"最优参数: {optimal_params}")
# print(f"最小化的 obj3 值: {optimal_obj3}")

# result = differential_evolution(neg_get_obj3, bounds_3, args=(0,), maxiter=10000)

# # output the result
# optimal_params = result.x
# optimal_obj3 = -result.fun

# print(f"最优参数: {optimal_params}")
# print(f"最大化的 obj3 值: {optimal_obj3}")


bounds_final = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (k_min, k_max),
    (k_min, k_max),
    (k_min, k_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
    (alpha_min, alpha_max),
]


def get_obj_final(params, t=0):
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
        k1,
        k2,
        k3,
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
    params_obj3 = tourist_ratio, m1, m2, rate, B3, k1, k2, k3
    return get_normalized_objectives(params_obj1, params_obj2, params_obj3, t)


def normalize_objective(value, min_val, max_val):
    """归一化函数，将值映射到[0,1]区间"""
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def get_normalized_objectives(params_obj1, params_obj2, params_obj3, t):
    # 获取原始值
    obj1_raw = -get_obj1(params_obj1, t)  # 最大化转最小化
    obj2_raw = get_obj2(params_obj2, t)  # 最小化
    obj3_raw = -get_obj3(params_obj3, t)  # 最大化转最小化

    # 设定或计算每个目标的最大最小值
    # 这些值可以通过采样或者先验知识获得
    obj1_min, obj1_max = -17492684.54, -17824.19  # 需要设定obj1的范围
    obj2_min, obj2_max = 10020.01, 17835375.23  # 需要设定obj2的范围
    obj3_min, obj3_max = -30000.0, -1.09e-34  # 需要设定obj3的范围

    # 归一化
    obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
    obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
    obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

    # 添加权重（可选）
    w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3  # 权重系数
    
    print(f"obj1_norm is {obj1_norm}, obj2_norm is {obj2_norm}, obj3_norm is {obj3_norm}")
    final_result = w1 * obj1_norm + w2 * obj2_norm + w3 * obj3_norm
    print(f"final_result is {final_result}")

    return final_result


result = differential_evolution(
    get_obj_final,
    bounds_final,
    constraints=[linear_constraint, nlc],  # 设置约束
    args=(0,),
    maxiter=10000,
)

optimal_params = result.x
optimal_final = result.fun

print(f"Optimized Parameters: {optimal_params}")
print(f"Minimized Final: {optimal_final}")
