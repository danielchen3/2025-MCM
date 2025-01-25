from income import satisfaction_factor, heuristic
from config import *
from scipy.optimize import differential_evolution, dual_annealing


def get_cost_PA(tourist_ratio, ratio, m1, m2, t=0):
    N = P * tourist_ratio
    return N * (m1 * heuristic(ratio, t) + m2 * (1 - heuristic(ratio, t)))


def technology(B, function):
    return 1 / (1 + function(B))


bounds = [
    (tourist_ratio_min, tourist_ratio_max),
    (m1_min, m1_max),
    (m2_min, m2_max),
    (rate_min, rate_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
    (B_min, B_max),
]  


# 设置约束，确保它们的和小于 1
# def constraint(params):
#     tourist_ratio, ratio, m1, m2, B11, B12, B13, B2, B3 = params
#     return 1 - (B11 + B12 + B13 + B2 + B3)  # 确保它们的和小于1


def get_obj2(params, t=0):
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
    functions = [f_g, f_w, f_r, f_infra]
    alphas = [alpha_g, alpha_w, alpha_r, alpha_infra]
    Bs = [B11, B12, B13, B2]
    technologies = 0.0
    for i in range(4):
        technologies += alphas[i] * technology(Bs[i], functions[i])

    return get_cost_PA(tourist_ratio, rate, m1, m2, t) * technologies


def neg_get_obj2(params, t=0):
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
    functions = [f_g, f_w, f_r, f_infra]
    alphas = [alpha_g, alpha_w, alpha_r, alpha_infra]
    Bs = [B11, B12, B13, B2]
    technologies = 0.0
    for i in range(4):
        technologies += alphas[i] * technology(Bs[i], functions[i])

    return -get_cost_PA(tourist_ratio, rate, m1, m2, t) * technologies


# from scipy.optimize import NonlinearConstraint, LinearConstraint
# import numpy as np

# nlc = NonlinearConstraint(constraint, 0, np.inf)

# A = np.array([
#     [0, 0, 0, 0, 1, 1, 1, 1, 1]  # B11 + B12 + B13 + B2 + B3 <= 1
# ])

# linear_constraint = LinearConstraint(A, -np.inf, 1.0)

# result = differential_evolution(
#     get_obj2,
#     bounds,
#     constraints=linear_constraint,  # 设置约束
#     args=(0,),
#     maxiter=1000,
# )

# optimal_params = result.x
# optimal_obj2 = result.fun

# print(f"Optimized Parameters: {optimal_params}")
# print(f"Minimized OBJ2: {optimal_obj2}")
