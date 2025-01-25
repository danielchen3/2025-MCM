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

# result = differential_evolution(get_obj1, bounds_1, args=(0,), maxiter=10000)

# # output the result
# optimal_params = result.x
# optimal_obj1 = result.fun

# print(f"最优参数: {optimal_params}")
# print(f"最小化的 obj1 值: {optimal_obj1}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution


# class OptimizationLogger:
#     def __init__(self):
#         self.obj_vals = []
#         self.best_params = []
#         self.convergence = []
#         self.param_names = ["tourist_ratio", "m1", "m2", "rate"]

#     def callback(self, xk, convergence):
#         self.obj_vals.append(-neg_get_obj1(xk))
#         self.best_params.append(xk)
#         self.convergence.append(convergence)
#         return False


# # Setup logger and optimization
# logger = OptimizationLogger()
# # plt.style.use()

# result = differential_evolution(
#     neg_get_obj1,
#     bounds_1,
#     args=(0,),
#     maxiter=10000,
#     tol=1e-7,  # 降低收敛容差
#     callback=logger.callback,
#     disp=True,
# )

# # Visualization
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # Objective Function Convergence
# axes[0, 0].plot(logger.obj_vals, "b-", linewidth=2)
# axes[0, 0].set_title("Objective Function Convergence")
# axes[0, 0].set_xlabel("Iteration")
# axes[0, 0].set_ylabel("Objective Value")
# axes[0, 0].grid(True)

# # Convergence Rate
# axes[0, 1].plot(logger.convergence, "r-", linewidth=2)
# axes[0, 1].set_title("Convergence Rate")
# axes[0, 1].set_xlabel("Iteration")
# axes[0, 1].set_ylabel("Convergence Value")
# axes[0, 1].grid(True)

# # Parameter Evolution Heatmap
# params = np.array(logger.best_params)
# im = axes[1, 0].imshow(params.T, aspect="auto", cmap="viridis")
# axes[1, 0].set_title("Parameter Evolution")
# axes[1, 0].set_xlabel("Iteration")
# axes[1, 0].set_yticks(range(len(logger.param_names)))
# axes[1, 0].set_yticklabels(logger.param_names)
# plt.colorbar(im, ax=axes[1, 0], label="Parameter Value")

# # Final Parameters Distribution
# x_pos = np.arange(len(logger.param_names))
# axes[1, 1].bar(x_pos, result.x, color="skyblue")
# axes[1, 1].set_title("Optimal Parameters")
# axes[1, 1].set_xlabel("Parameters")
# axes[1, 1].set_ylabel("Value")
# axes[1, 1].set_xticks(x_pos)
# axes[1, 1].set_xticklabels(logger.param_names, rotation=45)

# plt.tight_layout()
# plt.show()

# # result = differential_evolution(neg_get_obj1, bounds_1, args=(0,), maxiter=10000)

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

alpha_linear_constraint = LinearConstraint(alpha_cons, -np.inf, 1.0)

alpha_cons_obj2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

alpha_linear_constraint_obj2 = LinearConstraint(alpha_cons_obj2, -np.inf, 1.0)

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
    (B3_min, B3_max),
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
    (B3_min, B3_max),
    (k_min, k_max),
    (k_min, k_max),
    (k_min, k_max),
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
    obj1_min, obj1_max = -17492684.54, -17824.19
    obj2_min, obj2_max = 0.0, 17835375.23
    obj3_min, obj3_max = -30000.94, -1062.22

    # 归一化
    obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
    obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
    obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

    # 记录当前代的目标值
    obj1_values.append(obj1_norm)
    obj2_values.append(obj2_norm)
    obj3_values.append(obj3_norm)


# 用回调记录每代最优解的目标值
callback = record_objective_values


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
    obj2_min, obj2_max = 0.0, 17835375.23  # 需要设定obj2的范围
    obj3_min, obj3_max = -30000.94, -1062.22  # 需要设定obj3的范围

    # 归一化
    obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
    obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
    obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

    # 添加权重（可选）
    w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3  # 权重系数

    print(
        f"obj1_norm is {obj1_norm}, obj2_norm is {obj2_norm}, obj3_norm is {obj3_norm}"
    )
    final_result = w1 * obj1_norm + w2 * obj2_norm + w3 * obj3_norm
    print(f"final_result is {final_result}")

    return final_result


result = differential_evolution(
    get_obj_final,
    bounds_final,
    constraints=[linear_constraint, nlc, alpha_linear_constraint, max_nlc],  # 设置约束
    args=(0,),
    maxiter=4000,
    tol=1e-10,
    callback=callback,
)

optimal_params = result.x
optimal_final = result.fun

print(f"Optimized Parameters: {optimal_params}")
print(f"Minimized Final: {optimal_final}")


import numpy as np
from scipy.signal import savgol_filter


# 平滑化函数
def smooth_data(data, window=5):
    return savgol_filter(data, window, 3)  # window需要是奇数


plt.figure(figsize=(12, 6))

# 原始数据（半透明）
# plt.plot(obj1_values, 'b--', alpha=0.3, label='Original Obj1')
# plt.plot(obj2_values, 'g--', alpha=0.3, label='Original Obj2')
# plt.plot(obj3_values, 'r--', alpha=0.3, label='Original Obj3')

# 平滑后的数据
plt.plot(smooth_data(obj1_values), "b-", label="Obj1 value")
plt.plot(smooth_data(obj2_values), "g-", label="Obj2 value")
plt.plot(smooth_data(obj3_values), "r-", label="Obj3 value")

plt.xlabel("Generation")
plt.ylabel("Normalized Objective Value")
plt.legend()
plt.title("Normalized Objective Values Over Generations")
plt.grid(True)
plt.show()


# class MultiObjectiveLogger:
#     def __init__(self):
#         self.obj1_raw = []
#         self.obj2_raw = []
#         self.obj3_raw = []
#         self.obj1_norm = []
#         self.obj2_norm = []
#         self.obj3_norm = []
#         self.final_vals = []


# def get_normalized_objectives(params_obj1, params_obj2, params_obj3, t):
#     # Add global logger
#     global optimizer_logger

#     obj1_raw = -get_obj1(params_obj1, t)
#     obj2_raw = get_obj2(params_obj2, t)
#     obj3_raw = -get_obj3(params_obj3, t)

#     # Record raw values
#     optimizer_logger.obj1_raw.append(obj1_raw)
#     optimizer_logger.obj2_raw.append(obj2_raw)
#     optimizer_logger.obj3_raw.append(obj3_raw)

#     obj1_min, obj1_max = -17492684.54, -17824.19
#     obj2_min, obj2_max = 0.0, 17835375.23
#     obj3_min, obj3_max = -30000.0, -1.09e-34

#     obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
#     obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
#     obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

#     # Record normalized values
#     optimizer_logger.obj1_norm.append(obj1_norm)
#     optimizer_logger.obj2_norm.append(obj2_norm)
#     optimizer_logger.obj3_norm.append(obj3_norm)

#     w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3
#     final_result = w1 * obj1_norm + w2 * obj2_norm + w3 * obj3_norm
#     optimizer_logger.final_vals.append(final_result)

#     return final_result


# # Visualization function
# def plot_optimization_process():
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

#     # Raw objectives
#     axes[0, 0].plot(optimizer_logger.obj1_raw, label="Obj1")
#     axes[0, 0].plot(optimizer_logger.obj2_raw, label="Obj2")
#     axes[0, 0].plot(optimizer_logger.obj3_raw, label="Obj3")
#     axes[0, 0].set_title("Raw Objective Values")
#     axes[0, 0].set_xlabel("Iteration")
#     axes[0, 0].set_ylabel("Value")
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)

#     # Normalized objectives
#     axes[0, 1].plot(optimizer_logger.obj1_norm, label="Obj1")
#     axes[0, 1].plot(optimizer_logger.obj2_norm, label="Obj2")
#     axes[0, 1].plot(optimizer_logger.obj3_norm, label="Obj3")
#     axes[0, 1].set_title("Normalized Objective Values")
#     axes[0, 1].set_xlabel("Iteration")
#     axes[0, 1].set_ylabel("Value")
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)

#     # Final composite value
#     axes[1, 0].plot(optimizer_logger.final_vals, "r-")
#     axes[1, 0].set_title("Composite Objective Value")
#     axes[1, 0].set_xlabel("Iteration")
#     axes[1, 0].set_ylabel("Value")
#     axes[1, 0].grid(True)

#     # 3D visualization
#     ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
#     ax_3d.scatter(
#         optimizer_logger.obj1_norm,
#         optimizer_logger.obj2_norm,
#         optimizer_logger.obj3_norm,
#         c=range(len(optimizer_logger.obj1_norm)),
#         cmap="viridis",
#     )
#     ax_3d.set_xlabel("Obj1")
#     ax_3d.set_ylabel("Obj2")
#     ax_3d.set_zlabel("Obj3")
#     ax_3d.set_title("Solution Space")

#     plt.tight_layout()
#     plt.show()


# # 1. 创建记录器实例
# optimizer_logger = MultiObjectiveLogger()


# # 2. 定义回调函数
# def callback(xk, convergence):
#     params_obj1 = xk[:4]  # tourist_ratio, m1, m2, rate
#     params_obj2 = (
#         xk[0],
#         xk[1],
#         xk[2],
#         xk[3],
#         xk[4],
#         xk[5],
#         xk[6],
#         xk[7],
#         xk[12],
#         xk[13],
#         xk[14],
#         xk[15],
#     )
#     params_obj3 = (xk[0], xk[1], xk[2], xk[3], xk[8], xk[9], xk[10], xk[11])

#     # 计算原始目标值
#     obj1_raw = -get_obj1(params_obj1, 0)
#     obj2_raw = get_obj2(params_obj2, 0)
#     obj3_raw = -get_obj3(params_obj3, 0)

#     optimizer_logger.obj1_raw.append(obj1_raw)
#     optimizer_logger.obj2_raw.append(obj2_raw)
#     optimizer_logger.obj3_raw.append(obj3_raw)

#     obj1_min, obj1_max = -17492684.54, -17824.19
#     obj2_min, obj2_max = 0.0, 17835375.23
#     obj3_min, obj3_max = -30000.0, -1.09e-34

#     obj1_norm = normalize_objective(obj1_raw, obj1_min, obj1_max)
#     obj2_norm = normalize_objective(obj2_raw, obj2_min, obj2_max)
#     obj3_norm = normalize_objective(obj3_raw, obj3_min, obj3_max)

#     # Record normalized values
#     optimizer_logger.obj1_norm.append(obj1_norm)
#     optimizer_logger.obj2_norm.append(obj2_norm)
#     optimizer_logger.obj3_norm.append(obj3_norm)

#     w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3
#     final_result = w1 * obj1_norm + w2 * obj2_norm + w3 * obj3_norm
#     optimizer_logger.final_vals.append(final_result)

#     return False


# result = differential_evolution(
#     get_obj_final,
#     bounds_final,
#     constraints=[linear_constraint, nlc, alpha_linear_constraint, max_nlc],  # 设置约束
#     args=(0,),
#     maxiter=10000,
#     callback=callback,
#     disp=True,
# )


# # 4. 优化后可视化
# def plot_optimization_results():
#     fig = plt.figure(figsize=(20, 15))

#     # Raw objectives evolution
#     ax1 = fig.add_subplot(2, 3, 1)
#     ax1.plot(optimizer_logger.obj1_raw, label='Economic')
#     ax1.plot(optimizer_logger.obj2_raw, label='Environmental')
#     ax1.plot(optimizer_logger.obj3_raw, label='Social')
#     ax1.set_title('Raw Objectives Evolution')
#     ax1.set_xlabel('Iteration')
#     ax1.set_ylabel('Value')
#     ax1.legend()
#     ax1.grid(True)

#     # Normalized objectives evolution
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax2.plot(optimizer_logger.obj1_norm, label='Economic')
#     ax2.plot(optimizer_logger.obj2_norm, label='Environmental')
#     ax2.plot(optimizer_logger.obj3_norm, label='Social')
#     ax2.set_title('Normalized Objectives Evolution')
#     ax2.set_xlabel('Iteration')
#     ax2.set_ylabel('Value')
#     ax2.legend()
#     ax2.grid(True)

#     # Raw 3D trajectory
#     ax3 = fig.add_subplot(2, 3, 3, projection='3d')
#     scatter = ax3.scatter(optimizer_logger.obj1_raw,
#                          optimizer_logger.obj2_raw,
#                          optimizer_logger.obj3_raw,
#                          c=range(len(optimizer_logger.obj1_raw)),
#                          cmap='viridis')
#     ax3.set_xlabel('Economic')
#     ax3.set_ylabel('Environmental')
#     ax3.set_zlabel('Social')
#     ax3.set_title('Raw Optimization Trajectory')
#     plt.colorbar(scatter, ax=ax3, label='Iteration')

#     # Normalized 3D trajectory
#     ax4 = fig.add_subplot(2, 3, 4, projection='3d')
#     scatter = ax4.scatter(optimizer_logger.obj1_norm,
#                          optimizer_logger.obj2_norm,
#                          optimizer_logger.obj3_norm,
#                          c=range(len(optimizer_logger.obj1_norm)),
#                          cmap='viridis')
#     ax4.set_xlabel('Economic')
#     ax4.set_ylabel('Environmental')
#     ax4.set_zlabel('Social')
#     ax4.set_title('Normalized Optimization Trajectory')
#     plt.colorbar(scatter, ax=ax4, label='Iteration')

#     # Add more visualization if needed in positions (2,3,5) and (2,3,6)

#     plt.tight_layout()
#     plt.show()


# # 5. 显示结果
# plot_optimization_results()
