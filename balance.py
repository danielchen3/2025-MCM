# from scipy.optimize import differential_evolution
# from main import (
#     bounds_final_sensitivity,
#     get_obj_final,
#     linear_constraint,
#     nlc,
#     alpha_linear_constraint,
#     max_nlc,
# )
# from config import P
from config import *

# def modify_P():
#     global P
#     print(f"original P is {P}")
#     P += 10000.0  # 修改全局变量 P
#     print(f"modified P is {P}")

p_values = []
minimized_finals = []
optimized_params = []

R_values = []
A_values = []
from scipy.optimize import differential_evolution
from main import (
    linear_constraint,
    nlc,
    alpha_linear_constraint,
    max_nlc,
    get_obj_final,
    bounds_final_sensitivity,
)

for i in range(50):

    print("next round start")

    import main

    (
        obj1_min_copy,
        obj1_max_copy,
        obj2_min_copy,
        obj2_max_copy,
        obj3_min_copy,
        obj3_max_copy,
    ) = main.get_objs_min_max()

    import config

    config.set_obj_value(
        obj1_min_copy,
        obj1_max_copy,
        obj2_min_copy,
        obj2_max_copy,
        obj3_min_copy,
        obj3_max_copy,
    )

    import main

    optimal_params, optimal_final = main.get_final()

    import config

    print(f"old A is {config.A}")

    # 收集每轮的数据
    A_values.append(config.A)
    minimized_finals.append(optimal_final)
    optimized_params.append(optimal_params.tolist())  # 转换numpy数组为列表

    increment_A()
    import config

    print(f"new A is {config.A}")

# 循环结束后，创建数据字典
data = {
    "A": A_values,
    "Minimized Final": minimized_finals,
    "Optimized Parameters": optimized_params,
}

# 可选：将结果写入文件

import pandas as pd
import json

with open("optimization_results_A.json", "w") as f:
    json.dump(data, f, indent=4)

# 设置pandas显示选项
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: "%.8f" % x)

# 读取JSON文件
with open("optimization_results_A.json", "r") as f:
    data = json.load(f)

# 创建基本DataFrame
df = pd.DataFrame({"A": data["A"], "Minimized Final": data["Minimized Final"]})

# 创建参数DataFrame
params_df = pd.DataFrame(
    data["Optimized Parameters"],
    columns=[f"param_{i+1}" for i in range(len(data["Optimized Parameters"][0]))],
)

# 合并DataFrame
final_df = pd.concat([df, params_df], axis=1)

# 打印结果
print("\n=== 优化结果汇总 ===")
print(final_df)

df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import numpy as np

# 绘制 P 与 Minimized Final 之间的关系
plt.figure(figsize=(8, 6))
plt.plot(df["A"], df["Minimized Final"], marker="o", linestyle="-", color="b")
plt.title("Effect of A on Minimized Final", fontsize=14)
plt.xlabel("A", fontsize=12)
plt.ylabel("Minimized Final", fontsize=12)
plt.grid(True)
plt.show()


import seaborn as sns

import matplotlib.pyplot as plt

# # 将优化参数拉平并与 Minimized Final 结合
optimized_params_flat = np.array(df["Optimized Parameters"].tolist())
# params_columns = [f"Param_{i+1}" for i in range(optimized_params_flat.shape[1])]
# optimized_params_df = pd.DataFrame(optimized_params_flat, columns=params_columns)
# optimized_params_df["Minimized Final"] = df["Minimized Final"]

# # 单独为每个参数绘制与 Minimized Final 的关系图
# for param in params_columns:
#     plt.figure(figsize=(8, 6))
#     plt.plot(
#         optimized_params_df[param],
#         optimized_params_df["Minimized Final"],
#         marker="o",
#         linestyle="-",
#         color="b",
#     )
#     plt.title(f"Effect of {param} on Minimized Final", fontsize=14)
#     plt.xlabel(param, fontsize=12)
#     plt.ylabel("Minimized Final", fontsize=12)
#     plt.grid(True)
#     plt.show()

# 获取params_1数据(i=0的情况)
# 存储互信息值

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
mi_values = []
A_values = df["A"].values

# 计算每个变量的互信息
for i in range(13):
    params = optimized_params_flat[:, i]
    
    # 离散化
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    A_discrete = discretizer.fit_transform(A_values.reshape(-1, 1)).ravel()
    params_discrete = discretizer.fit_transform(params.reshape(-1, 1)).ravel()
    
    # 计算互信息
    mi = mutual_info_score(A_discrete, params_discrete)
    mi_values.append((i, mi))
    
param_names = [
    'tourist_ratio',
    'm1',
    'm2',
    'rate',
    'B11',
    'B12',
    'B13',
    'B2',
    'B3',
    'alpha_g',
    'alpha_w',
    'alpha_r',
    'alpha_infra'
]

# 排序并打印结果
mi_values.sort(key=lambda x: x[1], reverse=True)
print("\n互信息值排序结果:")
for idx, mi in mi_values:
    print(f"参数{param_names[idx]}的互信息值: {mi:.4f}")


# 为top3参数分别绘制图像
for i in range(3):
    idx = mi_values[i][0]
    params = optimized_params_flat[:, idx]
    param_name = param_names[idx]  # 获取参数实际名称
    
    plt.figure(figsize=(8, 6))
    plt.plot(A_values, params, marker="o", linestyle="-", color="b")
    plt.title(f"{param_name} vs A (MI: {mi_values[i][1]:.4f})")
    plt.xlabel("A", fontsize=12)
    plt.ylabel(param_name, fontsize=12)
    plt.grid(True)
    plt.show()

plt.tight_layout()
plt.show()
