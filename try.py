import pandas as pd
import json
import numpy as np

with open("optimization_results_R.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

optimized_params_flat = np.array(df["Optimized Parameters"].tolist())

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
mi_values = []
A_values = df["R"].values

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
    plt.xlabel("R", fontsize=12)
    plt.ylabel(param_name, fontsize=12)
    plt.grid(True)
    plt.show()

plt.tight_layout()
plt.show()