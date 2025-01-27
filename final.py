import main

(
    obj1_min_copy,
    obj1_max_copy,
    obj2_min_copy,
    obj2_max_copy,
    obj3_min_copy,
    obj3_max_copy,
    obj4_min_copy,
    obj4_max_copy,
) = main.get_objs_min_max()
import config

config.set_obj_value(
    obj1_min_copy,
    obj1_max_copy,
    obj2_min_copy,
    obj2_max_copy,
    obj3_min_copy,
    obj3_max_copy,
    obj4_min_copy,
    obj4_max_copy,
)

import main

optimal_params, optimal_final = main.get_final()

obj4_datas = main.obj4_real_datas

import main
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
import numpy as np

# 创建图表
plt.figure(figsize=(12, 8))

# 设置插值点数量
n_points = 10000
smooth_points = 10000

# 创建平滑插值
x_smooth = np.linspace(min(main.iteration_nums[:n_points]), 
                      max(main.iteration_nums[:n_points]), 
                      smooth_points)

# 对每条线进行平滑处理
spl1 = make_interp_spline(main.iteration_nums[:n_points], main.t1[:n_points])
spl2 = make_interp_spline(main.iteration_nums[:n_points], main.t2[:n_points])
spl3 = make_interp_spline(main.iteration_nums[:n_points], main.t3[:n_points])
spl4 = make_interp_spline(main.iteration_nums[:n_points], main.t4[:n_points])

# 绘制平滑曲线
plt.plot(x_smooth, spl1(x_smooth), "r-", 
         label="Targeted Attractions for Promotion", linewidth=1.5)
plt.plot(x_smooth, spl2(x_smooth), "b-", 
         label="Glacier Attraction", linewidth=1.5)
plt.plot(x_smooth, spl3(x_smooth), "g-", 
         label="Whale Watching Area", linewidth=1.5)
plt.plot(x_smooth, spl4(x_smooth), "y-", 
         label="Rainforest Area", linewidth=1.5)

# 设置图表属性
plt.title("Attraction Pressure Distribution", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Pressure Distribution", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=10)

# 优化布局
plt.tight_layout()

# 保存和显示
plt.savefig("B_parameters_optimization.png", dpi=300, bbox_inches="tight")
plt.show()
