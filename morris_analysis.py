from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import numpy as np
import matplotlib.pyplot as plt
from main import bounds_final_sensitivity, get_obj_final

# 定义灵敏度分析问题
problem = {
    "num_vars": 13,
    "names": [
        "tourist_ratio",
        "m1",
        "m2",
        "rate",
        "B11",
        "B12",
        "B13",
        "B2",
        "B3",
        "alpha_g",
        "alpha_w",
        "alpha_r",
        "alpha_infra",
    ],
    "bounds": bounds_final_sensitivity,  # 设置每个参数的范围
}

# 使用Morris方法生成采样点
param_values = morris.sample(problem, 1000000)  # 1000是采样点的数量


def check_constraints(params):
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
        alpha_g,
        alpha_w,
        alpha_r,
        alpha_infra,
    ) = params
    return (
        (alpha_g > alpha_w)
        and (alpha_g > alpha_r)
        and (alpha_g > alpha_infra)
        and (alpha_g + alpha_w + alpha_r + alpha_infra <= 1)
        and (B11 + B12 + B13 + B2 + B3 <= 1)
    )


# 计算每个采样点的目标函数值
Y = np.array(
    [get_obj_final(params) for params in param_values if check_constraints(params)]
)

# 过滤符合约束条件的参数和目标值
valid_params = []
valid_Y = []

for params in param_values:
    if check_constraints(params):
        valid_params.append(params)
        valid_Y.append(get_obj_final(params))

# 将有效参数和目标值转换为NumPy数组
valid_params = np.array(valid_params)
valid_Y = np.array(valid_Y)

# 计算最接近的14的倍数
target_size = (len(valid_params) // 14) * 14
print(f"原始大小: {len(valid_params)}, 目标大小: {target_size}")

# 生成随机索引
indices = np.random.choice(len(valid_params), target_size, replace=False)
indices.sort()  # 保持顺序

# 重新采样数据
valid_params = valid_params[indices]
valid_Y = valid_Y[indices]

# 确保样本数一致
print(f"Valid param_values shape: {valid_params.shape}")
print(f"Valid Y shape: {valid_Y.shape}")

# 确保 valid_params 和 valid_Y 维度一致
Si = morris_analyze.analyze(problem, valid_params, valid_Y)

# 打印敏感度分析结果
print(Si)  # 打印所有返回的结果，查看有哪些键

# 如果没有 'sigma_star'，可以检查 'sigma' 或其他相关的键
if "sigma" in Si:
    print("Sigma (standard deviation):")
    print(Si["sigma"])  # 输出标准差

print("Morris sensitivity analysis:")
print(Si["mu_star"])  # 输出平均灵敏度指数

# 1. 计算统计指标
sigma = Si["sigma"]  # 标准差
params = problem["names"]
mu = Si["mu"]  # 均值
mu_star = Si["mu_star"]  # 修正均值

# 2. 计算变异系数 (CV)
cv = sigma / mu_star
print("\n变异系数 (CV):")
for param, cv_value in zip(params, cv):
    print(f"{param}: {cv_value:.3f}")

# 3. 计算收敛指标
convergence_metric = sigma / (mu_star * np.sqrt(len(valid_Y)))
print("\n收敛指标:")
for param, conv in zip(params, convergence_metric):
    print(f"{param}: {conv:.3f}")
    
# 1. 绘制收敛指标图
plt.figure(figsize=(12, 6))
conv_bars = plt.bar(params, convergence_metric, color='skyblue')
threshold = 0.05
plt.axhline(y=threshold, color='r', linestyle='--', label='Convergence Threshold (0.05)')
plt.title('Convergence Metric Analysis')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# 设置y轴范围为最大值和阈值中的较大者
y_max = max(max(convergence_metric) * 1.2, threshold * 1.5)
plt.ylim(0, y_max)
plt.legend()

# 添加数值标签
for bar in conv_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 2. 绘制变异系数图
plt.figure(figsize=(12, 6))
cv_bars = plt.bar(params, cv, color='lightgreen')
plt.axhline(y=0.5, color='r', linestyle='--', label='CV Threshold (0.5)')
plt.title('Coefficient of Variation (CV) Analysis')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
plt.ylim(0, max(cv) * 1.2)
plt.legend()

# 添加数值标签
for bar in cv_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 3. 绘制可靠性评分图
plt.figure(figsize=(12, 6))
reliability_scores = 100 * (1 - np.minimum(convergence_metric/0.1, 1))
rel_bars = plt.bar(params, reliability_scores, color='lightcoral')
plt.axhline(y=80, color='r', linestyle='--', label='Reliability Threshold (80%)')
plt.title('Reliability Score Analysis')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend()

# 添加数值标签
for bar in rel_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom')
plt.tight_layout()
plt.show()

from adjustText import adjust_text

# 绘制散点图比较mu和sigma
plt.figure(figsize=(12, 8))  # 增大图形尺寸
scatter = plt.scatter(mu_star, sigma, alpha=0.5)
plt.xlabel("μ* (Modified Mean)")
plt.ylabel("σ (Standard Deviation)")
plt.title("Morris Reliability Analysis")

# 创建文本标注列表
texts = []
for i, param in enumerate(params):
    texts.append(plt.text(mu_star[i], sigma[i], param))

# 自动调整文本位置
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
    expand_points=(1.5, 1.5),
    force_points=(0.5, 0.5),
)

# 添加45度参考线
max_val = max(max(mu_star), max(sigma))
plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5)

plt.tight_layout()
plt.show()

# 5. 输出可靠性评估结果
reliability_threshold = 0.1  # 设定阈值
print("\n可靠性评估:")
for param, conv, cv_val in zip(params, convergence_metric, cv):
    if conv < reliability_threshold:
        status = "可靠"
    else:
        status = "需要更多样本"
    print(f"{param}: {status} (收敛指标={conv:.3f}, CV={cv_val:.3f})")

convergence_threshold = 0.1
reliability_scores = {}
print("可靠性分析:")

for i, param in enumerate(params):
    # 收敛指标评估
    conv_metric = sigma[i] / (mu_star[i] * np.sqrt(len(valid_Y)))
    cv_value = sigma[i] / mu_star[i]

    # 可靠性评分（0-100）
    reliability_score = 100 * (1 - min(conv_metric / convergence_threshold, 1))
    reliability_scores[param] = reliability_score

    print(f"\n参数 {param}:")
    print(
        f"- 收敛指标: {conv_metric:.3f} {'✓' if conv_metric < convergence_threshold else '✗'}"
    )
    print(f"- 变异系数: {cv_value:.3f}")
    print(f"- 可靠性评分: {reliability_score:.1f}%")

    if conv_metric < convergence_threshold:
        print("- 结论: 结果可靠")
    else:
        print("- 结论: 需要增加样本量")

# 可视化Morris方法结果
mu_star_values = Si["mu_star"]

# 绘制mu_star（平均灵敏度指数）
plt.figure(figsize=(10, 5))
plt.bar(params, mu_star_values, color="lightgreen")
plt.title("Morris Sensitivity Analysis")
plt.xlabel("Parameters")
plt.ylabel("Sensitivity Index")

plt.xticks(rotation=45, ha="right")  # ha='right' 确保标签右对齐
plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()

import matplotlib.pyplot as plt

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(
    mu_star_values,
    labels=params,
    autopct="%1.1f%%",
    startangle=90,
    colors=plt.cm.Paired.colors,
)

# 标题和调整
plt.title("Morris Sensitivity Analysis")
plt.axis("equal")  # 使饼图为圆形
plt.show()

# 清理之前的图
plt.clf()

# 计算角度值
num_params = len(params)  # 参数数量
angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # 首尾相连

# mu_star_values首尾相连
mu_star_values = np.concatenate((mu_star_values, [mu_star_values[0]]))

# 绘制雷达图
fig, ax = plt.subplots(figsize=(10, 10), dpi=120, subplot_kw=dict(polar=True))
ax.fill(angles, mu_star_values, color="lightgreen", alpha=0.25)
ax.plot(angles, mu_star_values, color="darkgreen", linewidth=2)

# 添加标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(params, fontsize=8)

# 标题
plt.title("Morris Sensitivity Analysis", pad=20)

# 显示图形
plt.tight_layout()
plt.show()
plt.close()