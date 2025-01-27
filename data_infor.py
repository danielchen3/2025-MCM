R = {
    "B12": 0.8314,
    "B13": 0.6965,
    "rate": 0.6913,
    "B11": 0.6479,
    "B2": 0.5736,
    "alpha_r": 0.5722,
    "tourist_ratio": 0.5360,
    "alpha_w": 0.5221,
    "B3": 0.5113,
    "m1": 0.5043,
    "m2": 0.4825,
    "alpha_g": 0.4627,
    "alpha_infra": 0.4194,
}

A = {
    "B12": 0.6897,
    "B2": 0.6840,
    "B13": 0.6546,
    "B11": 0.5814,
    "tourist_ratio": 0.4859,
    "alpha_g": 0.4754,
    "m2": 0.4525,
    "alpha_infra": 0.4143,
    "rate": 0.3835,
    "alpha_r": 0.3612,
    "m1": 0.3384,
    "alpha_w": 0.3230,
    "B3": 0.3180,
}

N0 = {
    "B2": 0.7703,
    "B11": 0.7682,
    "tourist_ratio": 0.7663,
    "rate": 0.7507,
    "B12": 0.7424,
    "alpha_g": 0.6975,
    "m2": 0.6881,
    "m1": 0.5777,
    "B13": 0.5497,
    "B3": 0.4857,
    "alpha_infra": 0.3566,
    "alpha_w": 0.3551,
    "alpha_r": 0.3125,
}

P = {
    "rate": 0.1668,
    "B11": 0.1668,
    "B12": 0.1668,
    "alpha_g": 0.1668,
    "alpha_w": 0.1668,
    "alpha_r": 0.1646,
    "tourist_ratio": 0.1391,
    "m1": 0.1391,
    "B2": 0.1391,
    "B3": 0.1391,
    "alpha_infra": 0.1328,
    "m2": 0.1201,
    "B13": 0.0330,
}

import matplotlib.pyplot as plt
import numpy as np


def calculate_means(P, A, R, N0):
    means = {
        "P": np.mean(list(P.values())),
        "A": np.mean(list(A.values())),
        "R": np.mean(list(R.values())),
        "N0": np.mean(list(N0.values())),
    }
    return means


def plot_means(P, A, R, N0):
    means = calculate_means(P, A, R, N0)

    plt.figure(figsize=(10, 6))
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

    bars = plt.barh(range(len(means)), means.values(), color=colors)
    plt.yticks(range(len(means)), means.keys())
    plt.title("Mean Mutual Information of External Variables on Decision Outcomes")
    plt.xlabel("Mean Mutual Information (MMI)")

    # 修正text()函数参数
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,  # x位置
            bar.get_y() + bar.get_height() / 2,  # y位置
            f"{width:.4f}",  # 显示文本
            ha="left",  # 水平对齐
            va="center",
        )  # 垂直对齐

    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()
    
    
# 调用函数
plot_means(P, A, R, N0)
