import math

# Constants
P = 30000.0
N0 = 20000
A = 0.8
beta = 1 / 3
R = 0.5


# Bounds
tourist_ratio_min = 0.01
tourist_ratio_max = 1.0

m1_min = 500
m1_max = 1000

m2_min = 10
m2_max = 70

rate_min = 0.1
rate_max = 0.5

alpha_min = 0
alpha_max = 1

B_min = 0
B_max = 1

B3_min = 0.15
B3_max = 1

k_min = 1e-4
k_max = 1e-2


# functions
def f_g(x):
    return math.tanh(x)


def f_w(x):
    return math.tanh(x)


def f_r(x):
    return math.tanh(x)


def f_infra(x):
    return math.tanh(x)
