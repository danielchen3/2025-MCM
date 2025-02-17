# from config import *
from config import *
import math

from scipy.optimize import differential_evolution, dual_annealing
from income import heuristic


def housing_heur(N, t=0):
    import config
    return math.log(1 + config.A) + math.log(1 + (N - 3000) / (20000 - 3000))


def overcrowding_heur(N, t=0):
    import config
    return math.log(1 + config.A) + math.log(1 + (N - 3000) / (20000 - 3000))


def get_N_anti(t=0):
    import config
    # from config import P, R, A, N0
    N_MIN = config.P * tourist_ratio_min
    N_MAX = config.P * tourist_ratio_max
    return config.R * N_MAX + (1 - config.R) * N_MIN


def housing(N, k1, t=0):
    from config import P, R, A, N0
    N_anti = get_N_anti(t)
    h = housing_heur(N, t)
    h_anti = housing_heur(N_anti, t)
    if h <= h_anti:
        return 1
    else:
        return 2 / (1 + math.exp(k1 * (h - h_anti)))


def overcrowding(N, k2, t=0):
    from config import P, R, A, N0
    N_anti = get_N_anti(t)
    h = overcrowding_heur(N, t)
    h_anti = overcrowding_heur(N_anti, t)
    if h <= h_anti:
        return 1
    else:
        return 2 / (1 + math.exp(k2 * (h - h_anti)))


def additional_income(B3, rate, m1, m2, N, k3, t=0):
    from config import P, R, A, N0
    N_anti = get_N_anti(t)
    I_anti = B3 * N_anti * (m1 * heuristic(rate, t) + m2 * (1 - heuristic(rate, t)))
    I = B3 * N * (m1 * heuristic(rate, t) + m2 * (1 - heuristic(rate, t)))
    # print(f"I is {I}, I_anti is {I_anti}")
    if I >= I_anti:
        return 1
    else:
        return 1 / (1 + k3 * math.sqrt((I_anti - I)))


def get_obj3(params, t=0):
    import config
    tourist_ratio, m1, m2, rate, B3 = params
    N = config.P * tourist_ratio
    S1 = housing(N, 3.86e-3, t)
    # print(f"S1 is {S1}")
    S2 = overcrowding(N, 5.89e-3, t)
    # print(f"S2 is {S2}")
    S3 = additional_income(B3, rate, m1, m2, N, 1.0e-4, t)
    # print(f"S3 is {S3}")
    return S1 * S2 * S3 * config.P


def neg_get_obj3(params, t=0):
    import config
    tourist_ratio, m1, m2, rate, B3 = params
    N = config.P * tourist_ratio
    S1 = housing(N, 3.86e-3, t)
    # print(f"S1 is {S1}")
    S2 = overcrowding(N, 5.89e-3, t)
    # print(f"S2 is {S2}")
    S3 = additional_income(B3, rate, m1, m2, N, 1.0e-4, t)
    # print(f"S3 is {S3}")
    return -S1 * S2 * S3 * config.P