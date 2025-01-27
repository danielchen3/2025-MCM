import math


def heuristic(rate, t=0):
    if t == 0:
        return math.tanh(rate)
    else:
        return math.tanh(rate)


def satisfaction_factor(N, P, ratio, t=0):
    get_heruistic = heuristic(ratio, t)
    S_nrt = (1 - N / P) * get_heruistic * (1 - get_heruistic)
    return S_nrt


def get_obj4(
    params,
    t=0,
):
    tourist_ratio, m1, m2, rate, B4 = params
    return I_pro(
        tourist_ratio,
        m1,
        m2,
        rate,
        B4,
        t,
    )


def neg_get_obj4(
    params,
    t=0,
):
    tourist_ratio, m1, m2, rate, B4 = params
    return -I_pro(
        tourist_ratio,
        m1,
        m2,
        rate,
        B4,
        t,
    )


def get_NH(tourist_ratio, m1, m2, rate, t):
    import config

    N = float(config.P) * tourist_ratio
    get_heruistic = heuristic(rate, t)
    return N * (m1 * get_heruistic + m2 * (1 - get_heruistic))


def I_pro(
    tourist_ratio,
    m1,
    m2,
    rate,
    B4,
    t=0,
):
    # Define the expression
    HN = get_NH(tourist_ratio, m1, m2, rate, t)
    k = 1 / HN
    L11 = HN * (math.exp(-1) + math.exp(-1) - 2)
    L12 = 2 * (math.exp(-1) + math.exp(-1) - 1)
    L21 = HN * math.exp(-k * B4 * HN) * (B4 + 1)
    return (L11 + L21) / L12
