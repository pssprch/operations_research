import scipy.optimize as opt
import numpy as np


def nash_equilibrium(pm):
    A = np.array(pm)
    n, m = A.shape

    c = np.ones(n)
    b = -np.ones(m)

    offset = abs(A.min())
    A += offset

    res1 = opt.linprog(c, A_ub=-A.T, b_ub=b)
    res2 = opt.linprog(b, A_ub=A, b_ub=c)

    v = 1 / res1.fun - offset
    strat1 = res1.x / np.sum(res1.x)
    strat2 = res2.x / np.sum(res2.x)

    return strat1, strat2, v


if __name__ == '__main__':
    n, m = int(input()), int(input())
    A = [list(map(int, input().split())) for _ in range(n)]

    strat1, strat2, value = nash_equilibrium(A)

    print(f"Цена игры: {value}")
    print(f"Массив стратегии первого игрока: {strat1}")
    print(f"Массив стратегии второго игрока: {strat2}")
