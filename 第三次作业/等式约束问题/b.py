import numpy as np
import matplotlib.pyplot as plt

seed = 123
np.random.seed(seed)

MAXITERS = 1000
ALPHA = 0.01
BETA = 0.5
RESTOL = 1e-7

def optimize(x0, A, b):
    n = len(x0)
    p = A.shape[0]
    x = x0.copy()
    nu = np.zeros(p)
    vals = []
    resdls = []
    for _ in range(MAXITERS):
        val = np.dot(x, np.log(x))
        vals.append(val)
        r = np.concatenate([1 + np.log(x) + np.dot(A.T, nu), np.dot(A, x) - b])
        resdls.append(np.linalg.norm(r))
        sol = -np.linalg.solve(np.block([[np.diag(1.0 / x), A.T], [A, np.zeros((p, p))]]), r)
        Dx = sol[:n]
        Dnu = sol[n:]

        if np.linalg.norm(r) < RESTOL:
            break

        t = 1
        while np.min(x + t * Dx) <= 0:
            t = BETA * t

        while np.linalg.norm(np.concatenate([1 + np.log(x + t * Dx) + np.dot(A.T, nu + t * Dnu), np.dot(A, x + t * Dx) - b])) > (1 - ALPHA * t) * np.linalg.norm(r):
            t = BETA * t

        x += t * Dx
        nu += t * Dnu

    vals.append(np.dot(x, np.log(x)))

    return x, vals


if __name__ == '__main__':
    n = 100
    p = 30
    # 生成p*n的满秩矩阵
    while True:
        A = np.random.rand(p, n)
        if np.linalg.matrix_rank(A) == p:
            break

    x0 = np.random.rand(n)

    b = np.dot(A, x0)

    x_1, vals_1 = optimize(x0, A, b)

    x0 = np.ones(n)
    x_2, vals_2 = optimize(x0, A, b)

    res_1 = [vals_1[i] - vals_1[-1] for i in range(len(vals_1) - 1)]
    res_2 = [vals_2[i] - vals_2[-1] for i in range(len(vals_2) - 1)]

    plt.plot([i+1 for i in range(6)], res_1[:6], 'r*-', label='x0 = random')
    plt.plot([i+1 for i in range(4)], res_2[:4], 'b*-', label='x0 = ones')
    plt.xlabel('iteration')
    plt.ylabel('f(x) - f(x*)')
    plt.yscale('log')
    plt.legend()
    plt.show()



