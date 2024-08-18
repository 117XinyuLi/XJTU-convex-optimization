import numpy as np
import matplotlib.pyplot as plt

seed = 123
np.random.seed(seed)

MAXITERS = 100
ALPHA = 0.01
BETA = 0.5
NTTOL = 1e-8

def optimize(A, b):
    nu = np.zeros(p)
    vals = []
    for _ in range(MAXITERS):
        val = np.dot(b.T, nu) + np.sum(np.exp(-np.dot(A.T, nu) - 1))
        vals.append(val)
        grad = b - np.dot(A, np.exp(-np.dot(A.T, nu) - 1))
        hess = np.dot(np.dot(A, np.diagflat(np.exp(-np.dot(A.T, nu) - 1))), A.T)
        v = -np.linalg.solve(hess, grad)
        fprime = np.dot(grad.T, v)
        if abs(fprime) < NTTOL:
            break

        t = 1
        while np.dot(b.T, nu + t * v) + np.sum(np.exp(-np.dot(A.T, nu + t * v) - 1)) > val + t * ALPHA * fprime:
            t = BETA * t

        nu = nu + t * v

    vals.append(np.dot(b, nu) + np.sum(np.exp(-np.dot(A.T, nu) - 1)))

    return nu, vals


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

    result, vals = optimize(A, b)

    res = [vals[i] - vals[-1] for i in range(len(vals) - 1)]
    plt.plot([i+1 for i in range(4)], res[:4], 'r*-')
    plt.xlabel('iteration')
    plt.ylabel('p*-g(v)')
    plt.yscale('log')
    plt.show()


