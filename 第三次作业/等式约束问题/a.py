import numpy as np
import matplotlib.pyplot as plt

seed = 123
np.random.seed(seed)

MAXITERS = 1000
ALPHA = 0.01
BETA = 0.5
NTTOL = 1e-7

def optimize(x0, A):
    n = len(x0)
    p = A.shape[0]
    x = x0.copy()
    vals = []
    for _ in range(MAXITERS):
        val = np.dot(x, np.log(x))
        vals.append(val)
        grad = 1 + np.log(x)
        hess = np.diag(1 / x)
        sol = -np.linalg.solve(np.block([[hess, A.T], [A, np.zeros((p, p))]]), np.block([grad, np.zeros(p)]))
        v = sol[:n]
        fprime = np.dot(grad, v)

        if abs(fprime) < NTTOL:
            break

        t = 1
        while np.min(x + t * v) <= 0:
            t = BETA * t

        while np.dot((x + t * v), np.log(x + t * v)) >= val + t * ALPHA * fprime:
            t = BETA * t

        x = x + t * v

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

    x, vals = optimize(x0, A)
    res = [vals[i]-vals[-1] for i in range(len(vals)-1)]
    plt.plot([i+1 for i in range(6)], res[:6], 'r*-')
    plt.xlabel('iteration')
    plt.ylabel('f(x)-f(x*)')
    plt.yscale('log')

    plt.show()
