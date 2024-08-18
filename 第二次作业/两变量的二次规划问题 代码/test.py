import math
import numpy as np
import matplotlib.pyplot as plt

def f(x, gama):
    return 1/2*(x[0]**2 + gama*x[1]**2)


def get_x_k(k, gama):
    k = int(k)
    coef = ((gama-1)/(gama+1))**k
    x_k = [coef*gama, coef*((-1)**k)]
    return x_k


def coverage_process(gama, eps):
    k = 0
    x_k_s = []
    x_k = get_x_k(k, gama)
    while math.fabs(f(x_k, gama)) > eps:
        x_k_s.append(x_k)
        k += 1
        x_k = get_x_k(k, gama)
    x_k_s.append(x_k)
    return k, x_k_s


def draw_f_contour(gama, x_k_s):
    x1 = np.arange(-gama-1, gama+1, 0.1)
    x2 = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(x1, x2)
    Z = 1/2*(X**2 + gama*Y**2)

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, 30)
    ax.plot([x[0] for x in x_k_s], [x[1] for x in x_k_s], 'r*-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('gama = ' + str(gama))
    plt.savefig('gama=' + str(gama) + '.png')
    plt.show()


if __name__ == '__main__':
    gamas = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 2, 3, 4, 5, 8, 10]
    eps = 1e-6
    k_s = []
    for gama in gamas:
        k, x_k_s = coverage_process(gama, eps)
        k_s.append(k)
        draw_f_contour(gama, x_k_s)

    plt.plot(gamas, k_s, 'r*-')
    plt.xlabel('gama')
    plt.ylabel('k')
    plt.title('k-gama')
    plt.savefig('k-gama.png')
    plt.show()

