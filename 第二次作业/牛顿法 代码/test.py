import numpy as np
import matplotlib.pyplot as plt


def f(x):
    x = np.array(x)
    y = np.array([x[0]+3*x[1]-0.1, x[0]-3*x[1]-0.1, -x[0]-0.1])
    return np.sum(np.exp(y))


def d_f(x):
    x = np.array(x)
    y = np.array([x[0]+3*x[1]-0.1, x[0]-3*x[1]-0.1, -x[0]-0.1])
    z = np.exp(y)
    return np.array([z[0]+z[1]-z[2], 3*z[0]-3*z[1]])


def d2_f(x):
    x = np.array(x)
    y = np.array([x[0]+3*x[1]-0.1, x[0]-3*x[1]-0.1, -x[0]-0.1])
    z = np.exp(y)
    return np.array([[z[0]+z[1]+z[2], 3*z[0]-3*z[1]], [3*z[0]-3*z[1], 9*z[0]+9*z[1]]])


def search(x, d, alpha=0.1, beta=0.7):
    t = 1
    while f(x+t*d) > f(x)+alpha*t*np.dot(d_f(x), d):
        t = beta*t
    return t


def Newton_method(x0, epsilon=1e-7):
    x = x0
    x_list = [x0]
    f_list = [f(x0)]
    k = 0
    t_s = []
    while True:
        k += 1
        x_nt = -np.dot(np.linalg.inv(d2_f(x)), d_f(x))
        lambda_2 = np.dot(np.dot(np.transpose(d_f(x)), np.linalg.inv(d2_f(x))), d_f(x))
        if lambda_2/2 <= epsilon:
            break
        t = search(x, x_nt)
        t_s.append(t)
        x = x + t*x_nt
        x_list.append(x)
        f_list.append(f(x))
    return x, k, x_list, f_list, t_s


if __name__ == '__main__':
    epsilon = 1e-8
    p_star = f([-np.log(2)/2, 0])
    x0 = np.array([-1, 1])

    x, k, x_list, f_list, t = Newton_method(x0, epsilon)
    print([np.abs(f_i - p_star) for f_i in f_list])

    # 画出误差曲线
    plt.figure()
    plt.plot(range(k), [np.abs(f_i - p_star) for f_i in f_list], 'r*-')
    plt.xlabel('k')
    plt.ylabel('f(x)-p*')
    plt.yscale('log')
    plt.title('Newton method')
    plt.show()

    # 画出步长曲线
    plt.figure()
    plt.plot(range(k-1), t, 'r*-')
    plt.xlabel('k')
    plt.ylabel('t')
    plt.title('Newton method')
    plt.show()

    # 画出收敛轨迹
    x1 = np.arange(-1.5, 1.5, 0.1)
    x2 = np.arange(-1.5, 1.5, 0.1)
    X, Y = np.meshgrid(x1, x2)
    Z = np.exp(X + 3 * Y - 0.1) + np.exp(X - 3 * Y - 0.1) + np.exp(-X - 0.1)
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, 100)
    ax.plot([x[0] for x in x_list], [x[1] for x in x_list], 'r*-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Newton method')
    plt.show()


