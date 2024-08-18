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


def search(x, d, alpha=0.1, beta=0.5):
    t = 1
    while f(x+t*d) > f(x)+alpha*t*np.dot(d_f(x), d):
        t = beta*t
    return t


def gradient_descent(x0, epsilon=1e-5, alpha=0.1, beta=0.5):
    x = x0
    x_list = [x]
    f_list = [f(x)]
    k = 0
    while np.linalg.norm(d_f(x)) > epsilon:
        x = x - search(x, -d_f(x), alpha, beta)*d_f(x)
        x_list.append(x)
        f_list.append(f(x))
        k += 1
    return x_list, k, f_list


def draw_contour(x_list, alpha, beta):
    x1 = np.arange(-1, 1, 0.1)
    x2 = np.arange(-1, 1, 0.1)
    X, Y = np.meshgrid(x1, x2)
    Z = np.exp(X+3*Y-0.1) + np.exp(X-3*Y-0.1) + np.exp(-X-0.1)

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, 30)
    ax.plot([x[0] for x in x_list], [x[1] for x in x_list], 'r*-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('alpha={}, beta={}'.format(alpha, beta))
    plt.savefig('pics/alpha={}, beta={}, contour.png'.format(alpha, beta))
    # plt.show()
    plt.close()


def draw_error_with_betas(f_list, k_s, alpha, betas, p_star):
    fig, ax = plt.subplots()
    for i in range(len(betas)):
        ax.plot(range(k_s[i]), [np.abs(f_list[i][k]-p_star) for k in range(k_s[i])], label='beta={}'.format(betas[i]))
    ax.set_xlabel('k')
    ax.set_ylabel('f(x)-p*')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('alpha={}'.format(alpha))
    plt.savefig('alpha={}, error.png'.format(alpha))
    plt.show()
    plt.close()


def draw_error_with_alphas(f_list, k_s, beta, alphas, p_star):
    fig, ax = plt.subplots()
    for i in range(len(alphas)):
        ax.plot(range(k_s[i]), [np.abs(f_list[i][k]-p_star) for k in range(k_s[i])], label='alpha={}'.format(alphas[i]))
    ax.set_xlabel('k')
    ax.set_ylabel('f(x)-p*')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('beta={}'.format(beta))
    plt.savefig('beta={}, error.png'.format(beta))
    plt.show()
    plt.close()


def draw_k_with_variables(k_s, variables, constant, type, constant_name):
    fig, ax = plt.subplots()
    ax.plot(variables, k_s, 'ro-')
    ax.set_xlabel(type)
    ax.set_ylabel('k')
    ax.set_title(f"the relationship between k and {type}, {constant_name}={constant}")
    plt.savefig(f'k-{type}-{constant_name}={constant}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    epsilon = 1e-7
    p_star = f([-np.log(2)/2, 0])
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x0 = np.array([0.1, 0.1])

    # beta变化的时候收敛速度
    alpha = 0.3
    k_s = []
    f_list_s = []
    for beta in betas:
        x_list, k, f_list = gradient_descent(x0, epsilon, alpha, beta)
        print('alpha={}, beta={}, k={}, f(x)={}'.format(alpha, beta, k, f_list[-1]))
        k_s.append(k)
        f_list_s.append(f_list)
        draw_contour(x_list, alpha, beta)

    draw_error_with_betas(f_list_s, k_s, alpha, betas, p_star)
    draw_k_with_variables(k_s, betas, alpha, type='beta', constant_name='alpha')

    # alpha变化的时候收敛速度
    beta = 0.8
    k_s = []
    f_list_s = []
    for alpha in alphas:
        x_list, k, f_list = gradient_descent(x0, epsilon, alpha, beta)
        print('alpha={}, beta={}, k={}, f(x)={}'.format(alpha, beta, k, f_list[-1]))
        k_s.append(k)
        f_list_s.append(f_list)
        draw_contour(x_list, alpha, beta)

    draw_error_with_alphas(f_list_s, k_s, beta, alphas, p_star)
    draw_k_with_variables(k_s, alphas, beta, type='alpha', constant_name='beta')






