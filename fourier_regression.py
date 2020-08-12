"""
liner regression by approximate fourier series

sample of f                : f(t_1), ..., f(t_k)
approximate fourier series : g(t) = Σ_{n=0}^N (an sin(nx) + bn cos(nx))
loss function              : Σ_{k=1}^K (f(t_k) - g(t_k))^2 = || bm{f} - Aw ||_2^2

    [cos(0t_1), sin(t_1), cos(t_1), sin(2t_1), cos(2t_1), ..., sin(Nt_1), cos(Nt_1)]
A = [cos(0t_2), sin(t_2), cos(t_2), sin(2t_2), cos(2t_2), ..., sin(Nt_2), cos(Nt_2)]
    [ ...     ,                                                         , ...      ]
    [cos(0t_K), sin(t_K), cos(t_K), sin(2t_K), cos(2t_K), ..., sin(Nt_K), cos(Nt_K)]

w = [b_0, a_1, b_1, a_2, b_2, ..., a_N, b_N]
"""

import numpy as np
from numpy import linalg as la
np.set_printoptions(precision=3)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_reg_func(t, f, N):
    # サンプル点の数の取得
    K = len(f)

    # 行列Aの計算
    A = np.empty((K,2*N+1), float)
    A[:, 0] = np.cos(0*t)
    for n in range(1, N+1):
        A[:, 2*n-1] = np.sin(n*t)
        A[:, 2*n]   = np.cos(n*t)

    # 回帰係数wの計算
    f = f.reshape(-1, 1)            # 列ベクトル化
    I = np.eye(A.shape[1])                   # 単位行列
    λ = 0.0000000001
    w = la.solve(A.T@A + λ*I, A.T@f)      # w = (A^T A + λI)^{-1} A^T f

    # 回帰関数を定義(クロージャ)
    def calc_reg_func(t, w):
        # 回帰関数の計算
        func = w[0]*np.cos(0*t)
        for n in range(1, N+1):
            func += w[2*n-1]*np.sin(n*t) + w[2*n]*np.cos(n*t)
        return func

    # 係数をセットした回帰関数を作成
    g = lambda t: calc_reg_func(t, w)
    return g

def plot_animation(t, x, y, x_s, y_s):
    fig = plt.figure()
    ims = []
    for i in range(0, len(t), int(len(t)/100)):
        if i % 100 == 0:
            print(i)
        im = plt.scatter(x_s, y_s, c='r', s=20)
        im = plt.plot(x[:i], y[:i], c='b')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    # ani.save('anim.mp4')

if __name__ == '__main__':
    data = np.loadtxt("path1.csv", delimiter=",")
    # data = np.loadtxt("path2.csv", delimiter=",")
    data = np.r_[data, np.flipud(data)]
    K = data.shape[0] # サンプリング数
    N = 5            # 三角関数の級数の個数

    t = np.linspace(0, 2*np.pi, K)
    f_x = data[:, 0]
    f_y = data[:, 1]
    g_x = get_reg_func(t, f_x, N)
    g_y = get_reg_func(t, f_y, N)

    plt.scatter(f_x, t, c='r') ###
    t = np.linspace(0, 2*np.pi, 10000)
    plt.plot(g_x(t), t, c='b') ###
    plt.show() ###

    plot_animation(t, g_x(t), g_y(t), f_x, f_y)
    plt.show()