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
    f = f.reshape(-1, 1)    # 列ベクトル化
    I = np.eye(A.shape[1])  # 単位行列
    λ = 0.0000000001
    w = la.solve(A.T@A + λ*I, A.T@f) # w = (A^T A + λI)^{-1} A^T f

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

def plot_animation(t, x, y, t_s, x_s, y_s):
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.98)
    ax1 = plt.subplot(2, 2, 1) # y-tグラフ
    ax2 = plt.subplot(2, 2, 2) # y-xグラフ
    ax4 = plt.subplot(2, 2, 4) # x-tグラフ(縦向き)

    ims = []
    for i in range(0, len(t)+int(len(t)/100), int(len(t)/100)):
        print(i)
        ax1.scatter(t_s, y_s, c='r')
        ax2.scatter(x_s, y_s, c='r')
        ax4.scatter(x_s, t_s, c='r')
        im1 = ax1.plot(t[:i], y[:i], c='b')[0]
        im2 = ax2.plot(x[:i], y[:i], c='b')[0]
        im4 = ax4.plot(x[:i], t[:i], c='b')[0]
        ims.append([im1, im2, im4])

    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
    ani.save('animation.mp4')

if __name__ == '__main__':
    data = np.loadtxt("path1.csv", delimiter=",")
    # data = np.loadtxt("path2.csv", delimiter=",")
    data = np.r_[data, data[0,:].reshape(1,-1)] # 始点を終点として追加
    K = data.shape[0] # サンプリング数
    N = 5             # 三角関数の級数の個数

    t_s = np.linspace(0, 2*np.pi, K)
    x_s = data[:, 0]
    y_s = data[:, 1]

    g_x = get_reg_func(t_s, x_s, N)
    g_y = get_reg_func(t_s, y_s, N)
    t = np.linspace(0, 2*np.pi, 10000)
    x = g_x(t)
    y = g_y(t)

    plot_animation(t, x, y, t_s, x_s, y_s)
    plt.show()