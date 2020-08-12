"""
liner regression by approximate fourier series

sample of f                : f(t_1), ..., f(t_N)
approximate fourier series : g(t) = Σ_{m=0}^M (a_m sin(mx) + b_m cos(mx))
loss function              : Σ_{n=1}^N (f(t_n) - g(t_n))^2 = || bm{f} - Aw ||_2^2

    [cos(0t_1), sin(t_1), cos(t_1), sin(2t_1), cos(2t_1), ..., sin(Mt_1), cos(Mt_1)]
A = [cos(0t_2), sin(t_2), cos(t_2), sin(2t_2), cos(2t_2), ..., sin(Mt_2), cos(Mt_2)]
    [ ...     ,                                                         , ...      ]
    [cos(0t_N), sin(t_N), cos(t_N), sin(2t_N), cos(2t_N), ..., sin(Mt_N), cos(Mt_N)]

w = [b_0, a_1, b_1, a_2, b_2, ..., a_M, b_M]
"""

import numpy as np
from numpy import linalg as la
np.set_printoptions(precision=3)

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_reg_func(t_s, f_s, M):
    # サンプル点の数の取得
    N = len(f_s)

    # 行列Aの計算
    A = np.empty((N,2*M+1), float)
    A[:, 0] = np.cos(0*t_s)
    for m in range(1, M+1):
        A[:, 2*m-1] = np.sin(m*t_s)
        A[:, 2*m]   = np.cos(m*t_s)

    # 回帰係数wの計算
    f_s = f_s.reshape(-1, 1)    # 列ベクトル化
    I = np.eye(A.shape[1])      # 単位行列
    λ = 0.0000000001
    w = la.solve(A.T@A + λ*I, A.T@f_s) # w = (A^T A + λI)^{-1} A^T f_s

    # 回帰関数を定義(クロージャ)
    def calc_reg_func(t, w):
        # 回帰関数の計算
        func = w[0]*np.cos(0*t)
        for m in range(1, M+1):
            func += w[2*m-1]*np.sin(m*t) + w[2*m]*np.cos(m*t)
        return func

    # 係数をセットした回帰関数を作成
    g = lambda t: calc_reg_func(t, w)
    return g

def plot_animation(t, x, y, t_s, x_s, y_s, frame_num=100):
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.98)
    ax1 = plt.subplot(2, 2, 1) # y-tグラフ
    ax2 = plt.subplot(2, 2, 2) # y-xグラフ
    ax4 = plt.subplot(2, 2, 4) # x-tグラフ(縦向き)

    ims = []
    for i in range(0, len(t)+int(len(t)/frame_num), int(len(t)/frame_num)):
        print(i)
        ax1.scatter(t_s, y_s, c='r')
        ax2.scatter(x_s, y_s, c='r')
        ax4.scatter(x_s, t_s, c='r')
        im1 = ax1.plot(t[:i], y[:i], c='b')[0]
        im2 = ax2.plot(x[:i], y[:i], c='b')[0]
        im4 = ax4.plot(x[:i], t[:i], c='b')[0]
        ims.append([im1, im2, im4])

    ani = animation.ArtistAnimation(fig, ims, interval=int(10000/frame_num))
    ani.save('animation.mp4')

if __name__ == '__main__':
    # data = np.loadtxt("sample_path1.csv", delimiter=",")
    # data = np.loadtxt("sample_path2.csv", delimiter=",")
    data = np.loadtxt("extracted_path3.csv", delimiter=",")
    data = np.r_[data, data[0,:].reshape(1,-1)] # 始点を終点として追加
    N = data.shape[0] # サンプリング数
    M = 100             # 三角関数の級数の個数

    t_s = np.linspace(0, 2*np.pi, N)
    x_s = data[:, 0]
    y_s = data[:, 1]

    g_x = get_reg_func(t_s, x_s, M)
    g_y = get_reg_func(t_s, y_s, M)
    t = np.linspace(0, 2*np.pi, 10000)
    x = g_x(t)
    y = g_y(t)

    plot_animation(t, x, y, t_s, x_s, y_s, frame_num=30)
    plt.show()