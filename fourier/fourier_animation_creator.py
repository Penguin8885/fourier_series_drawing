import sys, os
import re

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_points(filename):
    # データの読み込み
    data = np.loadtxt(filename, delimiter=",")
    data = np.r_[data, data[0,:].reshape(1,-1)] # 始点を終点として追加（周期関数化）
    N = data.shape[0]                           # サンプリング数

    # 各サンプル点の読み込み
    t_s = np.linspace(0, 2*np.pi, N)
    x_s = data[:, 0]
    y_s = data[:, 1]

    return t_s, x_s, y_s

def get_2D_regression_curve(t_s, x_s, y_s, M):
    """
    Mは有限フーリエ級数の三角関数の個数
    """

    # 回帰関数の作成
    g_x = get_reg_func(t_s, x_s, M)
    g_y = get_reg_func(t_s, y_s, M)

    # 回帰曲線の作成
    t = np.linspace(0, 2*np.pi, 10000)
    x = g_x(t)
    y = g_y(t)

    return t, x, y

def get_reg_func(t_s, f_s, M):
    """
    liner regression by finite fourier series (FFS)

    sample of f           : f(t_1), ..., f(t_N)
    finite fourier series : g(t) = Σ_{m=0}^M (a_m cos(mt) + b_m sin(mt))

    loss function         : Σ_{n=1}^N (f(t_n) - g(t_n))^2 = || bm{f} - Aw ||_2^2
        [cos(0t_1), cos(t_1), sin(t_1), cos(2t_1), sin(2t_1), ..., cos(Mt_1), sin(Mt_1)]
    A = [cos(0t_2), cos(t_2), sin(t_2), cos(2t_2), sin(2t_2), ..., cos(Mt_2), sin(Mt_2)]
        [ ...     ,                                                ...      ,          ]
        [cos(0t_N), cos(t_N), sin(t_N), cos(2t_N), sin(2t_N), ..., cos(Mt_N), sin(Mt_N)]

    w = [a_0, a_1, b_1, a_2, b_2, ..., a_M, b_M]
    """

    # サンプル点の数の取得
    N = len(f_s)

    # 行列Aの計算
    A = np.empty((N,2*M+1), float)
    A[:, 0] = np.cos(0*t_s)
    for m in range(1, M+1):
        A[:, 2*m-1] = np.cos(m*t_s)
        A[:, 2*m]   = np.sin(m*t_s)

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
            func += w[2*m-1]*np.cos(m*t) + w[2*m]*np.sin(m*t)
        return func

    # 係数をセットした回帰関数を作成
    g = lambda t: calc_reg_func(t, w)
    return g

def plot_animation(data_list, frame_num=100, save=False):
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.98)
    ax1 = plt.subplot(2, 2, 1) # y-tグラフ
    ax2 = plt.subplot(2, 2, 2) # y-xグラフ
    ax4 = plt.subplot(2, 2, 4) # x-tグラフ(縦向き)

    color_list = ['k', 'r', 'g', 'b', 'y']
    ims = []
    for frame_n in range(frame_num):
        print('frame', frame_n)
        im_list = []
        for k, datum in enumerate(data_list):
            t_s, x_s, y_s, t, x, y = datum
            i = int(len(t) / (frame_num-1) * frame_n)
            ax1.scatter(t_s, y_s, c='gray', s=1)
            # ax2.scatter(x_s, y_s, c='gray', s=1)
            ax4.scatter(x_s, t_s, c='gray', s=1)
            im1 = ax1.plot(t[:i], y[:i], c=color_list[k])[0]
            im2 = ax2.plot(x[:i], y[:i], c=color_list[k])[0]
            im4 = ax4.plot(x[:i], t[:i], c=color_list[k])[0]
            im_list += (im1, im2, im4)  # リストの結合
        ims.append(im_list)             # フレーム追加

    ani = animation.ArtistAnimation(fig, ims, interval=int(10000/frame_num))
    if save:
        ani.save('animation.mp4')
    plt.show()


if __name__ == '__main__':
    # 各ファイルからデータ読み込み, 回帰曲線の生成
    data_list = []
    for filename in os.listdir('./'):
        if re.match(r'extracted_path(\d+).csv', filename):
            print('file', filename)
            t_s, x_s, y_s = load_points(filename)
            t, x, y = get_2D_regression_curve(t_s, x_s, y_s, M=int(len(t_s)/2))
            data_list.append((t_s, x_s, y_s, t, x, y))

    # プロット
    plot_animation(data_list, frame_num=10)
    # plot_animation(data_list, frame_num=100, save=True)