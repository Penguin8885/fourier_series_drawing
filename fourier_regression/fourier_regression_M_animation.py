import sys, os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import fourier_regression as freg

def plot_animation(data_list, save=False):
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.98)

    # 各フレームごとに描画
    ims = []
    for i, line_list in enumerate(data_list):
        print('frame', i)

        im_list = []
        for t, x, y in line_list:
            im = plt.plot(x, y, c='b')[0]
            im_list.append(im)          # リストの結合
        ims.append(im_list)             # フレーム追加

    # 静止のために最終フレームを1秒分だけ追加
    frame_num = len(data_list)
    for i in range(int(frame_num/10)):
        ims.append(im_list)

    ani = animation.ArtistAnimation(fig, ims, interval=int(10000/frame_num))
    if save:
        ani.save('M_animation.mp4')
    plt.show()


if __name__ == '__main__':
    # 各ファイルからデータ読み込み, 回帰曲線の生成
    data_list = []
    for i, ratio in enumerate([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.2, 0.5, 1]):
        print('frame', i)
        line_list = []
        for filename in os.listdir('./'):
            if re.match(r'extracted_path(\d+).csv', filename):
                t_s, x_s, y_s = freg.load_points(filename)
                t, x, y = freg.get_2D_regression_curve(t_s, x_s, y_s, M=int(len(t_s)/2 * ratio)+1)
                line_list.append((t, x, y))
        data_list.append(line_list)

    # プロット
    plot_animation(data_list, save=True)