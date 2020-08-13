import sys, os
import re

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import fourier_regression as freg

def plot_animation(data_list, frame_num=100, save=False):
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.98)
    ax1 = plt.subplot(2, 2, 1) # y-tグラフ
    ax2 = plt.subplot(2, 2, 2) # y-xグラフ
    ax4 = plt.subplot(2, 2, 4) # x-tグラフ(縦向き)

    # 各フレームごとに描画
    cmap = plt.get_cmap("tab10")    # 10色のカラーマップを取得
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
            im1 = ax1.plot(t[:i], y[:i], c=cmap(k%10))[0]
            im2 = ax2.plot(x[:i], y[:i], c=cmap(k%10))[0]
            im4 = ax4.plot(x[:i], t[:i], c=cmap(k%10))[0]
            im_list += (im1, im2, im4)  # リストの結合
        ims.append(im_list)             # フレーム追加

    # 静止のために最終フレームを1秒分だけ追加
    for i in range(int(frame_num/10)):
        ims.append(im_list)

    ani = animation.ArtistAnimation(fig, ims, interval=int(10000/frame_num))
    if save:
        ani.save('time_animation.mp4')
    plt.show()


if __name__ == '__main__':
    # 各ファイルからデータ読み込み, 回帰曲線の生成
    data_list = []
    for filename in os.listdir('./'):
        if re.match(r'extracted_path(\d+).csv', filename):
            print('file', filename)
            t_s, x_s, y_s = freg.load_points(filename)
            t, x, y = freg.get_2D_regression_curve(t_s, x_s, y_s, M=int(len(t_s)/2))
            data_list.append((t_s, x_s, y_s, t, x, y))

    # プロット
    # plot_animation(data_list, frame_num=10)
    plot_animation(data_list, frame_num=20, save=True)