import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm
import argparse

def read_mat():
    train = sio.loadmat('../input/train.mat')
    test = sio.loadmat('../input/test.mat')
    reference = sio.loadmat('../input/reference.mat')
    return train, test, reference

def resize_image(img_array):
    return cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)

def create_img(df, user_id, trial_num, BASE_DIR, data_type, reverse, graph_type, fig=None, axes=None):
    npy_path = f'{BASE_DIR}/user_{user_id}/{data_type}/npy'
    png_path = f'{BASE_DIR}/user_{user_id}/{data_type}/png'

    os.makedirs(npy_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)

    # プロットの準備
    if fig is None or axes is None:
        fig, axes = plt.subplots(16, 1, figsize=(8, 8))
    else:
        for ax in axes.flat:
            ax.clear()  
    plt.subplots_adjust(wspace=0, hspace=0)
    
    results = []

    # スタンスを考慮
    condition = (user_id == "0001") != reverse
    indices = range(len(axes.flat)) if condition else [i+1 if i % 2 == 0 else i - 1 for i in range(len(axes.flat))]

    if graph_type == 0:
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.viridis  # 色マップの選択    
        for i, ax in zip(indices, axes.flat):
            sig = df[trial_num][i]
            color = cmap(norm(sig.max() - sig.min()))
            ax.plot(sig, color=color)
            ax.axis('off')
            results.append(sig)
    elif graph_type == 1:
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta", "lime", "teal", "coral", "gold", "navy"]
        for ii, (i, ax) in enumerate(zip(indices, axes.flat)):
            sig = df[trial_num][i]
            sig = np.clip(sig, -0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.plot(sig, color=colors[ii])
            ax.axis('off')
            results.append(sig)
 
    # 余白の削除
    fig.tight_layout(pad=0)  
    fig.canvas.draw()

    plt.savefig(f"{png_path}/trial_{trial_num}.png", bbox_inches='tight')
    
    # キャンバスからRGBデータを取得して配列に変換
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 224x224にリサイズ
    img = resize_image(img_array)

    # チャンネルを最初に
    img_array = np.array(img).transpose((2, 0, 1)) 

    # npyファイルとして保存
    np.save(f"{npy_path}/trial_{trial_num}.npy", img_array)
    
    return fig, axes


if __name__ == "__main__":
   # 引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument('--output_type', type=int, required=True) # 0 or 1
    parser.add_argument('--reverse', type=int, required=True) # 0 or 1

    # 引数を解析
    args = parser.parse_args()
    
    out_dir = f'./feature/type_{args.output_type}_rev_{args.reverse}'
    
    # フォルダが存在しない場合は作成
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    train, test, reference = read_mat()

    # 画像の作成、保存
    users = ['0001', '0002', '0003', '0004']
    fig = None
    axes = None
    for user_id in tqdm(users):
        length = len(train[user_id][0][0][0])
        for i in range(length):
            fig,axes = create_img(train[user_id][0][0][0], user_id, i, out_dir,'train', args.reverse, args.output_type, fig, axes)
        length = len(test[user_id][0][0][0])
        for i in range(length):
            fig,axes = create_img(test[user_id][0][0][0], user_id, i, out_dir,'test', args.reverse, args.output_type, fig, axes)

    user_id = "0005"
    length = len(reference[user_id][0][0][0])
    fig = None
    axes = None
    for i in range(length):
        fig,axes = create_img(reference[user_id][0][0][0], user_id, i, out_dir, 'train', args.reverse, args.output_type, fig, axes)
        
    length = len(reference[user_id][0][0][2])
    for i in range(length):
        fig,axes = create_img(reference[user_id][0][0][2], user_id, i, out_dir, 'test', args.reverse, args.output_type, fig, axes)
        