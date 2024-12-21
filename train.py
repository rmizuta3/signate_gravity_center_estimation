import os
import gc
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy.signal import resample
import glob
import timm
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from PIL import Image
import cv2

class HMSModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, in_channels: int, num_users: int, embedding_dim: int):
        super().__init__()
        
        # 画像モデルの作成 (timmを使用)
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            drop_rate=0.5, # 0.5
            drop_path_rate=0.2, #0.2
            num_classes=0  # 最終層は自分で定義するため0に設定
        )
        

        # 埋め込み層: user_id をベクトルに変換
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # モデルの出力次元数を取得し、user_idとの結合後の全結合層を定義
        num_features = self.model.num_features
        self.fc = nn.Linear(num_features + embedding_dim, 90)  

        # ドロップアウト層の追加
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, user_id):
        # 画像特徴を取得
        image_features = self.model(x)
        
        # user_idを埋め込みベクトルに変換
        user_features = self.user_embedding(user_id)
        
        # 画像特徴とuser_idの埋め込みを結合
        combined_features = torch.cat((image_features, user_features), dim=1)
        
        # ドロップアウトを適用
        combined_features = self.dropout(combined_features)

        # 結合した特徴量を使って予測
        output = self.fc(combined_features)
        
        return output
        #return image_features



def read_mat():
    train = sio.loadmat('../input/train.mat')
    test = sio.loadmat('../input/test.mat')
    reference = sio.loadmat('../input/reference.mat')
    return train, test, reference

def read_data(aug=1, add_ref=0, type=0):
    print("aug:", aug)
    
    if aug == 1:
        train_l = glob.glob(f"./feature/type_{type}_rev_0/*/train/npy/*") + glob.glob(f"./feature/type_{type}_rev_1/*/train/npy/*")
        test_l = glob.glob(f"./feature/type_{type}_rev_0/*/test/npy/*") + glob.glob(f"./feature/type_{type}_rev_1/*/test/npy/*")
    else:
        train_l = glob.glob(f"./feature/type_{type}_rev_0/*/train/npy/*")
        test_l = glob.glob(f"./feature/type_{type}_rev_0/*/test/npy/*")

    train_l.sort(key=lambda x: (int(x.split("/")[-4].split("_")[-1]), int(x.split("_")[-1].replace(".npy", ""))))
    test_l.sort(key=lambda x: (int(x.split("/")[-4].split("_")[-1]), int(x.split("_")[-1].replace(".npy", ""))))
    
    #referenceのtestデータを学習に含めるケース
    if add_ref==1:
        if aug == 1:
            ref_l = glob.glob(f"./feature/type_{type}_rev_0/user_0005/test/npy/*") + glob.glob(f"./feature//type_{type}_rev_1/user_0005/test/npy/*")
        else:
            ref_l = glob.glob(f"./feature/type_{type}_rev_0/user_0005/test/npy/*")
        ref_l.sort(key=lambda x: int(x.split("_")[-1].replace(".npy", "")))
        train_l += ref_l

    train_npy = np.array([np.load(f) for f in train_l])
    test_npy = np.array([np.load(f) for f in test_l])
    
    group_id = [f"{f.split('/')[-4].split('_')[-1]}_{f.split('_')[-1].replace('.npy', '')}" for f in train_l]
    group_id = pd.factorize(group_id)[0]
    
    train_user_id = np.array([int(f.split("/")[-4].split("_")[-1])-1 for f in train_l])
    test_user_id = np.array([int(f.split("/")[-4].split("_")[-1])-1 for f in test_l])
    
    return train_npy, test_npy, train_user_id, test_user_id, group_id

def get_target_data_full(aug=1, add_ref=0):
    targets = []
    for user_id in ["0001", "0002", "0003", "0004"]:
        targets.append(train[user_id][0][0][1].reshape(-1, 90))
    targets.append(reference["0005"][0][0][1].reshape(-1, 90))
    targets = np.concatenate(targets, axis=0)
    if add_ref == 1:
        tmp = reference["0005"][0][0][3]
        tmp[:,0,:] *= -1
        tmp[:,1,:] *= -1
        targets = np.concatenate([targets, tmp.reshape(-1, 90)], axis=0)

    if aug == 1:
        targets = np.repeat(targets, 2, axis=0)
    return targets

def custom_loss_function(outputs, targets):
    # 出力を(30,3)に変換
    batch_size = outputs.size(0)
    outputs = outputs.view(batch_size, 30, 3)
    targets = targets.view(batch_size, 30, 3)

    # 各地点ごとの MSE を計算 (batch, 30, 3) -> (batch, 30)
    mse_per_location = torch.sum((outputs - targets) ** 2, dim=-1)
    
    # 各地点ごとの MSE の平均を取り、全体の RMSE を計算 (batch, 30) -> (batch,)
    mean_mse_per_batch = torch.mean(mse_per_location, dim=-1)
    
    # バッチごとの MSE を平方根にしてRMSEを計算 (batch,) -> (1,)
    rmse = torch.mean(torch.sqrt(mean_mse_per_batch))

    return rmse

def initialize_wandb(config,fold):
    wandb.init(
        project="signate_kickboard",
        config={
            "exp": config.exp,
            "seed": config.seed,
            "model_name": config.model_name,
            "in_chans": config.in_chans,
            "num_users": config.num_users,
            "embedding_dim": config.embedding_dim,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "fold": fold,
        }
    )

def resize_image(img_array,size=(224,224)):
    return cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)


class CustomDataset(Dataset):
    def __init__(self, img_npy, labels, users, transform=None):
        self.img_npy = img_npy
        self.labels = labels
        self.users = users
        self.transform = transform

    def __len__(self):
        return len(self.img_npy)

    def __getitem__(self, idx):
        image = self.img_npy[idx] # (3,224,224)
        image = image.transpose((1, 2, 0))  # チャンネルを最後に (224, 224, 3)
      
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # 元に戻す
        image = image.transpose((2, 0, 1))  # チャンネルを最初に (3, 224, 224)

        image = torch.from_numpy(image).float().to(device)

        return image, self.labels[idx], self.users[idx]

if __name__ == "__main__":
   # 引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument('--exp', type=str, default="test", help='Experiment name')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--in_chans', type=int, default=3, help='Number of input channels')
    parser.add_argument('--num_users', type=int, default=5, help='Number of users')
    parser.add_argument('--embedding_dim', type=int, default=10, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--aug', type=int, default=1, help='Augmentation')
    parser.add_argument('--fivefold', type=int, default=0, help='5fold')
    parser.add_argument('--add_ref', type=int, default=0, help='add reference data')
    parser.add_argument('--type', type=str, required=True, help='type')
    parser.add_argument('--wandb', type=int, default=0, help='use wandb')

    # 引数を解析
    args = parser.parse_args()
    print("aug",args.aug, "fivefold", args.fivefold, "add_ref", args.add_ref)

    class CFG:
        exp = args.exp
        seed = args.seed
        device = torch.device("cuda")
        model_name = args.model_name
        in_chans = args.in_chans
        num_users = args.num_users  # ユーザーの数に応じて設定
        embedding_dim = args.embedding_dim  # 埋め込みベクトルの次元数
        batch_size = args.batch_size
        epochs = args.epochs
        aug = args.aug
        fivefold = args.fivefold
        add_ref = args.add_ref
        type = args.type
        use_wandb = args.wandb


    # transformの設定
    transform_params = {    
        "num_masks_x": 5,
        "num_masks_y": 5,    
        "mask_y_length": (10, 20),
        "mask_x_length": (10, 20),
        "fill_value": 0,  
    }
    transform = A.Compose([A.XYMasking(**transform_params, p=1)])

    # データの読み込み
    train, test, reference = read_mat()
    # 除外データの読み込み
    valrmse = pd.read_csv(f"./val_rmse.csv")
    remove_idx = list(valrmse[valrmse["rmse"] > 2.0].index) 
    if CFG.aug == 1:
        remove_idx = [i*2 for i in remove_idx] + [i*2+1 for i in remove_idx]

    train_npy, test_npy, train_user_id, test_user_id, group_id = read_data(aug=CFG.aug, add_ref=CFG.add_ref, type=CFG.type)
    
    # shapeの確認
    print(train_npy.shape, test_npy.shape, train_user_id.shape, test_user_id.shape, group_id.shape, len(remove_idx))

    #train_npy, train_user_id, group_idからremove_idxを削除
    train_npy = np.delete(train_npy, remove_idx, axis=0)
    train_user_id = np.delete(train_user_id, remove_idx)
    group_id = np.delete(group_id, remove_idx)
    
    targets = get_target_data_full(aug=CFG.aug, add_ref=CFG.add_ref)
    # remove_idxを削除
    targets = np.delete(targets, remove_idx, axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets_tensor = torch.from_numpy(targets).float().to(device)
    user_ids_tensor = torch.from_numpy(np.array(train_user_id)).long().to(device)
    test_user_ids_tensor = torch.from_numpy(np.array(test_user_id)).long().to(device)


    # KFoldの設定

    #group_kfold = GroupKFold(n_splits=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=CFG.seed) # group kfold
    best_epoch = -1
    best_model_state = None

    val_pred = np.zeros((len(train_user_id), 90))
    test_preds = np.zeros((len(test_user_id), 90, kf.get_n_splits()))  

    group_unique_ids = np.unique(group_id)
    for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(group_unique_ids)):
        tr_groups, va_groups = group_unique_ids[tr_group_idx], group_unique_ids[va_group_idx]
        #print(f'tr_groups: {tr_groups}, va_groups: {va_groups}')

        # 各レコードがtrain/validのどちらに属しているかによって分割する
        train_indices = np.isin(group_id, tr_groups)
        val_indices = np.isin(group_id, va_groups)
        
        best_val_loss = float('inf')
        print(f'Fold {fold+1}')

        #wandb
        if CFG.use_wandb == 1:
            initialize_wandb(CFG, fold)

        # データセットとデータローダーの作成
        train_dataset = CustomDataset(train_npy[train_indices], targets_tensor[train_indices], user_ids_tensor[train_indices], transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, shuffle=True)
    
        validation_dataset = CustomDataset(train_npy[val_indices], targets_tensor[val_indices], user_ids_tensor[val_indices])
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=CFG.batch_size, shuffle=False)

        test_loader = DataLoader(dataset=CustomDataset(test_npy, targets_tensor, test_user_ids_tensor), batch_size=CFG.batch_size, shuffle=False)

        # モデルの初期化
        model = HMSModel(
                model_name=CFG.model_name,
                pretrained=True,
                in_channels=CFG.in_chans,
                num_users=CFG.num_users,
                embedding_dim=CFG.embedding_dim
            ).to(CFG.device)
        criterion = custom_loss_function
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)        
        num_epochs = CFG.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            # トレーニングループ内での使用
            for inputs, targets, user_ids in train_loader:
                inputs, targets, user_ids = inputs.to(device), targets.to(device), user_ids.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, user_ids)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  

            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_inputs, val_targets, user_ids in validation_loader:
                    val_inputs, val_targets, user_ids  = val_inputs.to(device), val_targets.to(device), user_ids.to(device)
                    val_outputs = model(val_inputs, user_ids)
                    v_loss = criterion(val_outputs, val_targets)
                    val_loss += v_loss.item()

            avg_val_loss = val_loss / len(validation_loader)

            if CFG.use_wandb == 1:
                wandb.log({"train_rmse": avg_train_loss, "valid_rmse": avg_val_loss})

            # スケジューラのステップを実行
            scheduler.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # ベストモデルの保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()

            if CFG.use_wandb == 1:
                wandb.log({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
            # メモリの解放
            torch.cuda.empty_cache()

        # 最良のモデル状態で検証データに対する推論を実行
        model.load_state_dict(best_model_state)
        model.eval()
        
        with torch.no_grad():
            predictions = []
            for val_inputs, _, user_id in validation_loader:
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs, user_id)
                predictions.append(val_outputs)
        

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        val_pred[val_indices,:] = predictions#.flatten()
        
        
        with torch.no_grad():
            predictions = []
            for test_inputs, _ ,user_id in test_loader:
                test_inputs = test_inputs.to(device)
                test_outputs = model(test_inputs, user_id)
                predictions.append(test_outputs)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()#.flatten()
        test_preds[:, :, fold] = predictions.reshape(-1, 90)  # 形状を合わせて代入

        torch.cuda.empty_cache()

         # 最良のエポックとその時の検証損失を表示
        print(f'Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}')

        # 最良のモデル状態を保存
        model_save_path = f'./output/{CFG.exp}/models/epoch_{best_epoch}_valloss_{best_val_loss:.4f}_{fold}.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(best_model_state, model_save_path)

        # wandbの終了処理
        if CFG.use_wandb == 1:
            wandb.finish()

        if CFG.fivefold == 0:
            break
        
    # 最良のモデル状態を保存
    result_save_path = f'./output/{CFG.exp}/preds/test_preds.npy'
    val_save_path = f'./output/{CFG.exp}/valpreds/val_preds.npy'

    os.makedirs(os.path.dirname(val_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    np.save(result_save_path, test_preds)
    np.save(val_save_path, val_pred)
