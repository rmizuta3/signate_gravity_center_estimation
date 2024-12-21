Signate [Motion Decoding Using Biosignals スケートボーダー重心位置予測チャレンジ](https://signate.jp/competitions/1430)の使用コードです。

解法については下記に記載しています。
- https://zenn.dev/rmizuta/articles/f24d888ad6963b

### ファイルの説明
- preprocess.py
	- ../input/ 配下にtrain.mat等の入力ファイルが必要
	- モデルへの入力画像作成用。入力波形は画像にしてnpyファイルとpngファイルを生成
- train.py
	- モデルの学習用
	- preprocessで生成した入力特徴を読み込み、モデルと予測値を出力
- make_submission.ipynb
	- モデルの出力ファイルからsubmitするファイルを生成
- valrmse.csv
  - trainから外れ値のデータを除外するためのファイル。
  - 全データ学習におけるout of foldの予測値
  
### 実行手順
1. 画像特徴の作成
- python preprocess.py --output_type 0 --reverse 0; 
- python preprocess.py --output_type 0 --reverse 1; 
- python preprocess.py --output_type 1 --reverse 0; 
- python preprocess.py --output_type 1 --reverse 1

実行すると、./feature/type_{type}\_rev_{reverse}というファイルが生成されます。output_type 1,2の違いは生成する画像特徴の違いで、reverseは左右を反転するか否かです。

2. モデルの学習
- python train.py --epochs 1 --exp type_1_0 --seed 42 --model_name efficientnet_b0.ra_in1k  --aug 1 --fivefold 1 --add_ref 1 --type 0
- python train.py --epochs 1 --exp type_1_1 --seed 42 --model_name efficientnet_b0.ra_in1k  --aug 1 --fivefold 1 --add_ref 1 --type 1

実行すると、./output/{exp}/ に学習済みモデルと予測値が格納されます。

3. 推論
make_submission.ipynbを上から順に実行します。
./submit/best_submission.json がsubmissionできるファイルとして生成されます。
