# 3D Gaussian splatting (original) を windows 11 で動かす方法（書いてる途中）
Edit 2025-09-30:


## はじめに
3D gaussian splatting は、動画や写真からフォトリアリスティックな 3D 映像空間を再構成する人気の手法ですが、多くのオープンソースソフトウェアと同じく、windows 上で動かすために試行錯誤が必要となります。ここでは、windows 11 Home を前提として、各種ソフトウェアのインストールから、3D Gaussian splatting を動作させるまでの手順についてまとめます。

なお、miniconda など、 python 開発用仮想環境のセットアップについては完了していることを前提に話を進めます。

<!-- Here's a summary I've made of all past issues relating to simple-knn and diff-gaussian-rasterization. It eventually worked for me after 4 hours, 10+ reboots, and back-rolling VS2022 Community to an earlier version. -->

## 関連ソフトウェアのインストール
画像解像度の変更に使う imagemagik, ffmpeg や、画像からカメラパラメータを推定するための colmap が必要となるため、インストールした上でパスを通しておきます。インストールの方法については、[Jonathan Stephens 氏のビデオ](https://github.com/camenduru/gaussian-splatting-colab)が参考になります（インストールだけでなく、3D Gaussian Splatting を動かす方法についても）。
<!--
C:\work\research\colmap-x64-windows-cuda\bin
C:\Program Files\ImageMagick-7.1.2-Q16-HDRI -->

## 環境構築で起こる課題
3D Gaussian Splatting を実行する上で、"diff-gaussian-rasterization" と "simple-knn" が適切にインストールできないことがよくあるようで、ネット上でも issue として取り上げられています。以下、これらの問題を回避するための手順について記載します。

なお、以降の操作は、常に conda の仮想環境で実行するようにしてください。

まずは、公式リポジトリがサポートする CUDA toolkit をインストールしてください。サポートされているバージョンは、11.7 もしくは 11.8 とのこと。私は、11.7 をインストールしました。
> Note that cudatoolkit version in environment.yml is WRONG. Change it to be 11.7 or 11.8, not 11.6. Some reported problems using 11.8, but some succeeded with it.



次に Visual Studio 2022 をインストールします。Visual Studio にも不具合というか相性があるらしく、v17.10.3 だと動作せず、代わりに v17.6.4 だと動作するといわれています。なお、私は  Visual Studio Community 2022 (64 ビット) Version 17.9.6 をインストールして、問題なく動作しました。

なお、ツールのビルドに cl.exe（と C++ devtools）を使うので、これらが環境変数 PATH に入っている必要があります。というわけで、以下を PATH に加えます：
“C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64”

## 公式 Git リポジトリのクローン
不要なファイルが残っていると悪さをすることがあるので、以下を削除します。conda 仮想環境も同様に不要なものを完全に削除します。具体的には以下が削除対象です。
+ C:\Users\YOURNAME\\.conda\envs\gaussian_splatting
+ C:\Users\YOURNAME\gaussian-splatting

なお、conda 環境を削除する際には、以下のコマンドを先に実行するようにしてください。
```shell
> conda remove -n gaussian_splatting --all
```
ようやくリポジトリのクローンを行います。クローンの際には、`--recursive` オプションを忘れないようにしてください。でないと、`submodule` 以下の重要なモジュールがインポートされません。
```shell
> git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## 仮想環境の構築と pytorch のインストール
CUDA and PyTorch versions need to match: maybe use pip instead of conda. For example, for CUDA 11.7 (11.7.1 works):
```shell
> conda create -n gaussian_splatting python=3.7
> conda activate gaussian_splatting
> conda install -c conda-forge vs2022_win-64
> pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## サブモジュールと関連モジュールのインストール
```shell
> pip install submodules/diff-gaussian-rasterization
> pip install submodules/simple-knn
> pip install submodules/fused-ssim
> pip install plyfile
> pip install tqdm
> pip install python-opencv
```

# Pytorch の動作確認（必要に応じて）
Check if PyTorch can access CUDA:
```shell
(gaussian_splatting) c:\dev\gaussian-splatting> python
>>> import torch
>>> torch.cuda.is_available()
True
(then press Ctrl+Z to exit)
```

To restart an aborted installation:
conda env update --file environment.yml --name gaussian_splatting

# Training 前の注意
画像は連番でなければならないようなので、変換スクリプトを作成した：my_rename.py
```python
import os
import shutil
from pathlib import Path

# フォルダのパス
input_folder = Path("../south-building/input")
output_folder = Path("images")

# output フォルダが存在しなければ作成
output_folder.mkdir(parents=True, exist_ok=True)

# .jpg ファイルの一覧を取得して昇順にソート
jpg_files = sorted(input_folder.glob("*.JPG"))

# ファイルを番号付きでコピー
for i, jpg_file in enumerate(jpg_files):
    new_filename = f"{i:04}.jpg"  # 0000.jpg, 0001.jpg 形式
    destination = output_folder / new_filename
    shutil.copy2(jpg_file, destination)
    print(f"Copied: {jpg_file.name} → {new_filename}")

print("コピー完了。")
```

画像変換：
```shell
> python convert.py -s data\south-building
```

訓練：
```shell
> python train.py -s ..\south-building -r 4
```

## ビューワのインストールと実行
```shell
> pip install SIBR_viewers
> cmake -h
> cd SIBR_viewers
> cmake -Bbuild .
> viewers\bin\SIBR_gaussianViewer_app.exe -m output\0c114fdf-7
```

----
## 参考文献：
+ [情報源１](https://github.com/graphdeco-inria/gaussian-splatting/issues/865)

+ [情報源２](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file)


## 実行環境
+ Windows 11 home version
+ x64 system
+ CPU: Intel i9-10850K
+ GPU: NVIDIA RTX 3080
+ RAM: 64GB
