# WoLNamesBlackedOut

## 概要
FinalFantasy XIV（FF14）のキャプチャ動画からユーザ名を黒塗りする動画編集アプリです。<br>
MP4の動画を取り込み、Ultralytics YOLOv8でユーザ名を物体検出しています。<br>
動画の取り込み、書き出しはFFmpegを利用しています。<br>
UIにはFletを利用させていただいています。<br>
EXE化には、pyinstallerを利用させていただいています。<br>

## 動作環境
Windows11<br>
NVIDIA Geforce 2060以上を推奨（CUDA、NVENC利用）<br>
AMD Radeonでも動作すると思われる。未確認（ONNX、AMF利用）<br>
Python3.11<br>
Windowsでの実行EXEファイルのダウンロードはこちらから<br>
[ダウンロード](https://download-count-gpemgma5enb6cqe6.japaneast-01.azurewebsites.net/download_WoLNamesBlackedOut.php)

## コード利用上の注意点
次のファイルはファイルサイズ制限の問題から、githubではダミーファイルを置いています。
+ my_yolov8m.pt
+ my_yolov8m.onnx

ファイルが必要であれば、googleドライブよりダウンロード願います。<br>
[ダウンロード](https://download-count-gpemgma5enb6cqe6.japaneast-01.azurewebsites.net/download.php)

ffmpegのバイナリ（./ffmpeg/bin/）もファイルサイズ制限からgithubにありません。<br>
必要であれば、次のアドレスからダウンロードください。<br>
[Windows builds by BtbN](https://github.com/BtbN/FFmpeg-Builds/releases)

## pyinstallerによるexe化
```pyinstaller main.spec```
<br>exe実行時、起動（解凍）に時間を掛けたくなかったので、単体ファイルにはしていません。
ffmpegについてはexe化ではコピーしておらず、手でdist以下にフォルダコピーしています。

## Ultralytics YOLOv8での学習
以下を参考に学習しています。
https://docs.ultralytics.com/modes/train/#key-features-of-train-mode
<br>yolo_train.py
```python:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'   #エラー対策

from ultralytics import YOLO
model = YOLO('my_yolov8m.yaml')
model = YOLO("yolov8m.pt")

if __name__ == "__main__":
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='data.yaml',epochs=2000,batch=16,device=0,imgsz=1280,save_period=10,patience=150)

    # Evaluate the model's performance on the validation set
    results = model.val()
```
認識する物体はユーザ名の1つでよかったので、1つに編集しています。<br>
my_yolov8m.yaml（抜粋）
```python:
# Parameters
# nc: 80 # number of classes
nc: 1 # number of classes
```
画像とアノテーションのフォルダ指定。coco形式であったものをyolo形式に変換していたため、このようなフォルダ名になっています。環境に合わせて指定ください。<br>
data.yaml
```python:
path: D:\coco_converted # dataset root directory
train: images/train2017 # training images (relative to 'path')
val: images/val2017 # validation images (relative to 'path')
test: # optional test images
names:
    0: pc_name
```

## アノテーション
アノテーションには、[CVAT](https://www.cvat.ai/)を利用しました。<br>
セルフホストでの個人利用はフリーとなっています。<br>
dockerで利用しました。
