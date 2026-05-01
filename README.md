# SwinIR Image Upconvert GUI for Windows

このツールは、**顔を作り替えない低侵襲な高解像度化**を目的にした Windows 用 GUI です。

- 顔補正ツールではありません
- 美顔化はしません
- 顔復元はしません
- 生成補完はしません
- Stable Diffusion 系の再生成はしません
- GFPGAN / CodeFormer / SUPIR / StableSR / img2img は実装していません

目的は、人物参照 core 候補の画像をできるだけ元の顔立ちのまま拡大し、比較確認しやすくすることです。

## 何を使っているか

- GUI: `Python + PySide6`
- 推論エンジン: `vendor/SwinIR`
- モデル: **SwinIR 公式の学習済みモデル**
- 推論バックエンド: `PyTorch`

独自学習、追加学習、ファインチューニングは行いません。

## 特徴

- 入力フォルダ選択
- 出力フォルダ選択
- モデルファイル選択
- 倍率選択: `2x / 4x`
  - ただし GUI に表示される倍率は、実際に配置済みのモデルがあるものだけです
- 対応拡張子: `.jpg .jpeg .png .webp`
- 出力形式: `PNG`
- 処理済みファイルのスキップ
- サブフォルダ処理
- `tile size` / `tile overlap` 指定
- CPU / GPU 状況表示
- 処理開始 / 停止 / キャンセル
- 進捗バー表示
- 現在処理中ファイル名の表示
- エラー時も GUI 全体は落ちにくい設計
- `processing_log.csv` / `failed_log.csv` 出力
- 失敗ファイルを `failed` フォルダへコピー
- 透過 PNG の alpha 保持
- 元画像を上書きしない
- `テスト処理` で先頭 5 枚だけ先に確認可能

## 初期値

- 倍率: `2x`
- 出力形式: `PNG`
- tile size: `400`
- tile overlap: `32`
- 処理済みファイルはスキップ: `ON`
- サブフォルダ処理: `OFF`

## モデルの考え方

人物 core 用途では、まず **classical_sr 系の 2x** を推奨します。

- `classical_sr`
  - 比較的素直な拡大向け
  - まず最初に試す推奨系統
- `real_sr`
  - 劣化の強い写真向け
  - ただし見た目の変化が大きくなる可能性あり
  - GUI 上では比較用の位置づけ
- `lightweight_sr`
  - 軽いが画質上限は下がる可能性あり

最初の推奨モデル:

- `models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth`

モデル置き場の詳細は [models/swinir/README.md](models/swinir/README.md) を参照してください。

## 透過 PNG について

- alpha チャンネルは保持します
- SwinIR へは RGB 部分だけを渡します
- alpha は別経路で Lanczos 拡大します
- 最後に RGB と alpha を再合成して RGBA PNG として保存します
- 透明部分の色に引っ張られて白フチや黒フチが出にくいよう、完全透明部には近傍色をにじませてから RGB 拡大します

## 導入手順

### 1. Python を入れる

まず Python をインストールしてください。

推奨:

- Python `3.10` から `3.13`

`setup.bat` は Python が見つからない場合、または想定外バージョンしか無い場合に止まります。

### 2. `setup.bat` を実行する

初回は **必ず `setup.bat`** を実行してください。

このスクリプトは次を行います。

- `%LOCALAPPDATA%\imageUpconvert\venv` に venv 作成
- `requirements.txt` インストール
- CPU 版 PyTorch の最小構成インストール
- 簡易環境確認

注意:

- `setup.bat` は GPU 版 PyTorch を無理に自動判定しません
- 最初は CPU で確実に動くことを優先しています

### 3. 公式学習済みモデルを置く

`models/swinir/` に、**SwinIR 公式リリースの `.pth`** を置いてください。

公式:

- https://github.com/JingyunLiang/SwinIR/releases

ファイル名は変更せず、そのまま置いてください。GUI は公式ファイル名からモデル種別を判定します。

### 4. `run_gui.bat` を実行する

起動は **`run_gui.bat` のダブルクリック** です。

このバッチは `%LOCALAPPDATA%\imageUpconvert\venv` の Python を使って GUI を起動します。

## GPU を使いたい場合

CPU 版の代わりに GPU 版 PyTorch を使いたい場合は、**PyTorch 公式のインストールページ**で自分の CUDA 環境に合うコマンドを確認してください。

- https://pytorch.org/get-started/locally/

方針:

- `setup.bat` は GPU を決め打ちしません
- CUDA バージョンに合わないコマンドを自動で入れないためです
- GUI では `torch.cuda.is_available()` の結果を表示します

## 使い方

1. `run_gui.bat` を起動
2. 入力フォルダを選ぶ
3. 出力フォルダを選ぶ
4. モデルファイルを選ぶ
5. 倍率を選ぶ
6. 必ず先に **`テスト処理`** を実行して 5 枚だけ確認
7. 問題なければ `Start` で一括処理

**大量処理前に必ず「テスト処理」で 5 枚だけ確認してください。**

テスト処理の出力先は、選択した出力フォルダの兄弟に `*_test` として作られます。

例:

- 出力先: `output`
- テスト出力先: `output_test`

## 安全設計

- 元画像は絶対に上書きしません
- 出力先が入力先と同じ場合は止めます
- 出力先が入力フォルダの中にある場合も止めます
- 出力名は元名を維持し、末尾に `_swinir_x2` などを付けます
- 同名出力がある場合は `Skip` または `Serial` を選べます
- 日本語ファイル名・長めのパスを想定して `pathlib` ベースで処理しています

## ログ

出力フォルダに次の CSV を作成します。

- `processing_log.csv`
- `failed_log.csv`

失敗ファイルは出力フォルダ配下の `failed/` にコピーします。

`processing_log.csv` の主な列:

- 処理日時
- 入力ファイルパス
- 出力ファイルパス
- 元画像サイズ
- 出力画像サイズ
- 倍率
- 使用モデル
- tile size
- tile overlap
- alpha 有無
- 処理結果
- エラー内容
- 処理時間

## CLI での処理

GUI の前にコマンドラインで試したい場合は、次も使えます。

```bat
venv\Scripts\python.exe -m app.swinir_runner ^
  --input-dir "C:\path\to\input" ^
  --output-dir "C:\path\to\output" ^
  --model-path "H:\DropBox\Dropbox\70_MSVS\01.HOBBY\imageUpconvert\models\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth" ^
  --scale 2 ^
  --tile-size 400 ^
  --tile-overlap 32 ^
  --skip-existing
```

## リポジトリ構成

```text
app/
  main.py
  gui.py
  swinir_runner.py
  image_io.py
  alpha_utils.py
  log_utils.py
  env_check.py
vendor/
  SwinIR/
models/
  swinir/
output/
failed/
requirements.txt
setup.bat
run_gui.bat
README.md
```

## upstream 取り込み

SwinIR 本体は `vendor/SwinIR/` に upstream スナップショットとして取り込んでいます。

- upstream repo: https://github.com/JingyunLiang/SwinIR
- vendored commit: `6545850fbf8df298df73d81f3e8cba638787c8bd`

詳細は `vendor/SwinIR_UPSTREAM.txt` を参照してください。
