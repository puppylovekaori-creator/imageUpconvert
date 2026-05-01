# SwinIR + GIMP 低侵襲高解像度化 GUI

このツールは、**顔を作り替えない低侵襲な高解像度化**を目的にした Windows 用 GUI です。

- 顔補正ツールではありません
- 顔復元はしません
- 美顔化はしません
- 生成補完はしません
- Stable Diffusion 系の再生成はしません
- GFPGAN / CodeFormer / SUPIR / StableSR / img2img は実装していません

目的は、人物参照 core 候補の画像を、GIMP で確認済みの軽いノイズ除去とアンシャープマスクで整え、必要なときだけ SwinIR を使って、**本人性を崩さずに** 見やすさ改善や高解像度化を行うことです。

## このツールの考え方

- 最優先は **本人性維持** です
- 強い補正より、顔立ち・輪郭・目鼻口・髪型の保持を優先します
- 通常画面には SwinIR モデル選択 UI を出していません
- SwinIR は内部既定値の **classical_sr x2 / x4** を使います
- ユーザーが主に調整するのは **GIMP 前処理のノイズ除去とアンシャープマスク** です
- 処理モードで **GIMPのみ / SwinIRのみ / GIMP前処理 + SwinIR / GIMP前処理 + SwinIR + GIMP後処理** を切り替えます

## 何を使っているか

- GUI: `Python + PySide6`
- 高解像度化: `vendor/SwinIR`
- モデル: **SwinIR 公式の学習済みモデル**
- 推論バックエンド: `PyTorch`
- GIMP 前後処理: **ローカルにインストールされた GIMP を外部コマンドとして呼び出し**
- 人物切り抜き: `rembg` (`u2net_human_seg`)

独自学習、追加学習、ファインチューニングは行いません。

## GIMP 依存について

このツールは **GIMP に依存する処理を含みます**。

- GIMP 未導入でも、**SwinIRのみ** モードなら SwinIR 単体処理はできます
- **GIMPのみ**、**GIMP前処理 + SwinIR**、**GIMP前処理 + SwinIR + GIMP後処理** は GIMP が必要です
- GIMP 前処理 / 後処理を使う場合は、GUI で **GIMP 実行ファイルパス** を指定してください
- 例:
  - `C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe`
  - `C:\Program Files\GIMP 2\bin\gimp-2.10.exe`
  - `C:\Program Files\GIMP 3\bin\gimp-console-3.exe`
  - `C:\Program Files\GIMP 3\bin\gimp.exe`

起動時にパスの存在確認を行い、未設定や不正パスなら GUI 上に分かりやすく表示します。

## 特徴

- GUI は `実行` タブと `設定` タブに分かれています
- GIMP 実行ファイルパス指定
- 入力フォルダ / 出力フォルダ指定
- プレビュー対象画像 1 枚指定
- 処理モード
  - `GIMPのみ`
  - `SwinIRのみ`
  - `GIMP前処理 + SwinIR`
  - `GIMP前処理 + SwinIR + GIMP後処理`
- プレビュー範囲選択
  - `全体`
  - `中央crop`
  - `顔付近crop`
- GIMP 前処理のノイズ除去
  - `OFF / 弱 / 中 / 強 / 詳細設定`
- GIMP 前処理のアンシャープマスク
  - `OFF / 弱 / 中 / 強 / 詳細設定`
- SwinIR 倍率
  - `2x / 4x`
- GIMP 後処理の強さ
  - `なし / 弱 / 中 / 強`
- 人物切り抜き ON / OFF
- プレビュー比較表示
  - 本体 GUI とは別ウィンドウで開く
  - 左側の一覧をクリックして切り替え
  - 元画像
  - GIMP 前処理後または GIMP処理後
  - SwinIR 後または SwinIRのみ参考
  - 最終出力
  - プレビュー画面から `表示中を保存` / `元画像を保存` / `横並び比較を保存` ができる
- `1枚比較処理`
  - `original`
  - `gimp_only`
  - `swinir_only`
  - `gimp_pre_swinir`
  - `gimp_pre_swinir_gimp_post`
  - `gimp_only_cutout` (`人物切り抜き ON のとき`)
  - `gimp_pre_swinir_cutout` (`人物切り抜き ON のとき`)
- `テスト処理`
  - 先頭 5 枚だけを `output_test` に出力
- 設定保存 / 次回起動時の復元
- CPU / GPU 状況表示
- 停止 / キャンセル
- 進捗バー
- 失敗時も GUI 全体が落ちにくい設計
- `processing_log.csv` / `failed_log.csv`
- 失敗ファイルを `failed/` にコピー
- 透過 PNG の alpha 保持
- 元画像は絶対に上書きしない

## SwinIR モデル

通常画面ではモデル選択を出していません。SwinIR を使うモードのときだけ、内部で次を使います。

- `models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth`
- `models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`

人物参照用途では、まず **2x** を推奨します。

- `2x`
  - 変化が比較的穏やか
  - まず確認する基準
- `4x`
  - さらに大きくなる
  - ファイルサイズがかなり増えやすい
  - 見た目の改善が小さい場合もある

## 処理モード

- `GIMPのみ`
  - GIMP のノイズ除去とアンシャープマスクだけで見やすさ改善します
  - SwinIR は使いません
- `SwinIRのみ`
  - GIMP を使わずに高解像度化だけ行います
- `GIMP前処理 + SwinIR`
  - GIMP 前処理で眠さを軽く取り、その後 SwinIR で拡大します
- `GIMP前処理 + SwinIR + GIMP後処理`
  - GIMP 前処理と SwinIR に加え、最後に軽い GIMP 後処理をかけます

人物切り抜きは処理モードとは別の ON / OFF です。

## GIMP 前処理 / 後処理

### GIMP 前処理

目的:

- 圧縮ノイズやざらつきの軽減
- 元画像の眠さを軽く取る

初期値:

- ノイズ除去: `弱`
- アンシャープマスク: `弱`

### GIMP 後処理

目的:

- SwinIR 後に軽く輪郭を整える

初期値:

- `なし`

### 強すぎる設定について

強いノイズ除去や強いシャープは

- 白フチ
- 黒フチ
- ハロ
- ギラつき
- ノイズ強調
- 暗部つぶれ
- 本人らしさの低下

を起こす可能性があります。

**顔が変わって見える設定は採用しないでください。**

## 人物切り抜き

- 人物切り抜きは `rembg` の `u2net_human_seg` を使います
- 出力は PNG 透過です
- 既に透過がある画像は、元の alpha と切り抜き alpha を重ねて扱います

## 透過 PNG について

- alpha チャンネルは保持します
- SwinIR へは RGB 部分だけを渡します
- alpha は別経路で Lanczos 拡大します
- 最後に RGB と alpha を再合成して RGBA PNG として保存します
- 透明部分の色に引っ張られて白フチや黒フチが出にくいよう、完全透明部には近傍色をにじませてから RGB 拡大します

## 初期値

- GIMP 前処理: `ON`
- ノイズ除去: `弱`
- アンシャープマスク: `弱`
- 処理モード: `GIMP前処理 + SwinIR`
- SwinIR 倍率: `2x`
- GIMP 後処理: `OFF`
- 人物切り抜き: `OFF`
- プレビュー範囲: `顔付近crop`
- プレビュー crop 比率: `4:5`
- tile size: `400`
- tile overlap: `32`
- 既存出力スキップ: `ON`
- サブフォルダ処理: `OFF`

## 導入手順

### 1. Python を入れる

まず Python をインストールしてください。

推奨:

- Python `3.10` から `3.13`

### 2. GIMP を入れる

GIMP 前処理 / 後処理を使いたい場合は、先に GIMP をインストールしてください。

### 3. `setup.bat` を実行する

初回は **必ず `setup.bat`** を実行してください。

このスクリプトは次を行います。

- `%LOCALAPPDATA%\imageUpconvert\venv` に venv 作成
- `requirements.txt` のインストール
- CPU 版 PyTorch の最小構成インストール
- GUI が使う公式 SwinIR の `2x / 4x` モデルをダウンロード
- 簡易環境確認

`requirements.txt` には、人物切り抜き用の `rembg` も含まれます。

### 4. `run_gui.bat` を実行する

起動は **`run_gui.bat` のダブルクリック** です。

## GPU を使いたい場合

CPU 版の代わりに GPU 版 PyTorch を使いたい場合は、**PyTorch 公式のインストールページ**で自分の CUDA 環境に合うコマンドを確認してください。

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

方針:

- `setup.bat` は GPU を決め打ちしません
- GUI では `torch.cuda.is_available()` の結果を表示します

## 使い方

### 1. GIMP パスを設定する

GUI の `GIMP 実行ファイル` に、`gimp-console.exe` または `gimp.exe` を指定します。

### 2. 処理モードを決める

- まずは `GIMPのみ` か `GIMP前処理 + SwinIR` から確認してください
- 高解像度化が不要なら `GIMPのみ` を使えます
- 高解像度化したい場合だけ `SwinIR` を含むモードを選んでください

### 3. まず 1 枚プレビューで設定を決める

1. `プレビュー対象画像 / 比較用1枚` に 1 枚指定
2. `顔付近crop` または `中央crop` を選ぶ
3. ノイズ除去とアンシャープマスクを調整
4. `プレビュー生成`
5. プレビューは別ウィンドウで開くので、左側の一覧をクリックしながら `元画像 / 中間段階 / 最終出力` を見比べる
6. 必要ならプレビュー画面の `表示中を保存`、`元画像を保存`、`横並び比較を保存` を使って結果を書き出す

### 4. 必要なら 1 枚比較処理を行う

`1枚比較処理` を実行すると、`comparison` フォルダに次が出ます。

- `original`
- `gimp_only`
- `swinir_only`
- `gimp_pre_swinir`
- `gimp_pre_swinir_gimp_post`
- `gimp_only_cutout` (`人物切り抜き ON のとき`)
- `gimp_pre_swinir_cutout` (`人物切り抜き ON のとき`)

### 5. 大量処理前に必ず 5 枚テストを行う

大量処理前に、必ず **`テスト処理（先頭5枚）`** を実行してください。

### 6. 問題なければ一括処理

`一括処理開始` を実行します。処理前に、対象枚数と設定概要の確認ダイアログが出ます。

## 安全設計

- 元画像は絶対に上書きしません
- 出力先が入力先と同じ場合は止めます
- 出力先が入力フォルダの内側でも止めます
- GIMP 処理に失敗した場合は次工程へ進みません
- 失敗ファイルは `failed/` にコピーします
- キャンセル時に中途半端な出力を成功扱いしないよう、最後に一時ファイルからリネームしています
- 日本語パス、空白を含むパスを想定しています

## 出力ファイル名

通常処理:

- `元名_gimp_only.png`
- `元名_swinir_x2.png`
- `元名_gimp_pre_swinir_x2.png`
- `元名_gimp_pre_swinir_x2_gimp_post.png`
- 人物切り抜きを使う場合は `_cutout`

例:

- `sample_gimp_only.png`
- `sample_swinir_x2.png`
- `sample_gimp_pre_swinir_x2.png`
- `sample_gimp_pre_swinir_x4_gimp_post_cutout.png`

## ログ

出力フォルダに次を作成します。

- `processing_log.csv`
- `failed_log.csv`

失敗ファイルは `failed/` にコピーします。

主な記録項目:

- 処理日時
- 処理モード
- 入力ファイル
- 出力ファイル
- 元画像サイズ
- SwinIR 入力サイズ
- 出力画像サイズ
- SwinIR 使用有無
- 指定倍率
- 実際の倍率
- 使用内部モデル
- tile size
- tile overlap
- alpha 有無
- プレビュー範囲
- GIMP パス
- GIMP バージョン
- GIMP 前処理の有無
- ノイズ除去設定
- アンシャープマスク設定
- GIMP 後処理の有無
- 後処理アンシャープ設定
- 人物切り抜きの有無
- GIMP の終了コード
- GIMP の標準出力 / 標準エラー
- 成功 / 失敗
- エラー内容
- 処理時間

## 補足

- GIMP 未導入でも、GIMP 前後処理を OFF にすれば SwinIR 単体処理は可能です
- 最初は **2x + ノイズ除去 弱 + アンシャープ 弱** から確認してください
- 4x は常に正解ではありません
- 強いノイズ除去や強いシャープは本人性を損ねる可能性があります
