# GIMP AI Upscale 一括処理GUIツール

Windows 用の `Python + PySide6` GUI です。  
大量の人物画像に対して、GIMP 前処理と GIMP AI Upscale 系 backend を安全に一括適用し、元画像を壊さずに参照候補を増やすことを目的にしています。

## 重要

- このツールは **GIMP 依存** です
- **GIMP パスは GUI で指定** してください
- **元画像は絶対に上書きしません**
- **まず 1枚比較処理 を行ってください**
- 次に **先頭5枚テスト** を行ってください
- 問題なければ **先頭20枚テスト**、その後に **一括処理** を行ってください
- 4000 枚規模ではかなり時間がかかります
- まずは **AI Upscaleのみ** で回し、必要な場合だけ前処理を追加する運用を推奨します
- **切り抜きは標準ルートに含めません**
- **本人性維持を最優先** とし、不自然な画像は採用しないでください

## 実装方針

- GUI: `PySide6`
- 前処理: GIMP を `subprocess` でバッチ呼び出し
- AI Upscale:
  - まず `gimp_plugin_direct` を試行
  - 安定呼び出しが難しい場合は `external_realesrgan` にフォールバック
  - 実際に使った方式は `processing_log.csv` に記録
- 設定保存: `config.json`
- ログ: `processing_log.csv`, `failed_log.csv`
- 起動: `run_gui.bat`
- 初回セットアップ: `setup.bat`

現在の実装では、GIMP 2 の AI Upscale プラグインが入っている場合のみ `gimp_plugin_direct` を優先し、それ以外は `setup.bat` で取得する `external_realesrgan` backend を使います。  
GIMP 3 ではモデル選択付きの非対話プラグイン直実行が不安定なため、安定側として `external_realesrgan` を採用しています。

## 対応処理モード

- `AI Upscaleのみ`
- `GIMP前処理のみ`
- `GIMP前処理 + AI Upscale`

## GUI で指定できる項目

- 入力フォルダ
- 出力フォルダ
- GIMP 実行ファイルパス
- 処理モード
- アップスケール倍率
- アップスケールモデル
- GIMP 前処理 ON/OFF
- ノイズ除去設定
- アンシャープ設定
- サブフォルダ処理 ON/OFF
- 処理済みスキップ ON/OFF

## 対応モデル

- `UltraSharp-4x`
- `RealESRGAN_General_x4_v3`
- `realesrgan-x4plus`
- `realesrgan-x4plus-anime`
- `AnimeSharp-4x`
- `realesr-animevideov3-x4`

初期値:

- モデル: `UltraSharp-4x`
- 倍率: `4x`
- 前処理: `OFF`
- ノイズ除去: `弱`
- アンシャープ: `弱`
- サブフォルダ処理: `OFF`
- 処理済みスキップ: `ON`

## 前処理

前処理では以下を使えます。

- ノイズ除去: `OFF / 弱 / 中 / 強`
- アンシャープ: `OFF / 弱 / 中 / 強`

## 比較処理

`1枚比較処理` は入力フォルダの先頭 1 枚を使い、出力フォルダ配下の `comparison` フォルダに以下を保存します。

- `original`
- `gimp_pre`
- `upscale_only`
- `gimp_pre_upscale`

ファイル名には処理内容を含めます。

## 一括処理

- 対象拡張子: `jpg`, `jpeg`, `png`, `webp`
- サブフォルダ処理 ON のときは再帰処理します
- 出力ファイルは常に **別名 PNG** として保存します
- 例:
  - `sample_upscale_ultrasharp4x.png`
  - `sample_pre_noiseWeak_sharpMedium_upscale_ultrasharp4x.png`

## 安全設計

- 元画像は絶対に上書きしません
- 入力フォルダと出力フォルダが同じ場合は開始しません
- 出力フォルダが入力フォルダ配下でも開始しません
- GIMP パスが未設定または無効なら開始しません
- 処理済みスキップにより再開できます
- GUI は別スレッドで処理します
- 停止ボタンで現在ファイル単位の安全停止を行います
- 中途半端なファイルは成功扱いしません
- 日本語パス、空白を含むパスに対応します

## ログ

出力フォルダ配下に以下を出します。

- `processing_log.csv`
- `failed_log.csv`
- `failed\`

`processing_log.csv` には少なくとも以下を記録します。

- 処理日時
- 入力ファイル
- 出力ファイル
- 処理モード
- 元画像サイズ
- 出力画像サイズ
- GIMP パス
- 使用実行方式
- 使用モデル
- 倍率
- GIMP 前処理有無
- ノイズ除去設定
- アンシャープ設定
- 成功 / 失敗 / スキップ / 停止
- エラー内容
- 処理時間

## セットアップ

### 1. Python

Python `3.10` から `3.13` を使ってください。

### 2. GIMP

GIMP をインストールしてください。GUI で以下のような実行ファイルを指定できます。

- `C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe`
- `C:\Program Files\GIMP 2\bin\gimp-2.10.exe`
- `C:\Program Files\GIMP 3\bin\gimp-console-3.exe`
- `C:\Program Files\GIMP 3\bin\gimp.exe`

### 3. setup.bat

初回は `setup.bat` を実行してください。

`setup.bat` が行うこと:

- `%LOCALAPPDATA%\imageUpconvert\venv` を作成
- `requirements.txt` をインストール
- `gimp_upscale` リリース由来の `external_realesrgan` backend とモデルを `vendor\gimp_upscale\resrgan` に取得
- GUI スモークテスト実行

### 4. run_gui.bat

通常起動は `run_gui.bat` のダブルクリックです。

## 完成条件に対する現在の挙動

- `setup.bat` 実行後、`run_gui.bat` で起動
- GUI から GIMP パス / 入力 / 出力を指定
- `1枚比較処理`
- `先頭5枚テスト`
- `先頭20枚テスト`
- フォルダ一括処理
- CSV ログ出力
- 失敗画像分離
- 途中停止と再開

## 注意

- このツールは「劇的に別人レベルへ変える」用途ではありません
- 目的は「元画像より少し良くする」「使える候補を増やす」です
- 不自然さが出た画像は採用しないでください
