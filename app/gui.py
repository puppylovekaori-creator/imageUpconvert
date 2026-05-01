from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from threading import Event

from PySide6 import QtCore, QtWidgets

from .app_config import (
    APP_ROOT,
    NOISE_PRESETS,
    PRESET_LABELS,
    PROCESSING_MODE_CHOICES,
    PROCESSING_MODE_PREPROCESS_AND_UPSCALE,
    PROCESSING_MODE_PREPROCESS_ONLY,
    PROCESSING_MODE_UPSCALE_ONLY,
    TASK_KIND_BATCH,
    TASK_KIND_COMPARISON,
    TASK_KIND_TEST_20,
    TASK_KIND_TEST_5,
    UPSCALE_MODELS,
    UPSCALE_SCALES,
    AppSettings,
    mode_uses_preprocess,
)
from .gimp_runner import detect_gimp_info, find_first_gimp_path, is_gimp_available
from .image_utils import validate_input_output_paths
from .pipeline import TaskSummary, run_task, settings_to_pipeline_options
from .settings_store import load_settings, save_settings
from .upscale_runner import prepare_upscale_plan


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TaskWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(dict)
    message_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(dict)
    error_signal = QtCore.Signal(str)

    def __init__(self, task_kind: str, settings: AppSettings) -> None:
        super().__init__()
        self._task_kind = task_kind
        self._settings = settings
        self._stop_event = Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            summary: TaskSummary = run_task(
                self._task_kind,
                settings_to_pipeline_options(self._settings),
                progress_callback=self.progress_signal.emit,
                message_callback=self.message_signal.emit,
                stop_check=self._stop_event.is_set,
            )
            self.finished_signal.emit(asdict(summary))
        except Exception as exc:
            self.error_signal.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GIMP AI Upscale 一括処理GUIツール")
        self.resize(1040, 860)

        self._worker: TaskWorker | None = None
        self._syncing_mode_ui = False

        self._build_ui()
        self._load_initial_settings()
        self._refresh_mode_ui()
        self._refresh_gimp_status()
        self._append_log("GUI を起動しました。まず 1枚比較処理 で確認してください。")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        note_label = QtWidgets.QLabel(
            "推奨手順: 1枚比較処理 → 先頭5枚テスト → 先頭20枚テスト → 問題なければ一括処理。"
            " まずは AI Upscaleのみ で回し、必要な場合だけ GIMP前処理を追加してください。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("background: #f7f3d6; border: 1px solid #d9cf9d; padding: 8px;")
        root_layout.addWidget(note_label)

        paths_group = QtWidgets.QGroupBox("基本設定")
        paths_layout = QtWidgets.QGridLayout(paths_group)
        root_layout.addWidget(paths_group)

        self.input_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        self.gimp_edit = QtWidgets.QLineEdit()
        input_button = QtWidgets.QPushButton("参照...")
        output_button = QtWidgets.QPushButton("参照...")
        gimp_button = QtWidgets.QPushButton("参照...")
        input_button.clicked.connect(lambda: self._pick_directory(self.input_edit))
        output_button.clicked.connect(lambda: self._pick_directory(self.output_edit))
        gimp_button.clicked.connect(self._pick_gimp_path)

        paths_layout.addWidget(QtWidgets.QLabel("入力フォルダ"), 0, 0)
        paths_layout.addWidget(self.input_edit, 0, 1)
        paths_layout.addWidget(input_button, 0, 2)
        paths_layout.addWidget(QtWidgets.QLabel("出力フォルダ"), 1, 0)
        paths_layout.addWidget(self.output_edit, 1, 1)
        paths_layout.addWidget(output_button, 1, 2)
        paths_layout.addWidget(QtWidgets.QLabel("GIMP 実行ファイル"), 2, 0)
        paths_layout.addWidget(self.gimp_edit, 2, 1)
        paths_layout.addWidget(gimp_button, 2, 2)

        self.gimp_status_label = QtWidgets.QLabel("")
        self.backend_status_label = QtWidgets.QLabel("")
        self.gimp_status_label.setWordWrap(True)
        self.backend_status_label.setWordWrap(True)
        self.backend_status_label.setStyleSheet("color: #555555;")
        paths_layout.addWidget(self.gimp_status_label, 3, 0, 1, 3)
        paths_layout.addWidget(self.backend_status_label, 4, 0, 1, 3)

        options_group = QtWidgets.QGroupBox("処理設定")
        options_layout = QtWidgets.QGridLayout(options_group)
        root_layout.addWidget(options_group)

        self.mode_combo = QtWidgets.QComboBox()
        for label, value in PROCESSING_MODE_CHOICES:
            self.mode_combo.addItem(label, value)
        self.model_combo = QtWidgets.QComboBox()
        for model_name in UPSCALE_MODELS:
            self.model_combo.addItem(model_name, model_name)
        self.scale_combo = QtWidgets.QComboBox()
        for scale in UPSCALE_SCALES:
            self.scale_combo.addItem(f"{scale}x", scale)

        self.preprocess_checkbox = QtWidgets.QCheckBox("GIMP前処理を使う")
        self.noise_combo = QtWidgets.QComboBox()
        self.unsharp_combo = QtWidgets.QComboBox()
        for preset in NOISE_PRESETS:
            self.noise_combo.addItem(PRESET_LABELS[preset], preset)
            self.unsharp_combo.addItem(PRESET_LABELS[preset], preset)

        self.include_subfolders_checkbox = QtWidgets.QCheckBox("サブフォルダを処理する")
        self.skip_existing_checkbox = QtWidgets.QCheckBox("処理済みをスキップする")

        options_layout.addWidget(QtWidgets.QLabel("処理モード"), 0, 0)
        options_layout.addWidget(self.mode_combo, 0, 1)
        options_layout.addWidget(self.preprocess_checkbox, 0, 2)
        options_layout.addWidget(QtWidgets.QLabel("アップスケール倍率"), 1, 0)
        options_layout.addWidget(self.scale_combo, 1, 1)
        options_layout.addWidget(QtWidgets.QLabel("アップスケールモデル"), 1, 2)
        options_layout.addWidget(self.model_combo, 1, 3)
        options_layout.addWidget(QtWidgets.QLabel("ノイズ除去"), 2, 0)
        options_layout.addWidget(self.noise_combo, 2, 1)
        options_layout.addWidget(QtWidgets.QLabel("アンシャープ"), 2, 2)
        options_layout.addWidget(self.unsharp_combo, 2, 3)
        options_layout.addWidget(self.include_subfolders_checkbox, 3, 0, 1, 2)
        options_layout.addWidget(self.skip_existing_checkbox, 3, 2, 1, 2)

        buttons_group = QtWidgets.QGroupBox("実行")
        buttons_layout = QtWidgets.QHBoxLayout(buttons_group)
        root_layout.addWidget(buttons_group)

        self.comparison_button = QtWidgets.QPushButton("1枚比較処理")
        self.test5_button = QtWidgets.QPushButton("先頭5枚テスト")
        self.test20_button = QtWidgets.QPushButton("先頭20枚テスト")
        self.batch_button = QtWidgets.QPushButton("一括処理")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setEnabled(False)

        self.comparison_button.clicked.connect(lambda: self._start_task(TASK_KIND_COMPARISON))
        self.test5_button.clicked.connect(lambda: self._start_task(TASK_KIND_TEST_5))
        self.test20_button.clicked.connect(lambda: self._start_task(TASK_KIND_TEST_20))
        self.batch_button.clicked.connect(lambda: self._start_task(TASK_KIND_BATCH))
        self.stop_button.clicked.connect(self._request_stop)

        buttons_layout.addWidget(self.comparison_button)
        buttons_layout.addWidget(self.test5_button)
        buttons_layout.addWidget(self.test20_button)
        buttons_layout.addWidget(self.batch_button)
        buttons_layout.addWidget(self.stop_button)

        progress_group = QtWidgets.QGroupBox("進捗")
        progress_layout = QtWidgets.QGridLayout(progress_group)
        root_layout.addWidget(progress_group)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.current_file_label = QtWidgets.QLabel("現在処理中: -")
        self.counts_label = QtWidgets.QLabel("総件数 0 / 完了 0 / 失敗 0 / スキップ 0")
        self.rate_label = QtWidgets.QLabel("処理速度: -")
        self.eta_label = QtWidgets.QLabel("残り時間: -")
        self.output_hint_label = QtWidgets.QLabel(f"comparison / failed / test_* は出力フォルダ配下に作成します。既定ルート: {APP_ROOT / 'output'}")
        self.output_hint_label.setWordWrap(True)
        self.output_hint_label.setStyleSheet("color: #555555;")

        progress_layout.addWidget(self.progress_bar, 0, 0, 1, 4)
        progress_layout.addWidget(self.current_file_label, 1, 0, 1, 4)
        progress_layout.addWidget(self.counts_label, 2, 0, 1, 2)
        progress_layout.addWidget(self.rate_label, 2, 2)
        progress_layout.addWidget(self.eta_label, 2, 3)
        progress_layout.addWidget(self.output_hint_label, 3, 0, 1, 4)

        log_group = QtWidgets.QGroupBox("ログ")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        root_layout.addWidget(log_group, stretch=1)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)

        self.mode_combo.currentIndexChanged.connect(self._refresh_mode_ui)
        self.preprocess_checkbox.toggled.connect(self._handle_preprocess_toggle)
        self.gimp_edit.editingFinished.connect(self._refresh_gimp_status)
        self.model_combo.currentIndexChanged.connect(self._refresh_gimp_status)
        self.scale_combo.currentIndexChanged.connect(self._refresh_gimp_status)

    def _load_initial_settings(self) -> None:
        settings = load_settings()
        if not settings.gimp_path:
            detected = find_first_gimp_path()
            if detected is not None:
                settings.gimp_path = str(detected)

        self.input_edit.setText(settings.last_input_dir)
        self.output_edit.setText(settings.last_output_dir)
        self.gimp_edit.setText(settings.gimp_path)
        self._set_combo_value(self.mode_combo, settings.processing_mode)
        self._set_combo_value(self.model_combo, settings.model_name)
        self._set_combo_value(self.scale_combo, settings.scale)
        self._set_combo_value(self.noise_combo, settings.noise_reduction)
        self._set_combo_value(self.unsharp_combo, settings.unsharp_strength)
        self.include_subfolders_checkbox.setChecked(settings.include_subfolders)
        self.skip_existing_checkbox.setChecked(settings.skip_existing)

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: object) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _collect_settings(self) -> AppSettings:
        return AppSettings(
            last_input_dir=self.input_edit.text().strip(),
            last_output_dir=self.output_edit.text().strip(),
            gimp_path=self.gimp_edit.text().strip(),
            processing_mode=str(self.mode_combo.currentData()),
            model_name=str(self.model_combo.currentData()),
            scale=int(self.scale_combo.currentData()),
            noise_reduction=str(self.noise_combo.currentData()),
            unsharp_strength=str(self.unsharp_combo.currentData()),
            include_subfolders=self.include_subfolders_checkbox.isChecked(),
            skip_existing=self.skip_existing_checkbox.isChecked(),
        )

    def _save_current_settings(self) -> None:
        save_settings(self._collect_settings())

    def _pick_directory(self, target_edit: QtWidgets.QLineEdit) -> None:
        initial_dir = target_edit.text().strip() or str(APP_ROOT)
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "フォルダを選択", initial_dir)
        if selected:
            target_edit.setText(selected)
            self._save_current_settings()

    def _pick_gimp_path(self) -> None:
        initial_path = self.gimp_edit.text().strip() or str(find_first_gimp_path() or APP_ROOT)
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "GIMP 実行ファイルを選択",
            initial_path,
            "GIMP 実行ファイル (gimp*.exe);;実行ファイル (*.exe);;すべてのファイル (*.*)",
        )
        if selected:
            self.gimp_edit.setText(selected)
            self._refresh_gimp_status()
            self._save_current_settings()

    def _refresh_mode_ui(self) -> None:
        self._syncing_mode_ui = True
        mode = str(self.mode_combo.currentData())
        uses_preprocess = mode_uses_preprocess(mode)
        self.preprocess_checkbox.setChecked(uses_preprocess)
        self.preprocess_checkbox.setEnabled(mode != PROCESSING_MODE_PREPROCESS_ONLY)
        self.noise_combo.setEnabled(uses_preprocess)
        self.unsharp_combo.setEnabled(uses_preprocess)
        self._syncing_mode_ui = False

    def _handle_preprocess_toggle(self, checked: bool) -> None:
        if self._syncing_mode_ui:
            return
        mode = str(self.mode_combo.currentData())
        if mode == PROCESSING_MODE_PREPROCESS_ONLY:
            self.preprocess_checkbox.setChecked(True)
            return
        if checked and mode == PROCESSING_MODE_UPSCALE_ONLY:
            self._set_combo_value(self.mode_combo, PROCESSING_MODE_PREPROCESS_AND_UPSCALE)
        elif not checked and mode == PROCESSING_MODE_PREPROCESS_AND_UPSCALE:
            self._set_combo_value(self.mode_combo, PROCESSING_MODE_UPSCALE_ONLY)
        self._refresh_mode_ui()

    def _refresh_gimp_status(self) -> None:
        gimp_text = self.gimp_edit.text().strip()
        if not gimp_text:
            self.gimp_status_label.setText("GIMP 未設定")
            self.gimp_status_label.setStyleSheet("color: #b00020;")
            self.backend_status_label.setText("AI backend 判定前です。GIMP パスを指定してください。")
            return

        gimp_path = Path(gimp_text)
        ok, status_text = is_gimp_available(gimp_path)
        if not ok:
            self.gimp_status_label.setText(status_text)
            self.gimp_status_label.setStyleSheet("color: #b00020;")
            self.backend_status_label.setText("AI backend 判定前です。GIMP パスが有効になってから実行方式を表示します。")
            return

        self.gimp_status_label.setText(status_text)
        self.gimp_status_label.setStyleSheet("color: #1f5f2c;")
        try:
            info = detect_gimp_info(gimp_path)
            plan = prepare_upscale_plan(
                info,
                str(self.model_combo.currentData()),
                int(self.scale_combo.currentData()),
            )
            messages: list[str] = []
            if plan.direct_plugin_available:
                messages.append("gimp_plugin_direct 可")
            if plan.external_backend is not None:
                messages.append(f"external_realesrgan 可 ({plan.external_backend.source_label})")
            self.backend_status_label.setText("使用可能方式: " + " / ".join(messages))
        except Exception as exc:
            self.backend_status_label.setText(f"AI backend 判定: {exc}")

    def _validate_settings(self, settings: AppSettings) -> None:
        if not settings.last_input_dir:
            raise ValueError("入力フォルダを指定してください。")
        if not settings.last_output_dir:
            raise ValueError("出力フォルダを指定してください。")
        if not settings.gimp_path:
            raise ValueError("GIMP 実行ファイルパスを指定してください。")

        input_dir = Path(settings.last_input_dir)
        output_dir = Path(settings.last_output_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"入力フォルダが見つかりません: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        validate_input_output_paths(input_dir, output_dir)

    def _set_running_state(self, running: bool) -> None:
        self.comparison_button.setEnabled(not running)
        self.test5_button.setEnabled(not running)
        self.test20_button.setEnabled(not running)
        self.batch_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _start_task(self, task_kind: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        settings = self._collect_settings()
        try:
            self._validate_settings(settings)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "設定エラー", str(exc))
            return

        self._save_current_settings()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.current_file_label.setText("現在処理中: 準備中...")
        self.counts_label.setText("総件数 0 / 完了 0 / 失敗 0 / スキップ 0")
        self.rate_label.setText("処理速度: -")
        self.eta_label.setText("残り時間: -")
        self._append_log(f"{task_kind} を開始します。")

        self._worker = TaskWorker(task_kind, settings)
        self._worker.progress_signal.connect(self._handle_progress)
        self._worker.message_signal.connect(self._append_log)
        self._worker.finished_signal.connect(self._handle_finished)
        self._worker.error_signal.connect(self._handle_error)
        self._set_running_state(True)
        self._worker.start()

    def _request_stop(self) -> None:
        if self._worker is None:
            return
        self._append_log("停止要求を送信しました。現在のファイル単位で止まります。")
        self._worker.request_stop()

    def _handle_progress(self, payload: dict) -> None:
        total = int(payload.get("total", 0))
        completed = int(payload.get("completed", 0))
        failed = int(payload.get("failed", 0))
        skipped = int(payload.get("skipped", 0))
        current_file = str(payload.get("current_file", "") or "-")
        done = completed + failed + skipped
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(min(done, max(total, 1)))
        self.current_file_label.setText(f"現在処理中: {current_file}")
        self.counts_label.setText(
            f"総件数 {total} / 完了 {completed} / 失敗 {failed} / スキップ {skipped}"
        )
        rate = float(payload.get("rate_per_minute", 0.0))
        eta_seconds = int(payload.get("eta_seconds", 0))
        self.rate_label.setText(f"処理速度: {rate:.2f} 件/分")
        self.eta_label.setText(f"残り時間: {_format_duration(eta_seconds)}")

    def _handle_finished(self, payload: dict) -> None:
        self._set_running_state(False)
        self._worker = None
        summary = TaskSummary(**payload)
        self._append_log(
            f"完了: completed={summary.completed}, failed={summary.failed}, skipped={summary.skipped}, stopped={summary.stopped}"
        )
        message = (
            f"出力先: {summary.output_dir}\n"
            f"完了: {summary.completed}\n"
            f"失敗: {summary.failed}\n"
            f"スキップ: {summary.skipped}\n"
            f"停止: {'あり' if summary.stopped else 'なし'}\n"
            f"経過時間: {_format_duration(summary.elapsed_seconds)}"
        )
        QtWidgets.QMessageBox.information(self, "処理結果", message)

    def _handle_error(self, message: str) -> None:
        self._set_running_state(False)
        self._append_log(f"エラー: {message}")
        self._worker = None
        QtWidgets.QMessageBox.critical(self, "処理エラー", message)

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_current_settings()
        super().closeEvent(event)
