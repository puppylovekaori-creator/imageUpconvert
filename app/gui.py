from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .env_check import format_runtime_status, get_runtime_status
from .swinir_runner import (
    BatchOptions,
    InterruptController,
    detect_model_scale_from_filename,
    infer_model_descriptor,
    run_batch,
)


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = APP_ROOT / "models" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
DEFAULT_OUTPUT_DIR = APP_ROOT / "output"
SUPPORTED_SCALE_ORDER = [2, 4]
TASK_PRIORITY = {
    "classical_sr": 0,
    "lightweight_sr": 1,
    "real_sr": 2,
}


class BatchWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(dict)
    message_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(dict)
    error_signal = QtCore.Signal(str)

    def __init__(self, options: BatchOptions) -> None:
        super().__init__()
        self._options = options
        self._stop_requested = False
        self._cancel_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_stop_requested(self) -> bool:
        return self._stop_requested

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        controller = InterruptController(
            is_stop_requested=self._is_stop_requested,
            is_cancel_requested=self._is_cancel_requested,
        )
        try:
            summary = run_batch(
                self._options,
                progress_callback=self.progress_signal.emit,
                message_callback=self.message_signal.emit,
                controller=controller,
            )
            self.finished_signal.emit(asdict(summary))
        except Exception as exc:
            self.error_signal.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SwinIR 画像高解像度化 GUI")
        self.resize(980, 720)
        self._worker: BatchWorker | None = None
        self._available_models_by_scale: dict[int, list[Path]] = {}
        self._preferred_model_by_scale: dict[int, Path] = {}

        self._input_edit = QtWidgets.QLineEdit()
        self._output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self._model_edit = QtWidgets.QLineEdit(str(DEFAULT_MODEL_PATH))
        self._model_edit.setReadOnly(True)
        self._scale_combo = QtWidgets.QComboBox()

        self._collision_combo = QtWidgets.QComboBox()
        self._collision_combo.addItem("スキップ", "skip")
        self._collision_combo.addItem("連番で保存", "serial")

        self._skip_checkbox = QtWidgets.QCheckBox("既に出力済みのファイルはスキップする")
        self._skip_checkbox.setChecked(True)
        self._recursive_checkbox = QtWidgets.QCheckBox("サブフォルダも処理する")
        self._recursive_checkbox.setChecked(False)

        self._tile_size_spin = QtWidgets.QSpinBox()
        self._tile_size_spin.setRange(0, 8192)
        self._tile_size_spin.setValue(400)
        self._tile_size_spin.setToolTip("0 を指定するとタイル分割を無効化します。まずは 400 を推奨します。")

        self._tile_overlap_spin = QtWidgets.QSpinBox()
        self._tile_overlap_spin.setRange(0, 1024)
        self._tile_overlap_spin.setValue(32)

        self._device_label = QtWidgets.QLabel()
        self._model_info_label = QtWidgets.QLabel("モデル未選択です。")
        self._current_file_label = QtWidgets.QLabel("現在のファイル: -")
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(1500)

        self._start_button = QtWidgets.QPushButton("一括処理開始")
        self._test_button = QtWidgets.QPushButton("テスト処理（先頭5枚）")
        self._stop_button = QtWidgets.QPushButton("停止")
        self._cancel_button = QtWidgets.QPushButton("キャンセル")
        self._stop_button.setEnabled(False)
        self._cancel_button.setEnabled(False)

        self._build_ui()
        self._connect_signals()
        self._refresh_runtime_status()
        self._rebuild_available_models(preferred_path=DEFAULT_MODEL_PATH, select_scale=2)

    def _build_ui(self) -> None:
        intro_label = QtWidgets.QLabel(
            "このツールは低侵襲な高解像度化専用です。"
            " 顔補正、生成補完、Stable Diffusion、GFPGAN、CodeFormer は使いません。"
        )
        intro_label.setWordWrap(True)

        tips_label = QtWidgets.QLabel(
            "最初は classical_sr の 2x、PNG 出力、tile size 400、tile overlap 32、"
            "出力済みスキップ ON、サブフォルダ OFF を推奨します。"
        )
        tips_label.setWordWrap(True)

        input_button = QtWidgets.QPushButton("参照...")
        output_button = QtWidgets.QPushButton("参照...")
        model_button = QtWidgets.QPushButton("参照...")

        input_button.clicked.connect(lambda: self._pick_directory(self._input_edit))
        output_button.clicked.connect(lambda: self._pick_directory(self._output_edit))
        model_button.clicked.connect(self._pick_model_file)

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("入力フォルダ"), 0, 0)
        form.addWidget(self._input_edit, 0, 1)
        form.addWidget(input_button, 0, 2)
        form.addWidget(QtWidgets.QLabel("出力フォルダ"), 1, 0)
        form.addWidget(self._output_edit, 1, 1)
        form.addWidget(output_button, 1, 2)
        form.addWidget(QtWidgets.QLabel("モデルファイル"), 2, 0)
        form.addWidget(self._model_edit, 2, 1)
        form.addWidget(model_button, 2, 2)
        form.addWidget(QtWidgets.QLabel("倍率"), 3, 0)
        form.addWidget(self._scale_combo, 3, 1)
        form.addWidget(QtWidgets.QLabel("同名出力時"), 4, 0)
        form.addWidget(self._collision_combo, 4, 1)
        form.addWidget(QtWidgets.QLabel("tile size"), 5, 0)
        form.addWidget(self._tile_size_spin, 5, 1)
        form.addWidget(QtWidgets.QLabel("tile overlap"), 6, 0)
        form.addWidget(self._tile_overlap_spin, 6, 1)
        form.addWidget(QtWidgets.QLabel("CPU/GPU 状態"), 7, 0)
        form.addWidget(self._device_label, 7, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("判定モデル"), 8, 0)
        form.addWidget(self._model_info_label, 8, 1, 1, 2)

        checkbox_row = QtWidgets.QHBoxLayout()
        checkbox_row.addWidget(self._skip_checkbox)
        checkbox_row.addWidget(self._recursive_checkbox)
        checkbox_row.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._test_button)
        button_row.addWidget(self._stop_button)
        button_row.addWidget(self._cancel_button)
        button_row.addStretch(1)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(intro_label)
        layout.addWidget(tips_label)
        layout.addLayout(form)
        layout.addLayout(checkbox_row)
        layout.addLayout(button_row)
        layout.addWidget(self._current_file_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._log_view, stretch=1)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self._start_button.clicked.connect(lambda: self._start_processing(test_mode=False))
        self._test_button.clicked.connect(lambda: self._start_processing(test_mode=True))
        self._stop_button.clicked.connect(self._request_stop)
        self._cancel_button.clicked.connect(self._request_cancel)
        self._scale_combo.currentIndexChanged.connect(self._handle_scale_changed)

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)

    def _pick_directory(self, target_edit: QtWidgets.QLineEdit) -> None:
        start_dir = target_edit.text().strip() or str(APP_ROOT)
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "フォルダを選択", start_dir)
        if selected:
            target_edit.setText(selected)

    def _pick_model_file(self) -> None:
        start_path = self._model_edit.text().strip() or str(DEFAULT_MODEL_PATH.parent)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "SwinIR モデルを選択",
            start_path,
            "PyTorch モデル (*.pth *.pt);;すべてのファイル (*.*)",
        )
        if file_path:
            selected_path = Path(file_path)
            scale = detect_model_scale_from_filename(selected_path)
            if scale not in SUPPORTED_SCALE_ORDER:
                QtWidgets.QMessageBox.warning(
                    self,
                    "モデルファイル",
                    "公式 SwinIR の x2 または x4 モデルファイルを選択してください。",
                )
                return
            try:
                infer_model_descriptor(selected_path, scale)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "モデルファイル", str(exc))
                return

            self._rebuild_available_models(preferred_path=selected_path, select_scale=scale)
            self._append_log(f"モデルフォルダを再読込しました: {selected_path.parent}")

    def _refresh_runtime_status(self) -> None:
        status = get_runtime_status()
        self._device_label.setText(format_runtime_status(status))

    def _candidate_model_files(self, model_dir: Path) -> list[Path]:
        candidates: list[Path] = []
        for pattern in ("*.pth", "*.pt"):
            candidates.extend(model_dir.glob(pattern))
        return sorted({path.resolve() for path in candidates}, key=lambda path: path.name.lower())

    def _model_sort_key(self, model_path: Path) -> tuple[int, int, str]:
        scale = detect_model_scale_from_filename(model_path)
        if scale not in SUPPORTED_SCALE_ORDER:
            return (99, 99, model_path.name.lower())

        descriptor = infer_model_descriptor(model_path, scale)
        task_rank = TASK_PRIORITY.get(descriptor.task, 99)
        large_rank = 1 if descriptor.large_model else 0
        return (task_rank, large_rank, model_path.name.lower())

    def _scan_available_models(self, model_dir: Path) -> dict[int, list[Path]]:
        available: dict[int, list[Path]] = {scale: [] for scale in SUPPORTED_SCALE_ORDER}
        for model_path in self._candidate_model_files(model_dir):
            scale = detect_model_scale_from_filename(model_path)
            if scale not in SUPPORTED_SCALE_ORDER:
                continue
            try:
                infer_model_descriptor(model_path, scale)
            except Exception:
                continue
            available[scale].append(model_path)

        return {scale: sorted(paths, key=self._model_sort_key) for scale, paths in available.items() if paths}

    def _choose_preferred_model(
        self,
        *,
        scale: int,
        candidates: list[Path],
        preferred_path: Path | None,
    ) -> Path:
        if preferred_path is not None and preferred_path in candidates:
            return preferred_path

        current_preferred = self._preferred_model_by_scale.get(scale)
        if current_preferred in candidates:
            return current_preferred

        return candidates[0]

    def _set_model_path_text(self, model_path: Path) -> None:
        blocker = QtCore.QSignalBlocker(self._model_edit)
        self._model_edit.setText(str(model_path))
        del blocker

    def _set_scale_choices(self, available_scales: list[int], selected_scale: int) -> None:
        blocker = QtCore.QSignalBlocker(self._scale_combo)
        self._scale_combo.clear()
        for scale in available_scales:
            self._scale_combo.addItem(f"{scale}x", scale)
        index = self._scale_combo.findData(selected_scale)
        if index >= 0:
            self._scale_combo.setCurrentIndex(index)
        del blocker
        self._scale_combo.setEnabled(bool(available_scales))

    def _rebuild_available_models(self, *, preferred_path: Path | None, select_scale: int | None = None) -> None:
        model_dir = preferred_path.parent if preferred_path is not None else DEFAULT_MODEL_PATH.parent
        available_models = self._scan_available_models(model_dir)

        if not available_models:
            self._available_models_by_scale = {}
            self._preferred_model_by_scale = {}
            self._set_scale_choices([], selected_scale=2)
            self._set_model_path_text(DEFAULT_MODEL_PATH)
            self._model_info_label.setText(
                f"モデルが見つかりません。{model_dir} に公式 SwinIR の x2/x4 モデルを置いてください。"
            )
            return

        self._available_models_by_scale = available_models
        self._preferred_model_by_scale = {
            scale: self._choose_preferred_model(
                scale=scale,
                candidates=candidates,
                preferred_path=preferred_path,
            )
            for scale, candidates in available_models.items()
        }

        available_scales = [scale for scale in SUPPORTED_SCALE_ORDER if scale in available_models]
        target_scale = select_scale if select_scale in available_models else available_scales[0]
        self._set_scale_choices(available_scales, selected_scale=target_scale)
        self._update_model_path_for_selected_scale()
        self._refresh_model_info()

    def _update_model_path_for_selected_scale(self) -> None:
        scale = self._selected_scale()
        model_path = self._preferred_model_by_scale.get(scale)
        if model_path is None:
            self._set_model_path_text(DEFAULT_MODEL_PATH)
            return
        self._set_model_path_text(model_path)

    def _handle_scale_changed(self) -> None:
        if not self._available_models_by_scale:
            self._refresh_model_info()
            return
        self._update_model_path_for_selected_scale()
        self._refresh_model_info()

    def _refresh_model_info(self) -> None:
        model_text = self._model_edit.text().strip()
        if not model_text:
            self._model_info_label.setText("公式 SwinIR の .pth モデルを選択してください。")
            return
        if not self._available_models_by_scale:
            self._model_info_label.setText("利用可能なモデルがありません。")
            return

        try:
            descriptor = infer_model_descriptor(Path(model_text), self._selected_scale())
        except Exception as exc:
            self._model_info_label.setText(f"モデル確認: {exc}")
            return

        label_map = {
            "classical_sr": "classical_sr（素直な拡大向け）",
            "lightweight_sr": "lightweight_sr（軽量）",
            "real_sr": "real_sr（比較用）",
        }
        extra = "large" if descriptor.large_model else "standard"
        extra_map = {
            "large": "Large モデル",
            "standard": "標準モデル",
        }
        self._model_info_label.setText(
            f"{label_map.get(descriptor.label, descriptor.label)} / x{descriptor.scale} / "
            f"{extra_map.get(extra, extra)} / window {descriptor.window_size}"
        )

    def _selected_scale(self) -> int:
        current_data = self._scale_combo.currentData()
        if current_data is None:
            return 2
        return int(current_data)

    def _build_options(self, *, test_mode: bool) -> BatchOptions:
        input_dir = Path(self._input_edit.text().strip())
        output_dir = Path(self._output_edit.text().strip())
        model_path = Path(self._model_edit.text().strip())
        return BatchOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=model_path,
            scale=self._selected_scale(),
            tile_size=int(self._tile_size_spin.value()),
            tile_overlap=int(self._tile_overlap_spin.value()),
            skip_existing=self._skip_checkbox.isChecked(),
            recursive=self._recursive_checkbox.isChecked(),
            collision_policy=str(self._collision_combo.currentData()),
            test_mode=test_mode,
            test_limit=5,
        )

    def _start_processing(self, *, test_mode: bool) -> None:
        if self._worker and self._worker.isRunning():
            return

        input_text = self._input_edit.text().strip()
        output_text = self._output_edit.text().strip()
        model_text = self._model_edit.text().strip()
        if not input_text or not output_text or not model_text:
            QtWidgets.QMessageBox.warning(
                self,
                "設定不足",
                "入力フォルダ、出力フォルダ、モデルファイルはすべて必須です。",
            )
            return
        if not self._available_models_by_scale:
            QtWidgets.QMessageBox.warning(
                self,
                "モデルファイル",
                "利用可能なモデルがありません。models/swinir などに公式 SwinIR モデルを置いてください。",
            )
            return

        options = self._build_options(test_mode=test_mode)
        if not options.input_dir.exists():
            QtWidgets.QMessageBox.warning(self, "入力フォルダ", "入力フォルダが存在しません。")
            return
        if options.input_dir.resolve() == options.output_dir.resolve():
            QtWidgets.QMessageBox.warning(
                self,
                "出力フォルダ",
                "出力フォルダは入力フォルダと同じ場所にできません。",
            )
            return

        self._append_log("テスト処理を開始します..." if test_mode else "一括処理を開始します...")
        self._progress_bar.setValue(0)
        self._current_file_label.setText("現在のファイル: -")
        self._set_running_state(True)

        self._worker = BatchWorker(options)
        self._worker.progress_signal.connect(self._handle_progress)
        self._worker.message_signal.connect(self._append_log)
        self._worker.finished_signal.connect(self._handle_finished)
        self._worker.error_signal.connect(self._handle_error)
        self._worker.start()

    def _handle_progress(self, payload: dict) -> None:
        total = max(int(payload.get("total", 0)), 1)
        completed = int(payload.get("completed", 0))
        path_text = payload.get("path", "")
        if path_text:
            self._current_file_label.setText(f"現在のファイル: {path_text}")

        percentage = int((completed / total) * 100)
        self._progress_bar.setValue(min(max(percentage, 0), 100))

    def _request_stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._append_log("停止要求を受け付けました。現在のファイル完了後に停止します。")

    def _request_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._append_log("キャンセル要求を受け付けました。できるだけ早く中断します。")

    def _handle_finished(self, summary: dict) -> None:
        self._set_running_state(False)
        total = max(int(summary.get("total_files", 0)), 1)
        completed = int(summary.get("processed", 0)) + int(summary.get("skipped", 0)) + int(summary.get("failed", 0))
        self._progress_bar.setValue(int((completed / total) * 100))
        self._append_log(
            "処理完了。"
            f" 成功={summary['processed']}, スキップ={summary['skipped']}, 失敗={summary['failed']}, "
            f"停止={summary['stopped']}, キャンセル={summary['cancelled']}"
        )
        message = (
            f"出力先: {summary['output_dir']}\n"
            f"処理成功: {summary['processed']}\n"
            f"スキップ: {summary['skipped']}\n"
            f"失敗: {summary['failed']}\n"
            f"停止: {summary['stopped']}\n"
            f"キャンセル: {summary['cancelled']}"
        )
        QtWidgets.QMessageBox.information(self, "処理完了", message)
        self._worker = None

    def _handle_error(self, message: str) -> None:
        self._set_running_state(False)
        self._append_log(f"エラー: {message}")
        QtWidgets.QMessageBox.critical(self, "処理エラー", message)
        self._worker = None

    def _set_running_state(self, is_running: bool) -> None:
        self._start_button.setEnabled(not is_running)
        self._test_button.setEnabled(not is_running)
        self._stop_button.setEnabled(is_running)
        self._cancel_button.setEnabled(is_running)
