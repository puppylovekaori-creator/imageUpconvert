from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .env_check import format_runtime_status, get_runtime_status
from .gimp_runner import (
    NoiseReductionSettings,
    UnsharpMaskSettings,
    describe_noise_settings,
    describe_unsharp_settings,
    find_first_gimp_path,
    is_gimp_available,
)
from .settings_store import load_settings, save_settings
from .swinir_runner import (
    APP_ROOT,
    BatchOptions,
    ComparisonOptions,
    InterruptController,
    PipelineOptions,
    PreviewOptions,
    get_default_model_path,
    infer_model_descriptor,
    list_available_internal_scales,
    run_batch,
    run_comparison,
    run_preview,
)


DEFAULT_OUTPUT_DIR = APP_ROOT / "output"
SUPPORTED_FILE_FILTER = "画像 (*.jpg *.jpeg *.png *.webp);;すべてのファイル (*.*)"
GIMP_FILTER = (
    "GIMP 実行ファイル (gimp*.exe);;"
    "実行ファイル (*.exe);;"
    "すべてのファイル (*.*)"
)
PREVIEW_RANGES = (
    ("全体", "full"),
    ("中央crop", "center"),
    ("顔付近crop", "face_near"),
)
CROP_RATIO_CHOICES = ("1:1", "4:5", "3:4", "2:3", "16:9", "9:16")
NOISE_CHOICES = (
    ("OFF", "off"),
    ("弱", "weak"),
    ("中", "medium"),
    ("強", "strong"),
    ("詳細設定", "detail"),
)
UNSHARP_CHOICES = NOISE_CHOICES
POST_UNSHARP_CHOICES = (
    ("なし", "off"),
    ("弱", "weak"),
    ("中", "medium"),
    ("強", "strong"),
)


class TaskWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(dict)
    message_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(dict)
    error_signal = QtCore.Signal(str)

    def __init__(self, task_callable) -> None:
        super().__init__()
        self._task_callable = task_callable
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
            summary = self._task_callable(
                progress_callback=self.progress_signal.emit,
                message_callback=self.message_signal.emit,
                controller=controller,
            )
            payload = asdict(summary) if is_dataclass(summary) else summary
            self.finished_signal.emit(payload)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SwinIR + GIMP 低侵襲高解像度化 GUI")
        self.resize(1380, 980)
        self._worker: TaskWorker | None = None
        self._available_scales = list_available_internal_scales()
        self._preview_image_labels: dict[str, QtWidgets.QLabel] = {}
        self._preview_caption_labels: dict[str, QtWidgets.QLabel] = {}

        self._gimp_path_edit = QtWidgets.QLineEdit()
        self._input_edit = QtWidgets.QLineEdit()
        self._output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self._preview_source_edit = QtWidgets.QLineEdit()
        self._scale_combo = QtWidgets.QComboBox()
        self._preview_range_combo = QtWidgets.QComboBox()
        self._preview_ratio_combo = QtWidgets.QComboBox()

        self._skip_checkbox = QtWidgets.QCheckBox("既に出力済みのファイルはスキップする")
        self._skip_checkbox.setChecked(True)
        self._recursive_checkbox = QtWidgets.QCheckBox("サブフォルダも処理する")
        self._recursive_checkbox.setChecked(False)
        self._use_gimp_pre_checkbox = QtWidgets.QCheckBox("GIMP前処理を使う")
        self._use_gimp_pre_checkbox.setChecked(True)
        self._use_gimp_post_checkbox = QtWidgets.QCheckBox("GIMP後処理を使う")
        self._use_gimp_post_checkbox.setChecked(False)
        self._use_cutout_checkbox = QtWidgets.QCheckBox("人物切り抜きを使う")
        self._use_cutout_checkbox.setChecked(False)

        self._collision_combo = QtWidgets.QComboBox()
        self._collision_combo.addItem("スキップ", "skip")
        self._collision_combo.addItem("連番で保存", "serial")

        self._tile_size_spin = QtWidgets.QSpinBox()
        self._tile_size_spin.setRange(0, 8192)
        self._tile_size_spin.setValue(400)
        self._tile_size_spin.setToolTip("0 を指定するとタイル分割を無効化します。まずは 400 を推奨します。")

        self._tile_overlap_spin = QtWidgets.QSpinBox()
        self._tile_overlap_spin.setRange(0, 1024)
        self._tile_overlap_spin.setValue(32)

        self._noise_combo = QtWidgets.QComboBox()
        for label, value in NOISE_CHOICES:
            self._noise_combo.addItem(label, value)
        self._noise_combo.setCurrentIndex(1)

        self._pre_unsharp_combo = QtWidgets.QComboBox()
        for label, value in UNSHARP_CHOICES:
            self._pre_unsharp_combo.addItem(label, value)
        self._pre_unsharp_combo.setCurrentIndex(1)

        self._post_unsharp_combo = QtWidgets.QComboBox()
        for label, value in POST_UNSHARP_CHOICES:
            self._post_unsharp_combo.addItem(label, value)
        self._post_unsharp_combo.setCurrentIndex(0)

        self._noise_radius_spin = QtWidgets.QDoubleSpinBox()
        self._noise_radius_spin.setRange(0.0, 50.0)
        self._noise_radius_spin.setDecimals(2)
        self._noise_radius_spin.setValue(3.0)

        self._noise_black_spin = QtWidgets.QSpinBox()
        self._noise_black_spin.setRange(0, 255)
        self._noise_black_spin.setValue(1)

        self._noise_white_spin = QtWidgets.QSpinBox()
        self._noise_white_spin.setRange(0, 255)
        self._noise_white_spin.setValue(248)

        self._noise_iterations_spin = QtWidgets.QSpinBox()
        self._noise_iterations_spin.setRange(0, 32)
        self._noise_iterations_spin.setValue(1)

        self._pre_unsharp_radius_spin = QtWidgets.QDoubleSpinBox()
        self._pre_unsharp_radius_spin.setRange(0.0, 20.0)
        self._pre_unsharp_radius_spin.setDecimals(3)
        self._pre_unsharp_radius_spin.setValue(1.2)

        self._pre_unsharp_amount_spin = QtWidgets.QDoubleSpinBox()
        self._pre_unsharp_amount_spin.setRange(0.0, 10.0)
        self._pre_unsharp_amount_spin.setDecimals(3)
        self._pre_unsharp_amount_spin.setValue(0.35)

        self._pre_unsharp_threshold_spin = QtWidgets.QDoubleSpinBox()
        self._pre_unsharp_threshold_spin.setRange(0.0, 1.0)
        self._pre_unsharp_threshold_spin.setDecimals(3)
        self._pre_unsharp_threshold_spin.setSingleStep(0.01)
        self._pre_unsharp_threshold_spin.setValue(0.02)

        self._device_label = QtWidgets.QLabel()
        self._gimp_status_label = QtWidgets.QLabel("GIMP 未確認")
        self._model_info_label = QtWidgets.QLabel()
        self._current_file_label = QtWidgets.QLabel("現在のファイル: -")
        self._preview_status_label = QtWidgets.QLabel("プレビュー未生成")
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(24)
        self._progress_bar.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #9a9a9a;"
            " border-radius: 4px;"
            " text-align: center;"
            " background: #f6f6f6;"
            "}"
            "QProgressBar::chunk {"
            " background-color: #5aa9e6;"
            " border-radius: 4px;"
            "}"
        )

        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(3000)

        self._save_settings_button = QtWidgets.QPushButton("設定保存")
        self._preview_button = QtWidgets.QPushButton("プレビュー生成")
        self._comparison_button = QtWidgets.QPushButton("1枚比較処理")
        self._test_button = QtWidgets.QPushButton("テスト処理（先頭5枚）")
        self._start_button = QtWidgets.QPushButton("一括処理開始")
        self._stop_button = QtWidgets.QPushButton("停止")
        self._cancel_button = QtWidgets.QPushButton("キャンセル")
        self._stop_button.setEnabled(False)
        self._cancel_button.setEnabled(False)

        self._build_ui()
        self._connect_signals()
        self._load_initial_settings()
        self._refresh_runtime_status()
        self._refresh_gimp_status()
        self._refresh_model_info()
        self._update_detail_visibility()

    def _build_ui(self) -> None:
        intro_label = QtWidgets.QLabel(
            "このツールは、GIMP で確認しながらノイズ除去とアンシャープマスクを調整し、"
            " 顔を作り替えずに低侵襲な高解像度化を行うための GUI です。"
        )
        intro_label.setWordWrap(True)

        tips_label = QtWidgets.QLabel(
            "最初は 顔付近crop プレビュー、GIMP前処理 ON、ノイズ除去 弱、アンシャープ 弱、"
            " SwinIR 2x、GIMP後処理 OFF から確認してください。"
        )
        tips_label.setWordWrap(True)

        warning_label = QtWidgets.QLabel(
            "強いノイズ除去や強いシャープは、本人性を損ねたり白フチ・黒フチ・ギラつきを出す可能性があります。"
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #9a3b00;")

        for scale in self._available_scales:
            self._scale_combo.addItem(f"{scale}x", scale)
        if not self._available_scales:
            self._scale_combo.addItem("利用不可", 0)
            self._scale_combo.setEnabled(False)
        else:
            preferred_scale = 2 if 2 in self._available_scales else self._available_scales[0]
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(preferred_scale))

        for label, value in PREVIEW_RANGES:
            self._preview_range_combo.addItem(label, value)
        self._preview_range_combo.setCurrentIndex(self._preview_range_combo.findData("face_near"))

        for ratio_text in CROP_RATIO_CHOICES:
            self._preview_ratio_combo.addItem(ratio_text, ratio_text)
        self._preview_ratio_combo.setCurrentText("4:5")

        gimp_button = QtWidgets.QPushButton("参照...")
        input_button = QtWidgets.QPushButton("参照...")
        output_button = QtWidgets.QPushButton("参照...")
        preview_button = QtWidgets.QPushButton("参照...")
        gimp_button.clicked.connect(self._pick_gimp_file)
        input_button.clicked.connect(lambda: self._pick_directory(self._input_edit))
        output_button.clicked.connect(lambda: self._pick_directory(self._output_edit))
        preview_button.clicked.connect(self._pick_preview_file)

        path_form = QtWidgets.QGridLayout()
        path_form.addWidget(QtWidgets.QLabel("GIMP 実行ファイル"), 0, 0)
        path_form.addWidget(self._gimp_path_edit, 0, 1)
        path_form.addWidget(gimp_button, 0, 2)
        path_form.addWidget(QtWidgets.QLabel("入力フォルダ"), 1, 0)
        path_form.addWidget(self._input_edit, 1, 1)
        path_form.addWidget(input_button, 1, 2)
        path_form.addWidget(QtWidgets.QLabel("出力フォルダ"), 2, 0)
        path_form.addWidget(self._output_edit, 2, 1)
        path_form.addWidget(output_button, 2, 2)
        path_form.addWidget(QtWidgets.QLabel("プレビュー対象画像 / 比較用1枚"), 3, 0)
        path_form.addWidget(self._preview_source_edit, 3, 1)
        path_form.addWidget(preview_button, 3, 2)
        path_form.addWidget(QtWidgets.QLabel("倍率"), 4, 0)
        path_form.addWidget(self._scale_combo, 4, 1)
        path_form.addWidget(QtWidgets.QLabel("プレビュー範囲"), 5, 0)
        path_form.addWidget(self._preview_range_combo, 5, 1)
        path_form.addWidget(QtWidgets.QLabel("プレビュー crop 比率"), 6, 0)
        path_form.addWidget(self._preview_ratio_combo, 6, 1)
        path_form.addWidget(QtWidgets.QLabel("同名出力時"), 7, 0)
        path_form.addWidget(self._collision_combo, 7, 1)
        path_form.addWidget(QtWidgets.QLabel("tile size"), 8, 0)
        path_form.addWidget(self._tile_size_spin, 8, 1)
        path_form.addWidget(QtWidgets.QLabel("tile overlap"), 9, 0)
        path_form.addWidget(self._tile_overlap_spin, 9, 1)
        path_form.addWidget(QtWidgets.QLabel("CPU/GPU 状態"), 10, 0)
        path_form.addWidget(self._device_label, 10, 1, 1, 2)
        path_form.addWidget(QtWidgets.QLabel("GIMP 状態"), 11, 0)
        path_form.addWidget(self._gimp_status_label, 11, 1, 1, 2)
        path_form.addWidget(QtWidgets.QLabel("内部 SwinIR モデル"), 12, 0)
        path_form.addWidget(self._model_info_label, 12, 1, 1, 2)

        pre_group = QtWidgets.QGroupBox("GIMP前処理")
        pre_layout = QtWidgets.QGridLayout(pre_group)
        pre_layout.addWidget(self._use_gimp_pre_checkbox, 0, 0, 1, 2)
        pre_layout.addWidget(QtWidgets.QLabel("ノイズ除去"), 1, 0)
        pre_layout.addWidget(self._noise_combo, 1, 1)
        pre_layout.addWidget(QtWidgets.QLabel("ノイズ詳細 半径"), 2, 0)
        pre_layout.addWidget(self._noise_radius_spin, 2, 1)
        pre_layout.addWidget(QtWidgets.QLabel("ノイズ詳細 黒レベル"), 3, 0)
        pre_layout.addWidget(self._noise_black_spin, 3, 1)
        pre_layout.addWidget(QtWidgets.QLabel("ノイズ詳細 白レベル"), 4, 0)
        pre_layout.addWidget(self._noise_white_spin, 4, 1)
        pre_layout.addWidget(QtWidgets.QLabel("ノイズ詳細 iterations"), 5, 0)
        pre_layout.addWidget(self._noise_iterations_spin, 5, 1)
        pre_layout.addWidget(QtWidgets.QLabel("アンシャープマスク"), 6, 0)
        pre_layout.addWidget(self._pre_unsharp_combo, 6, 1)
        pre_layout.addWidget(QtWidgets.QLabel("詳細 半径"), 7, 0)
        pre_layout.addWidget(self._pre_unsharp_radius_spin, 7, 1)
        pre_layout.addWidget(QtWidgets.QLabel("詳細 量"), 8, 0)
        pre_layout.addWidget(self._pre_unsharp_amount_spin, 8, 1)
        pre_layout.addWidget(QtWidgets.QLabel("詳細 しきい値"), 9, 0)
        pre_layout.addWidget(self._pre_unsharp_threshold_spin, 9, 1)

        pre_note = QtWidgets.QLabel(
            "GIMP 2 系では despeckle + unsharp、GIMP 3 系では GEGL noise-reduction + unsharp-mask を使います。"
        )
        pre_note.setWordWrap(True)
        pre_layout.addWidget(pre_note, 10, 0, 1, 2)

        post_group = QtWidgets.QGroupBox("後処理 / 切り抜き")
        post_layout = QtWidgets.QGridLayout(post_group)
        post_layout.addWidget(self._use_gimp_post_checkbox, 0, 0, 1, 2)
        post_layout.addWidget(QtWidgets.QLabel("GIMP後処理の強さ"), 1, 0)
        post_layout.addWidget(self._post_unsharp_combo, 1, 1)
        post_layout.addWidget(self._use_cutout_checkbox, 2, 0, 1, 2)
        post_note = QtWidgets.QLabel(
            "後処理は軽いアンシャープ中心です。人物切り抜きは PNG 透過で保存します。"
        )
        post_note.setWordWrap(True)
        post_layout.addWidget(post_note, 3, 0, 1, 2)

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(self._skip_checkbox)
        toggle_row.addWidget(self._recursive_checkbox)
        toggle_row.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._save_settings_button)
        button_row.addWidget(self._preview_button)
        button_row.addWidget(self._comparison_button)
        button_row.addWidget(self._test_button)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._stop_button)
        button_row.addWidget(self._cancel_button)
        button_row.addStretch(1)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.addWidget(intro_label)
        controls_layout.addWidget(tips_label)
        controls_layout.addWidget(warning_label)
        controls_layout.addLayout(path_form)
        controls_layout.addWidget(pre_group)
        controls_layout.addWidget(post_group)
        controls_layout.addLayout(toggle_row)
        controls_layout.addLayout(button_row)
        controls_layout.addWidget(self._preview_status_label)
        controls_layout.addWidget(self._current_file_label)
        controls_layout.addWidget(self._progress_bar)
        controls_layout.addWidget(self._log_view, stretch=1)

        preview_group = QtWidgets.QGroupBox("プレビュー比較")
        preview_grid = QtWidgets.QGridLayout(preview_group)
        for index, (key, caption) in enumerate(
            (
                ("original", "元画像"),
                ("gimp_pre", "GIMP前処理後"),
                ("swinir", "SwinIR後"),
                ("post", "SwinIR + GIMP後処理後"),
            )
        ):
            caption_label = QtWidgets.QLabel(caption)
            image_label = QtWidgets.QLabel("未生成")
            image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image_label.setMinimumSize(320, 240)
            image_label.setStyleSheet("border: 1px solid #c8c8c8; background: #fafafa;")
            image_label.setScaledContents(False)
            preview_grid.addWidget(caption_label, (index // 2) * 2, index % 2)
            preview_grid.addWidget(image_label, (index // 2) * 2 + 1, index % 2)
            self._preview_caption_labels[key] = caption_label
            self._preview_image_labels[key] = image_label

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(controls_widget)
        splitter.addWidget(preview_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self._preview_button.clicked.connect(self._start_preview)
        self._comparison_button.clicked.connect(self._start_comparison)
        self._test_button.clicked.connect(lambda: self._start_processing(test_mode=True))
        self._start_button.clicked.connect(lambda: self._start_processing(test_mode=False))
        self._stop_button.clicked.connect(self._request_stop)
        self._cancel_button.clicked.connect(self._request_cancel)
        self._save_settings_button.clicked.connect(self._save_current_settings)
        self._gimp_path_edit.editingFinished.connect(self._refresh_gimp_status)
        self._scale_combo.currentIndexChanged.connect(self._refresh_model_info)
        self._noise_combo.currentIndexChanged.connect(self._update_detail_visibility)
        self._pre_unsharp_combo.currentIndexChanged.connect(self._update_detail_visibility)

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)

    def _pick_directory(self, target_edit: QtWidgets.QLineEdit) -> None:
        start_dir = target_edit.text().strip() or str(APP_ROOT)
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "フォルダを選択", start_dir)
        if selected:
            target_edit.setText(selected)

    def _pick_gimp_file(self) -> None:
        start_path = self._gimp_path_edit.text().strip() or str(Path(r"C:\Program Files"))
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "GIMP 実行ファイルを選択",
            start_path,
            GIMP_FILTER,
        )
        if file_path:
            self._gimp_path_edit.setText(file_path)
            self._refresh_gimp_status()

    def _pick_preview_file(self) -> None:
        start_path = self._preview_source_edit.text().strip()
        if not start_path:
            start_path = self._input_edit.text().strip() or str(APP_ROOT)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "プレビュー対象画像を選択",
            start_path,
            SUPPORTED_FILE_FILTER,
        )
        if file_path:
            self._preview_source_edit.setText(file_path)

    def _refresh_runtime_status(self) -> None:
        status = get_runtime_status()
        self._device_label.setText(format_runtime_status(status))

    def _refresh_gimp_status(self) -> None:
        gimp_text = self._gimp_path_edit.text().strip()
        available, message = is_gimp_available(Path(gimp_text)) if gimp_text else (False, "GIMP 未設定")
        self._gimp_status_label.setText(message)
        self._gimp_status_label.setStyleSheet("" if available else "color: #9a3b00;")

    def _refresh_model_info(self) -> None:
        scale = self._selected_scale()
        if scale <= 0:
            self._model_info_label.setText("内部モデルが見つかりません。setup.bat を再実行してください。")
            return
        model_path = get_default_model_path(scale)
        if not model_path.exists():
            self._model_info_label.setText(f"x{scale} の内部モデルが見つかりません: {model_path}")
            return
        descriptor = infer_model_descriptor(model_path, scale)
        self._model_info_label.setText(
            f"x{scale} 固定 / {descriptor.label} / {model_path.name}"
        )

    def _selected_scale(self) -> int:
        current_data = self._scale_combo.currentData()
        return int(current_data or 0)

    def _selected_preview_range(self) -> str:
        return str(self._preview_range_combo.currentData() or "face_near")

    def _build_noise_settings(self) -> NoiseReductionSettings:
        return NoiseReductionSettings(
            preset=str(self._noise_combo.currentData() or "off"),
            detail_radius=float(self._noise_radius_spin.value()),
            detail_black_level=int(self._noise_black_spin.value()),
            detail_white_level=int(self._noise_white_spin.value()),
            detail_iterations=int(self._noise_iterations_spin.value()),
        )

    def _build_pre_unsharp_settings(self) -> UnsharpMaskSettings:
        return UnsharpMaskSettings(
            preset=str(self._pre_unsharp_combo.currentData() or "off"),
            detail_radius=float(self._pre_unsharp_radius_spin.value()),
            detail_amount=float(self._pre_unsharp_amount_spin.value()),
            detail_threshold=float(self._pre_unsharp_threshold_spin.value()),
        )

    def _build_post_unsharp_settings(self) -> UnsharpMaskSettings:
        return UnsharpMaskSettings(
            preset=str(self._post_unsharp_combo.currentData() or "off"),
            detail_radius=float(self._pre_unsharp_radius_spin.value()),
            detail_amount=float(self._pre_unsharp_amount_spin.value()),
            detail_threshold=float(self._pre_unsharp_threshold_spin.value()),
        )

    def _build_pipeline_options(self) -> PipelineOptions:
        gimp_text = self._gimp_path_edit.text().strip()
        return PipelineOptions(
            scale=self._selected_scale(),
            tile_size=int(self._tile_size_spin.value()),
            tile_overlap=int(self._tile_overlap_spin.value()),
            gimp_path=Path(gimp_text) if gimp_text else None,
            use_gimp_pre=self._use_gimp_pre_checkbox.isChecked(),
            noise_settings=self._build_noise_settings(),
            pre_unsharp_settings=self._build_pre_unsharp_settings(),
            use_gimp_post=self._use_gimp_post_checkbox.isChecked(),
            post_unsharp_settings=self._build_post_unsharp_settings(),
            use_cutout=self._use_cutout_checkbox.isChecked(),
        )

    def _build_batch_options(self, *, test_mode: bool) -> BatchOptions:
        return BatchOptions(
            input_dir=Path(self._input_edit.text().strip()),
            output_dir=Path(self._output_edit.text().strip()),
            recursive=self._recursive_checkbox.isChecked(),
            skip_existing=self._skip_checkbox.isChecked(),
            collision_policy=str(self._collision_combo.currentData() or "skip"),
            pipeline=self._build_pipeline_options(),
            test_mode=test_mode,
            test_limit=5,
        )

    def _build_preview_options(self) -> PreviewOptions:
        return PreviewOptions(
            input_file=Path(self._preview_source_edit.text().strip()),
            preview_range=self._selected_preview_range(),
            preview_ratio=str(self._preview_ratio_combo.currentData() or self._preview_ratio_combo.currentText()),
            pipeline=self._build_pipeline_options(),
        )

    def _build_comparison_options(self) -> ComparisonOptions:
        return ComparisonOptions(
            input_file=Path(self._preview_source_edit.text().strip()),
            output_dir=Path(self._output_edit.text().strip()),
            skip_existing=self._skip_checkbox.isChecked(),
            collision_policy=str(self._collision_combo.currentData() or "skip"),
            pipeline=self._build_pipeline_options(),
        )

    def _validate_before_run(self, *, for_preview: bool) -> bool:
        if self._selected_scale() not in {2, 4}:
            QtWidgets.QMessageBox.warning(
                self,
                "内部モデル",
                "x2 または x4 の内部モデルが見つかりません。setup.bat を再実行してください。",
            )
            return False
        if not self._output_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "出力フォルダ", "出力フォルダを先に指定してください。")
            return False
        if for_preview:
            if not self._preview_source_edit.text().strip():
                QtWidgets.QMessageBox.warning(
                    self,
                    "プレビュー対象画像",
                    "プレビュー対象画像を 1 枚選択してください。",
                )
                return False
        else:
            if not self._input_edit.text().strip():
                QtWidgets.QMessageBox.warning(self, "入力フォルダ", "入力フォルダを先に指定してください。")
                return False

        if self._use_gimp_pre_checkbox.isChecked() or self._use_gimp_post_checkbox.isChecked():
            gimp_text = self._gimp_path_edit.text().strip()
            if not gimp_text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "GIMP パス",
                    "GIMP 前処理または後処理を使う場合は、GIMP 実行ファイルを指定してください。",
                )
                return False
            available, message = is_gimp_available(Path(gimp_text))
            if not available:
                QtWidgets.QMessageBox.warning(self, "GIMP パス", message)
                return False
        return True

    def _estimate_batch_count(self, recursive: bool) -> int:
        input_text = self._input_edit.text().strip()
        if not input_text:
            return 0
        input_dir = Path(input_text)
        if not input_dir.exists():
            return 0
        pattern = "**/*" if recursive else "*"
        return sum(
            1
            for path in input_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

    def _confirm_batch_run(self, *, test_mode: bool) -> bool:
        file_count = self._estimate_batch_count(self._recursive_checkbox.isChecked())
        if file_count <= 0:
            QtWidgets.QMessageBox.warning(self, "入力フォルダ", "入力フォルダ内に対応画像がありません。")
            return False

        pipeline = self._build_pipeline_options()
        title = "テスト処理確認" if test_mode else "一括処理確認"
        message = (
            f"対象枚数: {min(file_count, 5) if test_mode else file_count}\n"
            f"倍率: x{pipeline.scale}\n"
            f"GIMP前処理: {'ON' if pipeline.use_gimp_pre else 'OFF'} / "
            f"ノイズ={describe_noise_settings(pipeline.noise_settings)} / "
            f"アンシャープ={describe_unsharp_settings(pipeline.pre_unsharp_settings)}\n"
            f"GIMP後処理: {'ON' if pipeline.use_gimp_post else 'OFF'} / "
            f"{pipeline.post_unsharp_settings.preset}\n"
            f"人物切り抜き: {'ON' if pipeline.use_cutout else 'OFF'}\n"
            f"tile size / overlap: {pipeline.tile_size} / {pipeline.tile_overlap}\n"
            f"出力先: {self._output_edit.text().strip()}"
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _start_preview(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._validate_before_run(for_preview=True):
            return

        options = self._build_preview_options()
        task_callable = partial(run_preview, options)
        self._launch_worker(task_callable, "プレビュー生成を開始します。")

    def _start_comparison(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._validate_before_run(for_preview=True):
            return
        options = self._build_comparison_options()
        task_callable = partial(run_comparison, options)
        self._launch_worker(
            task_callable,
            "1枚比較処理を開始します。comparison フォルダへ 6 パターンを出力します。",
        )

    def _start_processing(self, *, test_mode: bool) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._validate_before_run(for_preview=False):
            return
        if not self._confirm_batch_run(test_mode=test_mode):
            return

        options = self._build_batch_options(test_mode=test_mode)
        task_callable = partial(run_batch, options)
        self._launch_worker(
            task_callable,
            "テスト処理を開始します。" if test_mode else "一括処理を開始します。",
        )

    def _launch_worker(self, task_callable, start_message: str) -> None:
        self._append_log(start_message)
        self._progress_bar.setValue(0)
        self._current_file_label.setText("現在のファイル: -")
        self._set_running_state(True)

        self._worker = TaskWorker(task_callable)
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
            self._append_log("停止要求を受け付けました。現在の処理単位完了後に停止します。")

    def _request_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._append_log("キャンセル要求を受け付けました。できるだけ早く中断します。")

    def _handle_finished(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind == "preview":
            self._handle_preview_finished(payload)
            return
        self._handle_run_finished(payload)

    def _handle_preview_finished(self, payload: dict) -> None:
        self._set_running_state(False)
        self._progress_bar.setValue(100)
        self._preview_status_label.setText(payload.get("message", "プレビュー生成完了"))
        self._load_preview_pixmap("original", payload.get("original_preview_path", ""))
        self._load_preview_pixmap("gimp_pre", payload.get("gimp_pre_preview_path", ""))
        self._load_preview_pixmap("swinir", payload.get("swinir_preview_path", ""))
        self._load_preview_pixmap("post", payload.get("post_preview_path", ""))
        self._append_log(payload.get("message", "プレビュー生成完了"))
        self._worker = None

    def _handle_run_finished(self, summary: dict) -> None:
        self._set_running_state(False)
        total = max(int(summary.get("total_files", 0)), 1)
        completed = (
            int(summary.get("processed", 0))
            + int(summary.get("skipped", 0))
            + int(summary.get("failed", 0))
        )
        self._progress_bar.setValue(int((completed / total) * 100))
        self._append_log(
            "処理完了。"
            f" 成功={summary.get('processed', 0)}, スキップ={summary.get('skipped', 0)}, "
            f"失敗={summary.get('failed', 0)}, 警告={summary.get('warnings', 0)}, "
            f"停止={summary.get('stopped', False)}, キャンセル={summary.get('cancelled', False)}"
        )
        message = (
            f"出力先: {summary.get('output_dir', '')}\n"
            f"処理成功: {summary.get('processed', 0)}\n"
            f"スキップ: {summary.get('skipped', 0)}\n"
            f"失敗: {summary.get('failed', 0)}\n"
            f"警告: {summary.get('warnings', 0)}\n"
            f"停止: {summary.get('stopped', False)}\n"
            f"キャンセル: {summary.get('cancelled', False)}"
        )
        QtWidgets.QMessageBox.information(self, "処理完了", message)
        self._worker = None

    def _handle_error(self, message: str) -> None:
        self._set_running_state(False)
        self._append_log(f"エラー: {message}")
        QtWidgets.QMessageBox.critical(self, "処理エラー", message)
        self._worker = None

    def _set_running_state(self, is_running: bool) -> None:
        self._save_settings_button.setEnabled(not is_running)
        self._preview_button.setEnabled(not is_running)
        self._comparison_button.setEnabled(not is_running)
        self._test_button.setEnabled(not is_running)
        self._start_button.setEnabled(not is_running)
        self._stop_button.setEnabled(is_running)
        self._cancel_button.setEnabled(is_running)

    def _load_preview_pixmap(self, key: str, path_text: str) -> None:
        label = self._preview_image_labels[key]
        if not path_text:
            label.setText("未生成")
            label.setPixmap(QtGui.QPixmap())
            return
        pixmap = QtGui.QPixmap(path_text)
        if pixmap.isNull():
            label.setText("読込失敗")
            label.setPixmap(QtGui.QPixmap())
            return
        scaled = pixmap.scaled(
            label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setText("")

    def _update_detail_visibility(self) -> None:
        noise_is_detail = str(self._noise_combo.currentData()) == "detail"
        pre_unsharp_is_detail = str(self._pre_unsharp_combo.currentData()) == "detail"
        for widget in (
            self._noise_radius_spin,
            self._noise_black_spin,
            self._noise_white_spin,
            self._noise_iterations_spin,
        ):
            widget.setEnabled(noise_is_detail)
        for widget in (
            self._pre_unsharp_radius_spin,
            self._pre_unsharp_amount_spin,
            self._pre_unsharp_threshold_spin,
        ):
            widget.setEnabled(pre_unsharp_is_detail)

    def _save_current_settings(self) -> None:
        data = {
            "gimp_path": self._gimp_path_edit.text().strip(),
            "input_dir": self._input_edit.text().strip(),
            "output_dir": self._output_edit.text().strip(),
            "preview_source": self._preview_source_edit.text().strip(),
            "scale": self._selected_scale(),
            "preview_range": self._selected_preview_range(),
            "preview_ratio": str(
                self._preview_ratio_combo.currentData() or self._preview_ratio_combo.currentText()
            ),
            "collision_policy": str(self._collision_combo.currentData() or "skip"),
            "tile_size": int(self._tile_size_spin.value()),
            "tile_overlap": int(self._tile_overlap_spin.value()),
            "skip_existing": self._skip_checkbox.isChecked(),
            "recursive": self._recursive_checkbox.isChecked(),
            "use_gimp_pre": self._use_gimp_pre_checkbox.isChecked(),
            "noise_preset": str(self._noise_combo.currentData() or "off"),
            "noise_detail_radius": float(self._noise_radius_spin.value()),
            "noise_detail_black": int(self._noise_black_spin.value()),
            "noise_detail_white": int(self._noise_white_spin.value()),
            "noise_detail_iterations": int(self._noise_iterations_spin.value()),
            "pre_unsharp_preset": str(self._pre_unsharp_combo.currentData() or "off"),
            "pre_unsharp_radius": float(self._pre_unsharp_radius_spin.value()),
            "pre_unsharp_amount": float(self._pre_unsharp_amount_spin.value()),
            "pre_unsharp_threshold": float(self._pre_unsharp_threshold_spin.value()),
            "use_gimp_post": self._use_gimp_post_checkbox.isChecked(),
            "post_unsharp_preset": str(self._post_unsharp_combo.currentData() or "off"),
            "use_cutout": self._use_cutout_checkbox.isChecked(),
        }
        save_settings(data)
        self._append_log("設定を config.json に保存しました。")

    def _load_initial_settings(self) -> None:
        data = load_settings()
        default_gimp = find_first_gimp_path()
        if data.get("gimp_path"):
            self._gimp_path_edit.setText(str(data["gimp_path"]))
        elif default_gimp is not None:
            self._gimp_path_edit.setText(str(default_gimp))

        self._input_edit.setText(str(data.get("input_dir", "")))
        self._output_edit.setText(str(data.get("output_dir", DEFAULT_OUTPUT_DIR)))
        self._preview_source_edit.setText(str(data.get("preview_source", "")))

        scale = int(data.get("scale", 2 if 2 in self._available_scales else (self._available_scales[0] if self._available_scales else 0)))
        scale_index = self._scale_combo.findData(scale)
        if scale_index >= 0:
            self._scale_combo.setCurrentIndex(scale_index)

        range_index = self._preview_range_combo.findData(str(data.get("preview_range", "face_near")))
        if range_index >= 0:
            self._preview_range_combo.setCurrentIndex(range_index)
        self._preview_ratio_combo.setCurrentText(str(data.get("preview_ratio", "4:5")))

        collision_index = self._collision_combo.findData(str(data.get("collision_policy", "skip")))
        if collision_index >= 0:
            self._collision_combo.setCurrentIndex(collision_index)

        self._tile_size_spin.setValue(int(data.get("tile_size", 400)))
        self._tile_overlap_spin.setValue(int(data.get("tile_overlap", 32)))
        self._skip_checkbox.setChecked(bool(data.get("skip_existing", True)))
        self._recursive_checkbox.setChecked(bool(data.get("recursive", False)))

        self._use_gimp_pre_checkbox.setChecked(bool(data.get("use_gimp_pre", True)))
        noise_index = self._noise_combo.findData(str(data.get("noise_preset", "weak")))
        if noise_index >= 0:
            self._noise_combo.setCurrentIndex(noise_index)
        self._noise_radius_spin.setValue(float(data.get("noise_detail_radius", 3.0)))
        self._noise_black_spin.setValue(int(data.get("noise_detail_black", 1)))
        self._noise_white_spin.setValue(int(data.get("noise_detail_white", 248)))
        self._noise_iterations_spin.setValue(int(data.get("noise_detail_iterations", 1)))

        pre_index = self._pre_unsharp_combo.findData(str(data.get("pre_unsharp_preset", "weak")))
        if pre_index >= 0:
            self._pre_unsharp_combo.setCurrentIndex(pre_index)
        self._pre_unsharp_radius_spin.setValue(float(data.get("pre_unsharp_radius", 1.2)))
        self._pre_unsharp_amount_spin.setValue(float(data.get("pre_unsharp_amount", 0.35)))
        self._pre_unsharp_threshold_spin.setValue(float(data.get("pre_unsharp_threshold", 0.02)))

        self._use_gimp_post_checkbox.setChecked(bool(data.get("use_gimp_post", False)))
        post_index = self._post_unsharp_combo.findData(str(data.get("post_unsharp_preset", "off")))
        if post_index >= 0:
            self._post_unsharp_combo.setCurrentIndex(post_index)
        self._use_cutout_checkbox.setChecked(bool(data.get("use_cutout", False)))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI behavior
        self._save_current_settings()
        super().closeEvent(event)
