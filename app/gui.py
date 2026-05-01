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
    PROCESSING_MODE_GIMP_ONLY,
    PROCESSING_MODE_GIMP_PRE_SWINIR,
    PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST,
    PROCESSING_MODE_SWINIR_ONLY,
    get_default_model_path,
    infer_model_descriptor,
    list_available_internal_scales,
    normalize_processing_mode,
    pipeline_uses_gimp_post,
    pipeline_uses_gimp_pre,
    pipeline_uses_swinir,
    processing_mode_uses_gimp_post,
    processing_mode_uses_gimp_pre,
    processing_mode_uses_swinir,
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
PROCESSING_MODE_CHOICES = (
    ("GIMPのみ", PROCESSING_MODE_GIMP_ONLY),
    ("SwinIRのみ", PROCESSING_MODE_SWINIR_ONLY),
    ("GIMP前処理 + SwinIR", PROCESSING_MODE_GIMP_PRE_SWINIR),
    ("GIMP前処理 + SwinIR + GIMP後処理", PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST),
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


class PreviewWindow(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("プレビュー")
        self.resize(1180, 820)
        self._current_path: str = ""
        self._current_pixmap = QtGui.QPixmap()

        self._status_label = QtWidgets.QLabel("プレビュー未生成")
        self._status_label.setWordWrap(True)

        self._list_widget = QtWidgets.QListWidget()
        self._list_widget.setMinimumWidth(240)
        self._list_widget.currentItemChanged.connect(self._handle_selection_changed)

        self._image_label = QtWidgets.QLabel("画像未選択")
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(720, 540)
        self._image_label.setStyleSheet("border: 1px solid #c8c8c8; background: #fafafa;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._status_label)

        content_row = QtWidgets.QHBoxLayout()
        content_row.addWidget(self._list_widget)
        content_row.addWidget(self._image_label, stretch=1)
        layout.addLayout(content_row, stretch=1)

    def set_preview_items(self, items: list[tuple[str, str]]) -> None:
        self._list_widget.clear()
        self._current_path = ""
        self._current_pixmap = QtGui.QPixmap()
        self._image_label.setText("画像未選択")
        self._image_label.setPixmap(QtGui.QPixmap())

        if not items:
            self._status_label.setText("表示できるプレビュー画像がありません。")
            return

        self._status_label.setText(f"クリックで切り替えできます。候補数: {len(items)}")
        for label, path_text in items:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path_text)
            self._list_widget.addItem(item)
        self._list_widget.setCurrentRow(0)

    def _handle_selection_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            self._current_path = ""
            self._current_pixmap = QtGui.QPixmap()
            self._image_label.setText("画像未選択")
            self._image_label.setPixmap(QtGui.QPixmap())
            return

        path_text = str(current.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        self._current_path = path_text
        pixmap = QtGui.QPixmap(path_text)
        if pixmap.isNull():
            self._current_pixmap = QtGui.QPixmap()
            self._image_label.setText("読込失敗")
            self._image_label.setPixmap(QtGui.QPixmap())
            return

        self._current_pixmap = pixmap
        self._render_current_pixmap()

    def _render_current_pixmap(self) -> None:
        if self._current_pixmap.isNull():
            return
        scaled = self._current_pixmap.scaled(
            self._image_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - GUI behavior
        super().resizeEvent(event)
        if not self._current_pixmap.isNull():
            self._render_current_pixmap()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GIMP / SwinIR 低侵襲人物画像処理 GUI")
        self.resize(1040, 820)
        self._worker: TaskWorker | None = None
        self._available_scales = list_available_internal_scales()
        self._preview_window = PreviewWindow(self)
        self._latest_preview_items: list[tuple[str, str]] = []

        self._gimp_path_edit = QtWidgets.QLineEdit()
        self._input_edit = QtWidgets.QLineEdit()
        self._output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self._preview_source_edit = QtWidgets.QLineEdit()
        self._processing_mode_combo = QtWidgets.QComboBox()
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
        self._use_gimp_pre_checkbox.hide()
        self._use_gimp_post_checkbox.hide()

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
        self._update_mode_state()

    def _build_ui(self) -> None:
        intro_label = QtWidgets.QLabel(
            "このツールは、GIMP のノイズ除去とアンシャープマスクを軸に、"
            " 顔を作り替えずに人物画像の見やすさ改善と必要時の高解像度化を行う GUI です。"
        )
        intro_label.setWordWrap(True)

        tips_label = QtWidgets.QLabel(
            "最初は 顔付近crop プレビュー、処理モード GIMPのみ または GIMP前処理 + SwinIR、"
            " ノイズ除去 弱、アンシャープ 弱 から確認してください。"
        )
        tips_label.setWordWrap(True)

        warning_label = QtWidgets.QLabel(
            "強いノイズ除去や強いシャープは、本人性を損ねたり白フチ・黒フチ・ギラつきを出す可能性があります。"
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #9a3b00;")

        for label, value in PROCESSING_MODE_CHOICES:
            self._processing_mode_combo.addItem(label, value)

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
        path_form.addWidget(QtWidgets.QLabel("入力フォルダ"), 0, 0)
        path_form.addWidget(self._input_edit, 0, 1)
        path_form.addWidget(input_button, 0, 2)
        path_form.addWidget(QtWidgets.QLabel("出力フォルダ"), 1, 0)
        path_form.addWidget(self._output_edit, 1, 1)
        path_form.addWidget(output_button, 1, 2)
        path_form.addWidget(QtWidgets.QLabel("プレビュー対象画像 / 比較用1枚"), 2, 0)
        path_form.addWidget(self._preview_source_edit, 2, 1)
        path_form.addWidget(preview_button, 2, 2)
        path_form.addWidget(QtWidgets.QLabel("処理モード"), 3, 0)
        path_form.addWidget(self._processing_mode_combo, 3, 1)

        pre_group = QtWidgets.QGroupBox("GIMP前処理")
        pre_layout = QtWidgets.QGridLayout(pre_group)
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
            "GIMPのみ / GIMP前処理 + SwinIR モードで使います。"
            " GIMP 2 系では despeckle + unsharp、GIMP 3 系では GEGL noise-reduction + unsharp-mask を使います。"
        )
        pre_note.setWordWrap(True)
        pre_layout.addWidget(pre_note, 10, 0, 1, 2)

        post_group = QtWidgets.QGroupBox("後処理 / 切り抜き")
        post_layout = QtWidgets.QGridLayout(post_group)
        post_layout.addWidget(QtWidgets.QLabel("GIMP後処理の強さ"), 1, 0)
        post_layout.addWidget(self._post_unsharp_combo, 1, 1)
        post_layout.addWidget(self._use_cutout_checkbox, 2, 0, 1, 2)
        post_note = QtWidgets.QLabel(
            "GIMP後処理は「GIMP前処理 + SwinIR + GIMP後処理」モードで使います。"
            " 人物切り抜きは処理モードとは別に ON/OFF でき、PNG 透過で保存します。"
        )
        post_note.setWordWrap(True)
        post_layout.addWidget(post_note, 3, 0, 1, 2)

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(self._skip_checkbox)
        toggle_row.addWidget(self._recursive_checkbox)
        toggle_row.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._preview_button)
        button_row.addWidget(self._comparison_button)
        button_row.addWidget(self._test_button)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._stop_button)
        button_row.addWidget(self._cancel_button)
        button_row.addStretch(1)

        run_tab = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_tab)
        run_layout.addWidget(intro_label)
        run_layout.addWidget(tips_label)
        run_layout.addWidget(warning_label)

        source_group = QtWidgets.QGroupBox("実行対象")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        source_layout.addLayout(path_form)
        run_layout.addWidget(source_group)

        execution_group = QtWidgets.QGroupBox("実行")
        execution_layout = QtWidgets.QVBoxLayout(execution_group)
        execution_layout.addLayout(toggle_row)
        execution_layout.addLayout(button_row)
        execution_layout.addWidget(self._preview_status_label)
        execution_layout.addWidget(self._current_file_label)
        execution_layout.addWidget(self._progress_bar)
        run_layout.addWidget(execution_group)

        log_group = QtWidgets.QGroupBox("ログ")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.addWidget(self._log_view)
        run_layout.addWidget(log_group, stretch=1)

        settings_tab = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)

        env_group = QtWidgets.QGroupBox("環境 / SwinIR")
        env_layout = QtWidgets.QGridLayout(env_group)
        env_layout.addWidget(QtWidgets.QLabel("GIMP 実行ファイル"), 0, 0)
        env_layout.addWidget(self._gimp_path_edit, 0, 1)
        env_layout.addWidget(gimp_button, 0, 2)
        env_layout.addWidget(QtWidgets.QLabel("GIMP 状態"), 1, 0)
        env_layout.addWidget(self._gimp_status_label, 1, 1, 1, 2)
        env_layout.addWidget(QtWidgets.QLabel("CPU/GPU 状態"), 2, 0)
        env_layout.addWidget(self._device_label, 2, 1, 1, 2)
        env_layout.addWidget(QtWidgets.QLabel("内部 SwinIR モデル"), 3, 0)
        env_layout.addWidget(self._model_info_label, 3, 1, 1, 2)
        env_layout.addWidget(QtWidgets.QLabel("倍率"), 4, 0)
        env_layout.addWidget(self._scale_combo, 4, 1)
        env_layout.addWidget(QtWidgets.QLabel("同名出力時"), 5, 0)
        env_layout.addWidget(self._collision_combo, 5, 1)
        env_layout.addWidget(QtWidgets.QLabel("tile size"), 6, 0)
        env_layout.addWidget(self._tile_size_spin, 6, 1)
        env_layout.addWidget(QtWidgets.QLabel("tile overlap"), 7, 0)
        env_layout.addWidget(self._tile_overlap_spin, 7, 1)
        settings_layout.addWidget(env_group)

        preview_group = QtWidgets.QGroupBox("プレビュー設定")
        preview_layout = QtWidgets.QGridLayout(preview_group)
        preview_layout.addWidget(QtWidgets.QLabel("プレビュー範囲"), 0, 0)
        preview_layout.addWidget(self._preview_range_combo, 0, 1)
        preview_layout.addWidget(QtWidgets.QLabel("プレビュー crop 比率"), 1, 0)
        preview_layout.addWidget(self._preview_ratio_combo, 1, 1)
        preview_note = QtWidgets.QLabel(
            "プレビュー生成後は別ウィンドウで開きます。左側の一覧をクリックして表示を切り替えます。"
        )
        preview_note.setWordWrap(True)
        preview_layout.addWidget(preview_note, 2, 0, 1, 2)
        settings_layout.addWidget(preview_group)

        settings_button_row = QtWidgets.QHBoxLayout()
        settings_button_row.addWidget(self._save_settings_button)
        settings_button_row.addStretch(1)
        settings_layout.addLayout(settings_button_row)
        settings_layout.addWidget(pre_group)
        settings_layout.addWidget(post_group)
        settings_layout.addStretch(1)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(run_tab, "実行")
        tabs.addTab(settings_tab, "設定")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(tabs, stretch=1)
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
        self._processing_mode_combo.currentIndexChanged.connect(self._update_mode_state)
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
        mode = self._selected_processing_mode()
        if not processing_mode_uses_swinir(mode):
            self._model_info_label.setText("SwinIR を使わない処理モードです。")
            return
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

    def _selected_processing_mode(self) -> str:
        return normalize_processing_mode(
            str(self._processing_mode_combo.currentData() or PROCESSING_MODE_GIMP_PRE_SWINIR)
        )

    def _processing_mode_label(self, mode: str | None = None) -> str:
        target_mode = normalize_processing_mode(mode or self._selected_processing_mode())
        for label, value in PROCESSING_MODE_CHOICES:
            if value == target_mode:
                return label
        return target_mode

    def _selected_scale(self) -> int:
        current_data = self._scale_combo.currentData()
        return int(current_data or 0)

    def _selected_preview_range(self) -> str:
        return str(self._preview_range_combo.currentData() or "face_near")

    def _preview_label_map(self, mode: str | None = None) -> dict[str, str]:
        target_mode = normalize_processing_mode(mode or self._selected_processing_mode())
        labels = {"original": "元画像", "post": "最終出力"}
        if target_mode == PROCESSING_MODE_GIMP_ONLY:
            labels["gimp_pre"] = "GIMP処理後"
            labels["swinir"] = "未使用"
        elif target_mode == PROCESSING_MODE_SWINIR_ONLY:
            labels["gimp_pre"] = "未使用"
            labels["swinir"] = "SwinIR後"
        else:
            labels["gimp_pre"] = "GIMP前処理後"
            labels["swinir"] = "SwinIRのみ参考"
        return labels

    def _update_mode_state(self) -> None:
        mode = self._selected_processing_mode()
        uses_swinir = processing_mode_uses_swinir(mode)
        uses_gimp_pre = processing_mode_uses_gimp_pre(mode)
        uses_gimp_post = processing_mode_uses_gimp_post(mode)

        self._scale_combo.setEnabled(uses_swinir and bool(self._available_scales))
        self._tile_size_spin.setEnabled(uses_swinir)
        self._tile_overlap_spin.setEnabled(uses_swinir)
        self._noise_combo.setEnabled(uses_gimp_pre)
        self._post_unsharp_combo.setEnabled(uses_gimp_post)
        self._use_gimp_pre_checkbox.setChecked(uses_gimp_pre)
        self._use_gimp_post_checkbox.setChecked(uses_gimp_post)

        self._refresh_model_info()
        self._update_detail_visibility()

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
            processing_mode=self._selected_processing_mode(),
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

    def _validate_before_run(self, *, task_kind: str) -> bool:
        mode = self._selected_processing_mode()
        requires_swinir = task_kind == "comparison" or processing_mode_uses_swinir(mode)
        requires_gimp = task_kind == "comparison" or processing_mode_uses_gimp_pre(mode) or processing_mode_uses_gimp_post(mode)

        if requires_swinir and self._selected_scale() not in {2, 4}:
            QtWidgets.QMessageBox.warning(
                self,
                "内部モデル",
                "x2 または x4 の内部モデルが見つかりません。setup.bat を再実行してください。",
            )
            return False
        if not self._output_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "出力フォルダ", "出力フォルダを先に指定してください。")
            return False
        if task_kind in {"preview", "comparison"}:
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

        if requires_gimp:
            gimp_text = self._gimp_path_edit.text().strip()
            if not gimp_text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "GIMP パス",
                    "この処理には GIMP が必要です。GIMP 実行ファイルを指定してください。",
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
        uses_swinir = pipeline_uses_swinir(pipeline)
        uses_gimp_pre = pipeline_uses_gimp_pre(pipeline)
        uses_gimp_post = pipeline_uses_gimp_post(pipeline)
        title = "テスト処理確認" if test_mode else "一括処理確認"
        lines = [
            f"対象枚数: {min(file_count, 5) if test_mode else file_count}",
            f"処理モード: {self._processing_mode_label(pipeline.processing_mode)}",
            (
                f"GIMP前処理: {'ON' if uses_gimp_pre else 'OFF'} / "
                f"ノイズ={describe_noise_settings(pipeline.noise_settings) if uses_gimp_pre else '-'} / "
                f"アンシャープ={describe_unsharp_settings(pipeline.pre_unsharp_settings) if uses_gimp_pre else '-'}"
            ),
            (
                f"GIMP後処理: {'ON' if uses_gimp_post else 'OFF'} / "
                f"{pipeline.post_unsharp_settings.preset if uses_gimp_post else '-'}"
            ),
            f"人物切り抜き: {'ON' if pipeline.use_cutout else 'OFF'}",
        ]
        if uses_swinir:
            lines.append(f"倍率: x{pipeline.scale}")
            lines.append(f"tile size / overlap: {pipeline.tile_size} / {pipeline.tile_overlap}")
        else:
            lines.append("SwinIR: この処理では使いません")
        lines.append(f"出力先: {self._output_edit.text().strip()}")
        message = "\n".join(lines)
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
        if not self._validate_before_run(task_kind="preview"):
            return

        options = self._build_preview_options()
        task_callable = partial(run_preview, options)
        self._launch_worker(task_callable, "プレビュー生成を開始します。")

    def _start_comparison(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._validate_before_run(task_kind="comparison"):
            return
        options = self._build_comparison_options()
        task_callable = partial(run_comparison, options)
        variant_count = 7 if self._use_cutout_checkbox.isChecked() else 5
        self._launch_worker(
            task_callable,
            f"1枚比較処理を開始します。comparison フォルダへ {variant_count} パターンを出力します。",
        )

    def _start_processing(self, *, test_mode: bool) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._validate_before_run(task_kind="batch"):
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
        labels = self._preview_label_map()
        self._latest_preview_items = []
        for key in ("original", "gimp_pre", "swinir", "post"):
            path_text = str(payload.get(f"{key}_preview_path", "") or "")
            if path_text:
                self._latest_preview_items.append((labels.get(key, key), path_text))
        self._preview_window.set_preview_items(self._latest_preview_items)
        self._preview_window.show()
        self._preview_window.raise_()
        self._preview_window.activateWindow()
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

    def _update_detail_visibility(self) -> None:
        mode = self._selected_processing_mode()
        uses_gimp_pre = processing_mode_uses_gimp_pre(mode)
        noise_is_detail = str(self._noise_combo.currentData()) == "detail"
        pre_unsharp_is_detail = str(self._pre_unsharp_combo.currentData()) == "detail"
        for widget in (
            self._noise_radius_spin,
            self._noise_black_spin,
            self._noise_white_spin,
            self._noise_iterations_spin,
        ):
            widget.setEnabled(uses_gimp_pre and noise_is_detail)
        for widget in (
            self._pre_unsharp_radius_spin,
            self._pre_unsharp_amount_spin,
            self._pre_unsharp_threshold_spin,
        ):
            widget.setEnabled(uses_gimp_pre and pre_unsharp_is_detail)

    def _save_current_settings(self) -> None:
        data = {
            "gimp_path": self._gimp_path_edit.text().strip(),
            "input_dir": self._input_edit.text().strip(),
            "output_dir": self._output_edit.text().strip(),
            "preview_source": self._preview_source_edit.text().strip(),
            "processing_mode": self._selected_processing_mode(),
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

        saved_mode = str(data.get("processing_mode", "")).strip()
        if not saved_mode:
            use_gimp_pre = bool(data.get("use_gimp_pre", True))
            use_gimp_post = bool(data.get("use_gimp_post", False))
            if use_gimp_pre and use_gimp_post:
                saved_mode = PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST
            elif use_gimp_pre:
                saved_mode = PROCESSING_MODE_GIMP_PRE_SWINIR
            else:
                saved_mode = PROCESSING_MODE_SWINIR_ONLY
        mode_index = self._processing_mode_combo.findData(saved_mode)
        if mode_index >= 0:
            self._processing_mode_combo.setCurrentIndex(mode_index)

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

        post_index = self._post_unsharp_combo.findData(str(data.get("post_unsharp_preset", "off")))
        if post_index >= 0:
            self._post_unsharp_combo.setCurrentIndex(post_index)
        self._use_cutout_checkbox.setChecked(bool(data.get("use_cutout", False)))
        self._update_mode_state()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI behavior
        self._save_current_settings()
        super().closeEvent(event)
