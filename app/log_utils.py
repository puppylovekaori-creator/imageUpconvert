from __future__ import annotations

import csv
from pathlib import Path


PROCESSING_HEADERS = [
    "who",
    "processed_at",
    "task_kind",
    "status",
    "processing_mode",
    "input_file",
    "output_file",
    "relative_input",
    "original_image_size",
    "output_image_size",
    "gimp_path",
    "gimp_version",
    "execution_method",
    "model",
    "scale",
    "gimp_pre_enabled",
    "noise_reduction_setting",
    "unsharp_setting",
    "error_message",
    "processing_time_seconds",
]


class CsvLogger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._processing_file = (output_dir / "processing_log.csv").open(
            "a", newline="", encoding="utf-8-sig"
        )
        self._failed_file = (output_dir / "failed_log.csv").open(
            "a", newline="", encoding="utf-8-sig"
        )
        self._processing_writer = csv.DictWriter(
            self._processing_file, fieldnames=PROCESSING_HEADERS
        )
        self._failed_writer = csv.DictWriter(self._failed_file, fieldnames=PROCESSING_HEADERS)

        if self._processing_file.tell() == 0:
            self._processing_writer.writeheader()
        if self._failed_file.tell() == 0:
            self._failed_writer.writeheader()

    def log_processing(self, row: dict[str, str]) -> None:
        self._processing_writer.writerow(row)
        self._processing_file.flush()

    def log_failed(self, row: dict[str, str]) -> None:
        self._failed_writer.writerow(row)
        self._failed_file.flush()

    def close(self) -> None:
        self._processing_file.close()
        self._failed_file.close()
