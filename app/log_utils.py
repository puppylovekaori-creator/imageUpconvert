from __future__ import annotations

import csv
from pathlib import Path


PROCESSING_HEADERS = [
    "processed_at",
    "input_file_path",
    "output_file_path",
    "original_image_size",
    "output_image_size",
    "scale",
    "model",
    "tile_size",
    "tile_overlap",
    "alpha_present",
    "processing_result",
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
