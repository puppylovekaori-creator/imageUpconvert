from __future__ import annotations

import argparse
import os
import sys

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from .gui import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args, remaining_args = parser.parse_known_args()

    if args.smoke_test and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication([sys.argv[0], *remaining_args])
    app.setApplicationName("GIMP AI Upscale 一括処理GUIツール")

    window = MainWindow()
    if args.smoke_test:
        QtCore.QTimer.singleShot(0, app.quit)
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
