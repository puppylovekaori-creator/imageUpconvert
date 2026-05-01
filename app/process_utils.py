from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class StopRequestedError(RuntimeError):
    pass


@dataclass(slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def ensure_not_stopped(stop_check: Callable[[], bool] | None) -> None:
    if stop_check is not None and stop_check():
        raise StopRequestedError("停止が要求されました。")


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> CommandResult:
    ensure_not_stopped(stop_check)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    terminated_for_stop = False
    timed_out = False
    try:
        while process.poll() is None:
            if stop_check is not None and stop_check():
                terminated_for_stop = True
                process.terminate()
                break
            if timeout_seconds is not None and (time.monotonic() - started_at) > timeout_seconds:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.2)

        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    finally:
        duration = time.monotonic() - started_at

    if terminated_for_stop:
        raise StopRequestedError("停止ボタンにより処理を中断しました。")
    if timed_out:
        raise RuntimeError("外部コマンドがタイムアウトしました。")

    return CommandResult(
        exit_code=int(process.returncode or 0),
        stdout=(stdout or "").strip(),
        stderr=(stderr or "").strip(),
        duration_seconds=duration,
    )
