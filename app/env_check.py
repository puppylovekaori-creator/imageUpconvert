from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RuntimeStatus:
    python_executable: str
    python_version: str
    platform: str
    torch_available: bool
    torch_version: str
    cuda_available: bool
    cuda_device_name: str
    cuda_device_count: int
    message: str


def get_runtime_status() -> RuntimeStatus:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    base = RuntimeStatus(
        python_executable=sys.executable,
        python_version=python_version,
        platform=platform.platform(),
        torch_available=False,
        torch_version="",
        cuda_available=False,
        cuda_device_name="",
        cuda_device_count=0,
        message="PyTorch がインストールされていません。",
    )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local runtime
        base.message = f"PyTorch の読み込みに失敗しました: {exc}"
        return base

    cuda_available = torch.cuda.is_available()
    cuda_device_name = ""
    cuda_device_count = 0
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)

    message = (
        f"PyTorch {torch.__version__} / CUDA {'利用可' if cuda_available else '利用不可'}"
    )
    if cuda_device_name:
        message += f" / {cuda_device_name}"

    return RuntimeStatus(
        python_executable=sys.executable,
        python_version=python_version,
        platform=platform.platform(),
        torch_available=True,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cuda_device_count=cuda_device_count,
        message=message,
    )


def format_runtime_status(status: RuntimeStatus) -> str:
    device_text = "GPU 利用可" if status.cuda_available else "CPU のみ"
    if status.cuda_device_name:
        device_text += f" ({status.cuda_device_name})"
    return (
        f"Python {status.python_version} | "
        f"PyTorch {status.torch_version or '未導入'} | {device_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="標準テキストではなく JSON を出力します。")
    args = parser.parse_args()

    status = get_runtime_status()
    if args.json:
        print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
    else:
        print(status.message)
        print(f"Python 実行ファイル: {status.python_executable}")
        print(f"プラットフォーム: {status.platform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
