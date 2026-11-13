from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from h3_training_compat import install


HOME = Path(os.environ.get("TRAINER_HOME", Path.home() / "modelscope-trainer-data")).expanduser().resolve()
TRAIN_SCRIPT = HOME / "repos" / "DiffSynth-Studio" / "examples" / "minimax_h3_image" / "model_training" / "train.py"


def main() -> None:
    if not TRAIN_SCRIPT.is_file():
        raise SystemExit(f"找不到 H3 训练脚本：{TRAIN_SCRIPT}")
    install()
    sys.argv = [str(TRAIN_SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
