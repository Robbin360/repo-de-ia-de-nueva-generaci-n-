from __future__ import annotations

import ast
from pathlib import Path


CELL_PATH = Path(__file__).with_name("AETHEL_D1B_ROUTER_BIAS_BLOCKED_CELL.py")


def main() -> None:
    source = CELL_PATH.read_text(encoding="utf-8")
    ast.parse(source)

    required = (
        'D1B_EXECUTION_ENABLED = False',
        'EXPECTED_RELEASE = "d1b-v1-router-bias-step-001-train-only"',
        "D1B_CELL_PREPARED_NOT_EXECUTED",
        "D1B_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT",
        "run_kaggle_d1b_router_bias_diagnostic.sh",
    )
    for value in required:
        assert value in source, value

    forbidden = (
        "import subprocess",
        "import shutil",
        "import torch",
        "AETHEL_DATA_DIR",
        "AETHEL_D1B_RUN_AUTHORIZED",
        "AETHEL_D1B_GPU_AUTHORIZED",
        "copytree(",
        "subprocess.run(",
        "os.environ",
    )
    for value in forbidden:
        assert value not in source, value

    print("D1B_BLOCKED_CELL_LOCAL_TESTS_PASSED")


if __name__ == "__main__":
    main()
