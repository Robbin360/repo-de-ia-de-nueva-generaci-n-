from select_precision import select_precision


def test_precision_selection() -> None:
    assert select_precision(6) == "fp16"  # P100
    assert select_precision(7) == "fp16"  # T4/V100
    assert select_precision(8) == "bf16"  # A100/L4
    assert select_precision(9, "fp32") == "fp32"


if __name__ == "__main__":
    test_precision_selection()
    print("OK")
