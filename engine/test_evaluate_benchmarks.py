"""Pruebas unitarias deterministas del agregador de benchmarks."""
from evaluate_benchmarks import evaluate


def main() -> None:
    reference = {
        "m1": {"task": "mmlu", "answer": "B"},
        "g1": {"task": "gsm8k", "answer": "42"},
        "h1": {"task": "humaneval"},
    }
    predictions = {
        "m1": {"answer": "b"},
        "g1": {"answer": "La respuesta es #### 42"},
        "h1": {"pass": True},
    }
    report = evaluate(reference, predictions)
    assert all(item["accuracy"] == 1.0 for item in report["scores"].values())
    print('{"benchmark_aggregator_verified":true}')


if __name__ == "__main__":
    main()

