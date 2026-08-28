from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    launcher = (ROOT / "run_kaggle_edge_long_session_v1.sh").read_text(encoding="utf-8")
    required = (
        "EDGE_DATA_ROOT",
        "SESSION_TARGET_STEP",
        "SCHEDULE_TOTAL_STEPS",
        "--schedule-total-steps",
        "--data-manifest",
        "--save-every 4000",
        "--metrics-every 256",
        "--console-every 4000",
        "--resume-checkpoint",
        "resume_args=(--resume-checkpoint",
        "package_edge_session.py",
        "sync",
        "train-*.jsonl.gz",
        "train-*.jsonl",
        "mezcla shards comprimidos y descomprimidos",
    )
    assert all(item in launcher for item in required)
    builder = (ROOT / "run_kaggle_build_edge_corpus_v1.sh").read_text(encoding="utf-8")
    assert "--approved-data-plan" in builder and "--allow-network" in builder
    assert "BASE_DATA_ROOT" in builder
    assert 'cp "${BASE_DATA_ROOT}/tokenizer.json"' in builder
    assert 'cmp --silent "${BASE_DATA_ROOT}/tokenizer.json"' in builder
    assert "train_aethel_gpu.py" not in builder
    packager = (ROOT / "package_edge_session.py").read_text(encoding="utf-8")
    assert "SAVE_KAGGLE_VERSION_NOW.txt" in packager
    assert '"checkpoint_uploaded": False' in packager
    assert "prepared_manifest.json" in packager
    print("AETHEL_EDGE_LONG_SESSION_CONTRACT_OK")


if __name__ == "__main__":
    main()
