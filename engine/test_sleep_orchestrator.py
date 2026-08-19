from __future__ import annotations

import hashlib

from sleep_orchestrator import apply_verified_preflight
from sleep_preflight import run_sleep_preflight
from sleep_replay_admission import build_quarantined_replay_manifest
from sleep_state_machine import PREFLIGHT_PASS, SleepLifecycle


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def fixture() -> tuple[SleepLifecycle, dict]:
    rock_hash = digest("rock")
    rock = {"kind": "aethel_rock_reference", "lora_active": False, "rock_state_sha256": rock_hash}
    candidate = {
        "kind": "aethel_sleep_candidate", "candidate_id": "sleep-candidate-0001",
        "parent_rock_state_sha256": rock_hash, "candidate_base_state_sha256": rock_hash,
        "training_started": False, "optimizer_created": False, "eligible_for_promotion": False,
        "holdout_access_enabled": False, "external_action_enabled": False,
    }
    source_hash = digest("source")
    event = {
        "event_id": "event-1", "source": "local", "source_sha256": source_hash,
        "language": "es", "domain": "science", "priority": 0.6, "ttl_observations": 2,
        "eligible_for_sleep": True, "curation_status": "curated", "holdout_member": False,
    }
    approval = {"approval_id": "review-1", "event_id": "event-1", "source_sha256": source_hash,
                "status": "approved", "approved_by": "human-review-v1"}
    replay = build_quarantined_replay_manifest([event], [approval], [], rock_hash)
    dataset = {
        "dataset_id": "dataset-v1", "offline_training_ready": True, "holdout_excluded_from_tokenizer": True,
        "tokenizer": {"derived_from": "train split only", "sha256": digest("tokenizer")},
        "corpus_files": [{"path": "corpus/train-en-00000.jsonl.gz", "sha256": digest("train")},
                         {"path": "corpus/holdout-en-00000.jsonl.gz", "sha256": digest("holdout")}],
        "counts": {"train:en": 1, "train:es": 1, "holdout:en": 1, "holdout:es": 1},
    }
    return SleepLifecycle("sleep-candidate-0001", rock_hash), run_sleep_preflight(rock, candidate, replay, dataset)


def fails(action, expected: str) -> None:
    try:
        action()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("Se esperaba rechazo")


def main() -> None:
    lifecycle, report = fixture()
    event = apply_verified_preflight(lifecycle, report)
    assert event["to_state"] == PREFLIGHT_PASS
    assert lifecycle.verify_ledger()["state"] == PREFLIGHT_PASS
    fails(lambda: apply_verified_preflight(lifecycle, report), "cuarentena")

    other, altered = fixture()
    altered["eligible_for_training"] = True
    fails(lambda: apply_verified_preflight(other, altered), "hash del reporte")
    other, mismatch = fixture()
    mismatch["candidate_id"] = "other-candidate"
    mismatch["report_sha256"] = "0" * 64
    fails(lambda: apply_verified_preflight(other, mismatch), "hash del reporte")
    print("sleep_orchestrator OK")


if __name__ == "__main__":
    main()
