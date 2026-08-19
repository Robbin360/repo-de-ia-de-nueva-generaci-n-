from __future__ import annotations

import copy
import hashlib
import json

from sleep_preflight import run_sleep_preflight
from sleep_replay_admission import build_quarantined_replay_manifest


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rehash_replay(manifest: dict) -> None:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


ROCK_HASH = digest("rock")


def event() -> dict:
    return {
        "event_id": "physics-es",
        "source": "curated-source-es",
        "source_sha256": digest("curated-source-es"),
        "language": "es",
        "domain": "science",
        "priority": 0.7,
        "ttl_observations": 4,
        "eligible_for_sleep": True,
        "curation_status": "curated",
        "holdout_member": False,
    }


def approval() -> dict:
    return {
        "approval_id": "approval-physics-es",
        "event_id": "physics-es",
        "source_sha256": digest("curated-source-es"),
        "status": "approved",
        "approved_by": "human-review-v1",
    }


def manifests() -> tuple[dict, dict, dict, dict]:
    rock = {"kind": "aethel_rock_reference", "lora_active": False, "rock_state_sha256": ROCK_HASH}
    candidate = {
        "kind": "aethel_sleep_candidate",
        "candidate_id": "sleep-candidate-0001",
        "parent_rock_state_sha256": ROCK_HASH,
        "candidate_base_state_sha256": ROCK_HASH,
        "training_started": False,
        "optimizer_created": False,
        "eligible_for_promotion": False,
        "holdout_access_enabled": False,
        "external_action_enabled": False,
    }
    replay = build_quarantined_replay_manifest([event()], [approval()], [], ROCK_HASH)
    dataset = {
        "dataset_id": "aethel-knowledge-reasoning-bilingual-v1",
        "offline_training_ready": True,
        "holdout_excluded_from_tokenizer": True,
        "tokenizer": {"derived_from": "train split only", "sha256": digest("tokenizer")},
        "corpus_files": [
            {"path": "corpus/train-en-00000.jsonl.gz", "sha256": digest("train")},
            {"path": "corpus/holdout-en-00000.jsonl.gz", "sha256": digest("holdout")},
        ],
        "counts": {"train:en": 1, "train:es": 1, "holdout:en": 1, "holdout:es": 1},
    }
    return rock, candidate, replay, dataset


def fails(action, expected: str) -> None:
    try:
        action()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("Se esperaba rechazo de preflight")


def main() -> None:
    rock, candidate, replay, dataset = manifests()
    report = run_sleep_preflight(rock, candidate, replay, dataset)
    assert report["preflight_status"] == "quarantined_preflight_pass"
    assert report["eligible_for_training"] is False
    assert report["requires_runtime_authorization"] is True
    assert len(report["report_sha256"]) == 64

    wrong_candidate = copy.deepcopy(candidate)
    wrong_candidate["parent_rock_state_sha256"] = digest("wrong")
    fails(lambda: run_sleep_preflight(rock, wrong_candidate, replay, dataset), "no pertenece")
    wrong_replay = copy.deepcopy(replay)
    wrong_replay["holdout_access_enabled"] = True
    rehash_replay(wrong_replay)
    fails(lambda: run_sleep_preflight(rock, candidate, wrong_replay, dataset), "holdout_access_enabled")
    fails(lambda: run_sleep_preflight(rock, candidate, replay, dataset, [event()["source_sha256"]]), "colisiona")
    wrong_dataset = copy.deepcopy(dataset)
    wrong_dataset["tokenizer"]["derived_from"] = "mixed split"
    fails(lambda: run_sleep_preflight(rock, candidate, replay, wrong_dataset), "tokenizador")
    print("sleep_preflight OK")


if __name__ == "__main__":
    main()
