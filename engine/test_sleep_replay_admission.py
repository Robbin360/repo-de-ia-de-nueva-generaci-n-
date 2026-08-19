from __future__ import annotations

import hashlib

from sleep_replay_admission import build_quarantined_replay_manifest, review_event


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def event(event_id: str, source: str, priority: float, **changes: object) -> dict:
    record = {
        "event_id": event_id,
        "source": source,
        "source_sha256": digest(source),
        "language": "es" if event_id.endswith("es") else "en",
        "domain": "science",
        "priority": priority,
        "ttl_observations": 4,
        "eligible_for_sleep": True,
        "curation_status": "curated",
        "holdout_member": False,
        "content": "Este campo nunca debe aparecer en el manifiesto.",
    }
    record.update(changes)
    return record


def approval(record: dict, **changes: object) -> dict:
    result = {
        "approval_id": f"approval-{record['event_id']}",
        "event_id": record["event_id"],
        "source_sha256": record["source_sha256"],
        "status": "approved",
        "approved_by": "human-review-v1",
    }
    result.update(changes)
    return result


def fails(action, expected: str) -> None:
    try:
        action()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("Se esperaba rechazo de admisión")


def main() -> None:
    rock_hash = digest("rock")
    first = event("physics-es", "curated-source-es", 0.70)
    second = event("math-en", "curated-source-en", 0.95)
    manifest = build_quarantined_replay_manifest([first, second], [approval(first), approval(second)], [], rock_hash)
    assert manifest["record_count"] == 2
    assert [record["event_id"] for record in manifest["records"]] == ["math-en", "physics-es"]
    assert all("content" not in record for record in manifest["records"])
    assert manifest["eligible_for_training"] is False
    assert manifest["holdout_access_enabled"] is False
    assert manifest["approval_required_before_training"] is True
    assert len(manifest["manifest_sha256"]) == 64

    fails(lambda: review_event(event("raw-en", "raw-source", 0.5, curation_status="pending"), [], {}), "curado")
    fails(lambda: review_event(event("expired-en", "expired-source", 0.5, ttl_observations=0), [], {"expired-en": approval(event("expired-en", "expired-source", 0.5, ttl_observations=0))}), "TTL")
    holdout_event = event("holdout-en", "holdout-source", 0.5)
    fails(lambda: review_event(holdout_event, [holdout_event["source_sha256"]], {"holdout-en": approval(holdout_event)}), "holdout")
    fails(lambda: review_event(first, [], {}), "aprobación independiente")
    fails(lambda: build_quarantined_replay_manifest([first, dict(first)], [approval(first)], [], rock_hash), "duplicado")
    fails(lambda: build_quarantined_replay_manifest([first], [approval(first, source_sha256=digest("other"))], [], rock_hash), "no coincide")
    print("sleep_replay_admission OK")


if __name__ == "__main__":
    main()
