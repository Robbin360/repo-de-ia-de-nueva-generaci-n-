from __future__ import annotations

import hashlib

from sleep_state_machine import (
    AUTHORIZED,
    EVALUATED,
    PREFLIGHT_PASS,
    PROMOTABLE,
    PROMOTED,
    QUARANTINED,
    REJECTED,
    ROLLED_BACK,
    RUNNING,
    SleepLifecycle,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def fails(action, error_type: type[Exception], expected: str) -> None:
    try:
        action()
    except error_type as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("Se esperaba rechazo")


def main() -> None:
    lifecycle = SleepLifecycle("sleep-candidate-0001", digest("rock"))
    assert lifecycle.state == QUARANTINED
    fails(lambda: lifecycle.transition(RUNNING, "runtime-executor", digest("skip")), ValueError, "prohibida")
    fails(lambda: lifecycle.transition(PREFLIGHT_PASS, "runtime-executor", digest("preflight")), PermissionError, "preflight-verifier")

    lifecycle.transition(PREFLIGHT_PASS, "preflight-verifier", digest("preflight"))
    lifecycle.transition(AUTHORIZED, "human-execution-approver", digest("execution-approval"))
    lifecycle.transition(RUNNING, "runtime-executor", digest("run"))
    lifecycle.transition(EVALUATED, "evaluation-runner", digest("evaluation"))
    lifecycle.transition(PROMOTABLE, "evaluation-reviewer", digest("review"))
    lifecycle.transition(PROMOTED, "human-promotion-approver", digest("promotion"))
    lifecycle.transition(ROLLED_BACK, "rollback-operator", digest("rollback"))
    assert lifecycle.verify_ledger()["ledger_valid"] is True

    rejected = SleepLifecycle("sleep-candidate-0002", digest("rock-2"))
    rejected.transition(REJECTED, "system-or-reviewer", digest("reject"))
    rejected.transition(ROLLED_BACK, "rollback-operator", digest("rollback-2"))
    assert rejected.verify_ledger()["state"] == ROLLED_BACK

    tampered = SleepLifecycle("sleep-candidate-0003", digest("rock-3"))
    tampered.ledger[0]["authority"] = "altered"
    fails(tampered.verify_ledger, ValueError, "alterado")
    print("sleep_state_machine OK")


if __name__ == "__main__":
    main()
