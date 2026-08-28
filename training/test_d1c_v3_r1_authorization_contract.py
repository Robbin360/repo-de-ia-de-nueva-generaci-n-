"""Static validation for the local D1C V3-R1 authorization contract."""

import json
from pathlib import Path


CONTRACT = Path(__file__).with_name("aethel_d1c_v3_r1_authorization_contract.json")


def main() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PREPARATION_ONLY_ALL_GATES_CLOSED"
    assert payload["experiment_id"] == "D1C-V3-R1"
    assert payload["source_release"] == "d1c-v3-retry-cell-train-only"

    experiment = payload["experiment_contract"]
    assert experiment["fresh_initialization_required"] is True
    assert experiment["resume_forbidden"] is True
    assert experiment["e0_weights_forbidden"] is True
    assert experiment["d1c_v1_artifacts_forbidden"] is True
    assert experiment["train_only_required"] is True
    assert experiment["holdout_en_es_sealed"] is True
    assert experiment["steps"] == 768
    assert experiment["seed"] == 17
    assert experiment["router_bias_step"] == 0.05
    assert experiment["router_aux_loss_weight"] == 0.05
    assert experiment["runtime"] == "pytorch-fallback-experimental"
    assert experiment["triton_strict_blocked"] is True
    assert experiment["fresh_work_and_output_roots_required"] is True

    authorizations = payload["authorizations"]
    assert authorizations
    assert all(value is False for value in authorizations.values())
    assert payload["current_effect"] == "No execution path is permitted. This contract is local documentation only."
    print("D1C_V3_R1_AUTHORIZATION_CONTRACT_STATIC_OK")


if __name__ == "__main__":
    main()
