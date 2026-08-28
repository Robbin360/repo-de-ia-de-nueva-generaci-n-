"""Contrato estático del perfil de release D1C V3-R1; no ejecuta el lanzador."""

from pathlib import Path


LAUNCHER = Path(__file__).with_name("run_kaggle_d1c_router_aux_loss_diagnostic.sh")


def main() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE="d1c-v1-router-aux-loss-005-train-only"' in text
    assert 'EXPECTED_RELEASE="${AETHEL_D1C_EXPECTED_RELEASE:-$EXPECTED_RELEASE}"' in text
    assert 'V3_R1_RELEASE="d1c-v4-v3-r1-launcher-profile-train-only"' in text
    assert '"d1c-v1-router-aux-loss-005-train-only")' in text
    assert '"$V3_R1_RELEASE")' in text
    assert "AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED" in text
    assert "perfil V3-R1 requiere autorización separada de release" in text
    assert "perfil de release no permitido" in text
    assert "--router-aux-loss-weight 0.05" in text
    assert "--diagnostic-id D1C" in text
    assert "AETHEL_RESUME_CHECKPOINT" in text
    assert "evaluate_nextgen.py" not in text
    assert "inspect_checkpoint.py" not in text
    assert "--resume" not in text
    print("D1C_V3_R1_RELEASE_PROFILE_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
