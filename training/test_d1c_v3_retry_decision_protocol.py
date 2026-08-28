"""Contrato estático del protocolo local D1C V3-R1; no accede a datos ni artefactos."""

from pathlib import Path


PROTOCOL = Path(__file__).with_name("AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md")


def require(text: str, fragment: str) -> None:
    assert fragment in text, f"Falta la restricción: {fragment}"


def main() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for fragment in (
        "D1C_V3_R1_PREPARATION_ONLY",
        "sin pesos E0, sin checkpoint D1C V1",
        "holdout EN/ES permanece sellado",
        "768 pasos, seed 17",
        "router_aux_loss_weight=0.05",
        "Fallback PyTorch experimental",
        "G0",
        "G6",
        "autorización literal e inmediata",
        "No hay retry preparado para ejecutar.",
    ):
        require(text, fragment)
    print("D1C_V3_R1_DECISION_PROTOCOL_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
