from pathlib import Path


SPEC = Path(__file__).resolve().parents[1] / "AETHEL_BASE_CAPABILITY_SPEC.md"


def main() -> None:
    text = SPEC.read_text(encoding="utf-8")
    required_any = {
        "100M": ("100M",),
        "EN/ES": ("EN/ES",),
        "razonamiento": ("razonamiento",),
        "matemáticas": ("matem",),
        "adaptador": ("adaptador",),
        "retención/no olvido": ("retención", "no olvido"),
        "reversión": ("rollback", "revertir", "descartar"),
    }
    lowered = text.lower()
    missing = [
        label for label, alternatives in required_any.items()
        if not any(term.lower() in lowered for term in alternatives)
    ]
    if missing:
        raise AssertionError(f"Faltan requisitos en la especificación: {missing}")
    forbidden_claims = ("modelo ya demostrado", "AGI demostrada", "IQ 300 demostrado")
    present = [claim for claim in forbidden_claims if claim.lower() in text.lower()]
    if present:
        raise AssertionError(f"La especificación contiene afirmaciones no demostradas: {present}")
    print("BASE_CAPABILITY_SPEC_CONTRACT_OK")


if __name__ == "__main__":
    main()
