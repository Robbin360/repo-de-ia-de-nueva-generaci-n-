"""Verifica que la unidad no declare exposición pública ni rutas de escritura inseguras."""
from pathlib import Path


def main() -> None:
    unit = (Path(__file__).parent / "aethel-memory.service").read_text(encoding="utf-8")
    required = ("User=aethel", "--socket /run/aethel-memory/memory.sock", "Restart=always", "NoNewPrivileges=true", "ProtectSystem=strict", "UMask=0077")
    assert all(value in unit for value in required)
    assert "--port" not in unit and "ListenStream=" not in unit
    print("OK: plantilla systemd Rust endurecida")


if __name__ == "__main__":
    main()
