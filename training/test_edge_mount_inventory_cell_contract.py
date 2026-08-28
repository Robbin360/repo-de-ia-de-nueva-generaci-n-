from pathlib import Path


ROOT = Path(__file__).resolve().parent
GUIDE = ROOT / "AETHEL_EDGE_MOUNT_INVENTORY_CELL_V1.md"


def main() -> None:
    contents = GUIDE.read_text(encoding="utf-8")
    required = (
        "# CELDA 1",
        "rglob(\"prepared_manifest.json\")",
        "path.stat().st_size",
        "CELDA 1 — INVENTARIO_EDGE_MONTAJE_SEGURO",
        "no se usó GPU",
        "no se leyeron ejemplos",
        "no se entrenó",
    )
    for expected in required:
        assert expected in contents, expected
    forbidden = (
        "torch",
        "subprocess",
        "gzip.open",
        "read_text(encoding=\"utf-8\")",
        "write_text",
        "mkdir",
        "unlink",
        "rmtree",
    )
    for prohibited in forbidden:
        assert prohibited not in contents, prohibited
    print("AETHEL_EDGE_MOUNT_INVENTORY_CELL_CONTRACT_OK")


if __name__ == "__main__":
    main()
