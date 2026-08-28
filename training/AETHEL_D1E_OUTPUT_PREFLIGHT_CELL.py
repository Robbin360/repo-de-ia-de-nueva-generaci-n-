# CELDA D1E-PREFLIGHT — retirar sólo una salida residual vacía
from pathlib import Path

OUTPUT_ROOT = Path("/kaggle/working/aethel-d1e-router-entropy-strength-v3")

if not OUTPUT_ROOT.exists():
    print("D1E_OUTPUT_PREFLIGHT: no existe salida residual; puede ejecutarse CELDA 12")
else:
    entries = sorted(OUTPUT_ROOT.iterdir())
    print(f"D1E_OUTPUT_PREFLIGHT_ENTRIES: {len(entries)}")
    for entry in entries:
        print(f"- {entry}")
    if entries:
        raise RuntimeError(
            "La salida residual contiene archivos; no se borrará nada. "
            "Inspección detenida antes de D1E."
        )
    OUTPUT_ROOT.rmdir()
    print("D1E_OUTPUT_PREFLIGHT_EMPTY_REMOVED")
    print("No se borraron checkpoints, métricas ni archivos; el directorio estaba vacío.")

print("D1E_OUTPUT_PREFLIGHT_READY")
print("No se usó GPU, no se leyó corpus/holdout y no se ejecutó entrenamiento.")
