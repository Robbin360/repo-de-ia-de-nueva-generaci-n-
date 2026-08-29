# Aethel — baseline CPU de cómputo adaptativo

Se ejecutó `engine/compare_arc_baseline.py` con el mismo corpus, batches y semilla para comparar una ruta base contra la ruta ARC (`steps=8`, `batch_size=1`, `seq_len=16`, 128 tokens).

| Métrica | Baseline | ARC | Diferencia ARC − baseline |
|---|---:|---:|---:|
| Parámetros | 167.428 | 192.581 | +25.153 |
| Pérdida media | 5,4572 | 5,4132 | −0,0440 |
| Pérdida final | 5,3373 | 5,3113 | −0,0260 |
| Desequilibrio medio del router | 0,1953 | 0,5000 | +0,3047 |
| Tokens/s CPU | 1.964,95 | 1.655,80 | −309,14 |
| RSS pico | 357.404.672 B | 360.255.488 B | +2.850.816 B |

## Interpretación

El experimento es reproducible y emparejado, pero corto y ejecutado en CPU. En esta muestra, ARC baja ligeramente la pérdida, pero añade parámetros y reduce el throughput CPU aproximadamente un 15,7%; además, empeora la métrica de desequilibrio del router. Por tanto, **no demuestra ultra-eficiencia** y no justifica activar ARC por defecto. Sirve como baseline y como señal para revisar la política de selección adaptativa antes de ampliar escala.

No se infieren VRAM, coste por token, calidad bilingüe, razonamiento ni rendimiento GPU a partir de esta prueba.
