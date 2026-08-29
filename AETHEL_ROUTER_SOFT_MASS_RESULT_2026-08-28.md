# Experimento CPU — masa probabilística suave del router

Se comparó un router top-2 inicialmente concentrado en dos expertos con y sin una señal diferenciable que penaliza la masa probabilística media por experto. El experimento es un modelo de juguete CPU, determinista, de 256 tokens, 8 expertos y 160 pasos; no modifica el núcleo Aethel.

| Métrica | Sin señal suave | Con masa suave |
|---|---:|---:|
| Desequilibrio top-2 | 0,7500 | 0,7500 |
| Desequilibrio de masa probabilística | 0,5932 | 0,5778 |
| Expertos usados por top-2 | 2 | 2 |

## Conclusión

La señal suave redujo ligeramente el desequilibrio de probabilidad, pero **no cambió la selección top-2** en este escenario: los mismos dos expertos siguieron recibiendo todos los tokens. Por tanto, la masa suave aislada no resuelve el atractor de dos expertos. Es una señal complementaria posible, pero requiere combinarse con ruido/jitter controlado, capacidad diferenciable, penalización de carga efectiva o una política explícita de exploración. No se habilita en producción a partir de este resultado.

El resultado no permite inferir calidad lingüística, rendimiento GPU, VRAM ni estabilidad en entrenamiento real.
