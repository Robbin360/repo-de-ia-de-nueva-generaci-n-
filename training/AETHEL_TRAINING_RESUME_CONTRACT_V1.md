# Contrato de reanudación de entrenamiento Aethel V2

> **Propósito.** Permitir que una sesión posterior continúe exactamente un tramo de entrenamiento previamente planificado, sin convertir un checkpoint de pesos en una afirmación de calidad del modelo.

El checkpoint reanudable debe almacenar pesos del modelo, el `reference_state` de los parámetros al inicio de la fase, estado de AdamW, `GradScaler`, contador de paso, configuración íntegra, hash del tokenizador, estado RNG de Python/PyTorch/CUDA, memoria mutable de Aethel y el contrato de datos/entrenamiento. `reference_state` conserva el objetivo de regularización de replay: sin él, la sesión reanudada cambia ese objetivo y no es numéricamente equivalente a una corrida continua. Los checkpoints portátiles `step_*.pt` siguen siendo sólo contingencias de pesos y no se aceptan para reanudación fiel.

Una fase larga se define inicialmente con `--schedule-total-steps`; cada sesión usa `--max-steps` como el límite global al que debe detenerse en ese día. Por ejemplo, una fase de 100.000 pasos puede ejecutar hasta 5.000 en la primera sesión y reanudar hasta 10.000 en la segunda, manteniendo el mismo scheduler. `schedule_total_steps` no puede cambiar al reanudar; `max_steps` sí puede aumentar, siempre por encima del paso guardado.

| Elemento | Verificación antes de cargar | Motivo |
|---|---|---|
| Topología y tokenizador | Igualdad exacta | Evita cargar tensores o IDs incompatibles. |
| Datos | Hash de manifiesto y distribución de shards iguales | Evita continuar sobre una fuente distinta sin declararlo. |
| Perfil | Estrategia, mundo, precisión, lote, acumulación, scheduler y semillas iguales | Conserva la trayectoria del optimizador. |
| Estado mutable | Memoria, traza líquida, replay y curiosidad dentro de capacidad | Preserva el estado que `state_dict` omite deliberadamente. |
| RNG | Python, CPU y CUDA | Evita cambiar la secuencia aleatoria tras reanudar. |

Una nueva fase de entrenamiento puede cambiar datos, horizonte o perfil, pero debe recibir un nuevo identificador de fase y una evaluación de frontera. No debe presentarse como continuación bit-a-bit del tramo anterior.
