# Contrato experimental de arquitectura cognitiva Aethel v1

**Estado:** diseño de validación. No autoriza entrenamiento, GPU, publicación en Kaggle ni actualización de pesos.

## Regla de medición

Cada módulo nuevo debe compararse con una configuración de control que use el mismo tokenizador, Dataset, número de tokens, semilla, precisión, hardware y presupuesto de pasos. El resultado no puede ser una impresión cualitativa: debe incluir artefacto de checkpoint, manifiesto, métricas por idioma, trazas del router y una prueba de no regresión. Hasta tener una línea base real, los umbrales numéricos de calidad se dejan deliberadamente sin fijar; se establecerán a partir de la primera corrida reproducible, no mediante números inventados.

| Activo congelado | Uso permitido en experimentos autorizados | Uso prohibido |
|---|---|---|
| `aethel-knowledge-reasoning-bilingual-v1` | Entrenamiento en `train` y evaluación en `holdout` sólo tras autorización. | Tokenizar, ajustar, seleccionar hiperparámetros o alimentar replay con `holdout`. |
| Tokenizador BPE 32k | Codificar `train` y `holdout`; su hash debe registrarse. | Reentrenarlo usando `holdout`. |
| La Roca | Línea base y referencia de rollback. | Cambiarla durante una sesión o sin promoción validada. |
| El Líquido | Proponer memorias efímeras y un candidato LoRA aislado. | Actualizar directamente pesos de producción. |

## Métricas y artefactos mínimos

El evaluador existente `engine/evaluate_nextgen.py` calcula pérdida y perplejidad en JSONL retenido y las desglosa por idioma. Esa medición es necesaria, pero no basta para evaluar memoria, coste ni estabilidad.

| Dimensión | Medida primaria | Segmentación obligatoria | Evidencia guardada |
|---|---|---|---|
| Modelado lingüístico | Pérdida y perplejidad del holdout. | Inglés y español; cuando el volumen lo permita, por dominio. | JSON de evaluación con hash de checkpoint y tokenizador. |
| Retención | Diferencia de pérdida entre La Roca de referencia y candidato después de consolidación. | Inglés, español y dominios disponibles. | Matriz de regresión y conjunto de hashes evaluados. |
| Recuperación | Proporción de consultas con fuente recuperada, similitud y precisión de proveniencia. | Memoria episódica frente a semántica. | Trazas externas, no cadena de pensamiento privada. |
| Workspace | Número de propuestas, K ranuras usadas, fuente ganadora y entropía del selector. | Por tipo de entrada y idioma. | Telemetría por lote y ablación de fusión actual. |
| Cómputo | Latencia, tokens/s, VRAM máxima, energía si está disponible y pasos de refinamiento efectivos. | Control frente a variante, mismo hardware. | Log de entorno, semilla y profiler. |
| Seguridad de mutación | Igualdad de hashes de La Roca durante sesión; reversibilidad del candidato. | Antes/después de conversación y antes/después de sueño. | Manifiestos y prueba de rollback. |
| Contaminación | Intersección de hashes entre `train`, `holdout`, tokenizer input y replay. | Por idioma. | Informe offline que falle si hay una coincidencia. |

## Matriz de ablación

Cada fila es una ejecución futura independiente, no un cambio acumulativo. El orden evita atribuir una mejora a varios mecanismos a la vez.

| Experimento | Control | Variante única | Hipótesis falsable | Criterio de decisión |
|---|---|---|---|---|
| E0: tronco | Modelo NextGen sin memoria persistente ni refinamiento. | Ninguna. | Produce una línea base reproducible. | El run conserva artefactos y el evaluador genera pérdida para ambos idiomas. |
| E1: memoria recuperable | E0. | Episódica + semántica sólo en inferencia. | La recuperación aporta contexto sin degradar el holdout. | Comparar calidad y coste; rechazar si introduce fuga o regresión definida. |
| E2: Líquido efímero | E1. | Traza hebbiana con TTL y procedencia. | Mejora continuidad de sesión sin cambiar La Roca. | Hash de La Roca idéntico; trazas expiran y son revocables. |
| E3: sueño aislado | E1. | Replay estratificado + LoRA candidato. | La consolidación mejora o retiene sin olvidar. | Sólo considerar promoción si supera todas las particiones y rollback. |
| E4: workspace escaso | E3 o E1 según resultado. | K-ranuras con competición de fuentes. | Seleccionar pocas fuentes es más útil que fusionarlas siempre. | Ablación con mismo presupuesto y telemetría completa. |
| E5: cómputo adaptativo | Mejor variante previa. | Refinamiento condicionado por incertidumbre y presupuesto. | Más pasos se asignan a casos difíciles con valor neto. | Curva de coste frente a calidad, no sólo promedio de pérdida. |

## Protocolo de sueño propuesto

El ciclo de sueño tiene entrada y salida declaradas. No puede leer fuentes de Internet, memoria no autorizada ni el conjunto de evaluación. Su entrada está limitada a eventos líquidos con procedencia válida y a segmentos `train` permitidos.

| Paso | Entrada | Operación determinista/auditable | Salida |
|---|---|---|---|
| 1. Curación | Eventos líquidos en cuarentena. | Validar esquema, TTL, permiso, deduplicación y riesgo. | Lista elegible y lista rechazada con causa. |
| 2. Muestreo | Lista elegible + `train`. | Replay estratificado por idioma, dominio, novedad y saliencia con semilla guardada. | Índice de replay, no texto copiado al log. |
| 3. Adaptación | Índice de replay + La Roca congelada. | Ajustar solamente un LoRA candidato. | Checkpoint candidato y curva de entrenamiento. |
| 4. Evaluación | Candidato y referencia. | Ejecutar holdout bloqueado, regresión y controles de contaminación. | Informe comparativo. |
| 5. Promoción | Informe, hashes y política. | Aprobar o rechazar; generar rollback. | Nueva referencia o rechazo trazable. |

## Puertas de promoción

Las puertas son acumulativas. Un fallo no debe reintentarse alterando el conjunto retenido; debe producir un diagnóstico y un nuevo candidato claramente versionado.

| Puerta | Condición que debe verificarse antes de continuar |
|---|---|
| P0: integridad | El paquete, tokenizador y checkpoint coinciden con hashes declarados. |
| P1: aislamiento | No existe intersección entre `holdout` y cualquier entrada de adaptación, replay o tokenización. |
| P2: estabilidad | LaRoca permanece sin cambios durante inferencia, y el candidato se puede desactivar sin afectar el baseline. |
| P3: calidad | La variante satisface la regla de no regresión que se fijará tras E0, desglosada por idioma. |
| P4: eficiencia | El coste adicional se explica con métricas reales y se compara contra el control. |
| P5: trazabilidad | El manifiesto permite reproducir datos, semilla, código, configuración y resultados. |
| P6: revisión | La promoción requiere aprobación explícita; no se activa automáticamente por una métrica. |

## Resultado que se espera de la primera corrida autorizada

La primera corrida no busca competir con modelos de frontera. Su propósito es **calibrar**: medir la capacidad de un modelo pequeño con el corpus real, comprobar que el tokenizador funciona para inglés y español, recuperar un checkpoint persistible y obtener una línea base de pérdida/latencia/VRAM. Sólo después de E0 será técnicamente honesto fijar presupuestos, tamaños de modelo, umbrales de regresión y decidir si E1–E5 aportan valor.

> La capacidad de “mejorarse” se define aquí de forma limitada: generar candidatos de adaptación en una partición aislada, evaluarlos contra una referencia y promoverlos sólo si superan puertas reproducibles. No implica autonomía abierta, objetivos propios ni auto-modificación sin control humano.
