# Currículo de datos y escalado de Aethel

## Principio rector

Aethel no debe intentar cubrir todos los dominios desde el primer token. El entrenamiento debe progresar de texto limpio y lenguaje general a dominios técnicos y razonamiento, y cada cambio de etapa requiere conservar o mejorar un conjunto de evaluación congelado.

| Etapa | Composición propuesta | Objetivo verificable | Condición para avanzar |
|---|---|---|---|
| 0. Piloto | Corpus local y texto abierto de alta calidad; 10M–50M tokens | Validar infraestructura, BPE, routing y persistencia | Pérdida de validación descendente, exportación verificable y `router_health.healthy` estable |
| 1. Fundamentos | FineWeb-Edu u otra fuente aprobada por licencia, español/multilingüe equilibrado | Modelar lenguaje general sin contaminación de evaluación | Validación por idioma, deduplicación y trazabilidad por fuente |
| 2. Técnica | Documentación, libros con licencia compatible, código permitido y matemáticas | Mejorar precisión técnica y comprensión de código | Métricas propias de código/matemáticas y ausencia de regresión general |
| 3. Razonamiento | Datos de razonamiento con respuestas verificables y programas ejecutables | Mejorar verificación y planificación | Evaluación separada de GSM8K/HumanEval sin datos de prueba en entrenamiento |
| 4. Continuo | Flujos recientes aprobados, replay estratificado y holdout congelado | Adaptar sin olvido catastrófico | Delta de retención dentro del umbral definido y auditoría de cambios de El Líquido |

## Receta de escalado

1. Ajustar primero una familia piloto de **100M–300M parámetros activos** y medir rendimiento, balance MoE y tokens/s en la GPU disponible.
2. Fijar la relación entre parámetros activos y tokens mediante pilotos; la evidencia de Chinchilla indica que escalar datos junto con el modelo importa tanto como aumentar parámetros [1].
3. Escalar los expertos totales sin aumentar proporcionalmente los expertos activos por token; mantener un presupuesto de latencia, VRAM y saturación de router.
4. Usar BF16, activación checkpointing cuando sea necesario, DDP/FSDP según número de GPU y checkpoints atómicos frecuentes.
5. No continuar a una escala mayor sin evaluación de pérdida, retención, salud MoE, seguridad y coste por millón de tokens.

## Hitos de capacidad — no promesas

| Escala aproximada | Qué se puede evaluar de forma honesta | Qué no debe afirmarse |
|---|---|---|
| 100M–300M | Coherencia local, modelado básico de lenguaje, estabilidad MoE y retención en tareas pequeñas | Capacidad frontier, autonomía general o cognición humana |
| 1B–3B con corpus suficiente | Mejor razonamiento de patrones, código breve y especialización de expertos medible | Razonamiento fiable en dominios críticos o aprendizaje humano general |
| Mayor escala con entrenamiento/evaluación rigurosos | Competencia relativa en benchmarks reproducibles y eficiencia por token | Consciencia, comprensión humana o superación garantizada de modelos frontera |

## Referencia

[1] Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*: https://arxiv.org/abs/2203.15556

