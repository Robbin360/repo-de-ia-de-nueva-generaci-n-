# Notas de auditoría del repositorio Aethel Meta

**Origen auditado:** `Robbin360/repo-de-ia-de-nueva-generaci-n-`, rama `main`, clon local en `/home/ubuntu/aethel-meta-original` el 26 de agosto de 2026.

## Principio de clasificación

La documentación Meta distingue entre código presente, contratos o diseños, y resultados de ejecución. Esta nota conserva esa frontera: una capacidad documentada no se tratará como entrenada, desplegada o validada en GPU sin el artefacto correspondiente.

## Capacidades recuperadas del núcleo cognitivo

| Capacidad | Evidencia original | Estado descrito por el repositorio Meta |
|---|---|---|
| La Roca | `ARCHITECTURE_COGNITIVE_OPERATING_MODEL.md`, líneas 45–49 | Referencia estable parcial; objetivo de manifiesto inmutable, promoción y rollback atómico aún pendiente. |
| El Líquido | Mismo documento, líneas 51–57 | Traza hebbiana y eventos versionados; falta persistencia en `state_dict`, aislamiento, revocación y promoción segura. |
| Curiosidad | Mismo documento, líneas 71–75 | Árbitro acotado de incertidumbre/novedad/riesgo/coste; no agencia autónoma ni acciones externas. |
| Ciclo de Sueño | Mismo documento, líneas 59–63 | Replay y consolidación previstos; falta pipeline completo de adaptador candidato, regresión y promoción. |
| Memoria | Mismo documento, líneas 23–24 y `server/aethelSpecs.ts`, líneas 69–74 | Trabajo (GRU), episódica y semántica persistente, consolidación/replay; faltan políticas completas de procedencia, privacidad y expiración. |
| Espacio de Trabajo Global | Mismo documento, líneas 65–69 | Fusión de tres vías existente; objetivo futuro de bus competitivo con K ranuras, fuentes, confianza y presupuesto. |
| Neuromodulación | Mismo documento, líneas 71–75 | Señales instrumentales para recursos; no motivación, voluntad ni objetivos autónomos. |
| Refinamiento adaptativo / ARC | `server/aethelSpecs.ts`, líneas 75–80 | Experimental, apagado por defecto; debe probar pérdida/router/latencia/VRAM frente a baseline. |
| Razonamiento observable | `server/aethelSpecs.ts`, líneas 75–79 | Protocolo recuperación → integración → refinamiento presupuestado → predicción; expone evidencia, no cadena de pensamiento privada. |

## Eficiencia y escalado recuperados

| Mecanismo | Evidencia original | Estado descrito |
|---|---|---|
| Sparse MoE top-2 | `EFFICIENCY_AND_REASONING_ROADMAP.md`, líneas 25–35 | Implementado con telemetría; exige medir balance, capacidad y tokens descartados. |
| RoPE, GQA y KV-cache | Mismo documento, líneas 29–32 y 49–53 | Implementados; KV-cache tiene microbenchmark CPU, no evidencia GPU/VRAM. |
| Embeddings atados | Mismo documento, línea 31 | Implementado. |
| LoRA | Mismo documento, líneas 32–35 | Opcional; falta comparación medida frente a ajuste completo. |
| Triton/CUDA | `AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md`, líneas 71–75 | Parcial; kernels y puente existen, pero aceptación CUDA de todas las rutas sigue siendo una puerta de evidencia. |
| MTP, decodificación especulativa y SSM selectivos | `EFFICIENCY_AND_REASONING_ROADMAP.md`, líneas 55–66 | Hipótesis de investigación futura, no capacidades activas. |
| Presets de escala | `server/aethelSpecs.ts`, líneas 33–38 | Pilot-100m, research-300m, scale-1b y variante ARC; el escalado está condicionado a GPU y validación distribuida. |

## Runtime y producto recuperados

| Área | Evidencia original | Estado descrito |
|---|---|---|
| Runtime Rust de memoria | `runtime/aethel-memory-rust/src/lib.rs`, `AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md`, líneas 77–81 | Código para JSONL, snapshots, recuperación trazable, consolidación y Unix socket; no servicio 24/7 desplegado. |
| Runtime Mojo | `runtime/mojo/INFERENCE_CONTRACT.md`, matriz de runtime, líneas 83–87 | Contrato de prefill/decode/KV-cache; no implementación validada. |
| Plataforma TS/React/Node | Matriz de runtime, líneas 41–44 y 59–64 | Dashboard y gateway implementados; no son inferencia Aethel propia. |
| C++/CUDA C++ y C# | Matriz de runtime, líneas 89–91 | Extensiones futuras condicionadas; sin código auditado. |
| Benchmarking | `server/aethelSpecs.ts`, líneas 90–96 | MMLU, GSM8K y HumanEval requieren predicciones reales; no deben reportarse sin ejecución. |

## Implicación para la corrida directa

La primera corrida no debe sustituir la auditoría de Meta ni prometer toda la visión. Debe producir evidencia mínima: checkpoint recuperable, pérdidas finitas, telemetría MoE, memoria/replay observados, throughput y estado de los artefactos. Después se incorporarán contratos específicos para promoción de La Roca, ciclo de Sueño offline, workspace competitivo, ARC y runtime de inferencia.

## Fuentes internas auditadas

1. `/home/ubuntu/aethel-meta-original/ARCHITECTURE_COGNITIVE_OPERATING_MODEL.md`
2. `/home/ubuntu/aethel-meta-original/EFFICIENCY_AND_REASONING_ROADMAP.md`
3. `/home/ubuntu/aethel-meta-original/AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md`
4. `/home/ubuntu/aethel-meta-original/POLYGLOT_RUNTIME_ARCHITECTURE.md`
5. `/home/ubuntu/aethel-meta-original/RESEARCH_AETHEL_NEXTGEN.md`
6. `/home/ubuntu/aethel-meta-original/server/aethelSpecs.ts`
