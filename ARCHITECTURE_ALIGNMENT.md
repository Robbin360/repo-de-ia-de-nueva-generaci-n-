# Alineación de Aethel NextGen con la arquitectura documentada

## Criterio

Aethel NextGen no sustituye la arquitectura del repositorio por un Transformer genérico. Conserva los mecanismos documentados y los conecta en un flujo entrenable, con estados observables y sin datos sintéticos.

| Componente documentado | Evidencia original | Implementación NextGen | Estado verificable |
|---|---|---|---|
| **La Roca** | Núcleo estable y conocimiento persistente | `LaRoca`: proyección estable con ancla no entrenable | Implementado |
| **El Líquido** | Adaptación y plasticidad | `ElLiquido`: proyección plástica + traza Hebbiana acotada y snapshot versionado | Implementado |
| **MoE** | Expertos dispersos y router | `AethelModel` conserva router, top-k y carga por experto | Implementado |
| **RoPE** | Posición relativa rotatoria | Atención del núcleo existente | Implementado |
| **GQA** | Menos cabezas K/V que Q | Atención del núcleo existente con `n_kv_heads` y repetición de K/V | Implementado |
| **KV-Cache** | Inferencia incremental | API de atención y generación del núcleo existente | Implementado en núcleo; NextGen lo hereda |
| **Ultra-eficiencia** | RMSNorm, peso compartido, MoE activo y GQA | Se mantienen RMSNorm, weight tying, top-k MoE y GQA | Implementado |
| **Ciclo de Sueño** | Consolidación y replay | `CicloDeSueno`: replay priorizado y manifiesto de consolidación | Implementado |
| **Neuromodulación** | Señal de sorpresa y prioridad | `Neuromodulacion`: prioridad derivada del estado y pérdida real | Implementado |
| **Espacio de Trabajo Global** | Integración de estados especializados | `EspacioTrabajoGlobal`: gating entre Roca, Líquido y memoria recuperada | Implementado |

## Desviaciones corregidas

La arquitectura original de prueba incluía una atención GQA simplificada y un MoE que ejecutaba siempre el primer experto; NextGen utiliza el núcleo `AethelModel`, que sí enruta y reporta expertos reales. La arquitectura original también no tenía memoria persistente, consolidación ni una ruta explícita de plasticidad; NextGen añade esos módulos sin afirmar que constituyan consciencia o inteligencia general.

El entrenamiento original mezclaba datos de ejemplo y fuentes externas. NextGen recibe un corpus local real mediante `--corpus`, persiste checkpoints y métricas, y deja sin puntuaciones cualquier benchmark que no haya sido ejecutado con su harness correspondiente.

## Límites

La alineación arquitectónica no equivale a competir con modelos frontier. La corrida local valida integración, pérdida, memoria, plasticidad y telemetría; no demuestra capacidades generales. La comparación frontier requiere más datos, más pasos, tokenización de producción, evaluación oficial y hardware de entrenamiento adecuado.
