# Aethel — runbook de bloqueos externos y reanudación segura

**Estado:** documento operativo; no ejecuta acciones externas  
**Fecha:** 2026-08-29

Este documento evita que el proyecto confunda preparación local con ejecución real. Las acciones descritas en la tabla requieren recursos o decisiones fuera del repositorio y deben conservar sus recibos.

| Bloqueo | Estado verificable | Requisito de desbloqueo | Evidencia mínima |
|---|---|---|---|
| Nueva corrida Kaggle | No iniciada en esta sesión | Cuota disponible, notebook correcto y autorización inmediata | Log de preflight, GPU, pasos, checkpoint y recibo |
| Dataset privado de datos | Preparado localmente; su publicación puede estar suspendida | Confirmación del usuario y cuenta Kaggle conectada | ID/versión privada y hashes de 22 shards |
| Save Version / Save & Run All | No se ejecuta automáticamente | Usuario conectado a Kaggle y confirmación final de consumo de cuota | Número de versión, log completo y artefactos persistentes |
| FSDP real | Contrato y rechazo local disponibles | Al menos dos procesos y dos GPU CUDA utilizables | Rango 0, estado distribuido, checkpoint y reanudación |
| Triton estricto | Kernel experimental y matriz disponibles | Validación numérica y de gradientes en GPU objetivo | Paridad, gradientes, memoria y benchmark |
| Servicio Rust 24/7 | Plantilla local, no servicio desplegado | Host autorizado, supervisión y almacenamiento persistente | Healthcheck, snapshot, restauración y logs |
| Runtime Mojo | Contrato de inferencia definido, sin runtime certificado | Toolchain compatible y benchmark contra PyTorch | Paridad token a token y latencia/memoria |

## Secuencia segura para Kaggle

Primero se debe confirmar que el usuario puede abrir su sesión personal de Kaggle y que el notebook utiliza exactamente el bundle de código y el Dataset de datos aprobados. Después se ejecuta únicamente el preflight: debe informar rutas, release, manifiesto, tokenizador, conteo de shards, holdout y disponibilidad CUDA sin cargar pesos ni iniciar entrenamiento.

Sólo después de revisar ese recibo se solicita autorización específica para consumir cuota. La corrida debe usar una salida inédita, guardar checkpoints atómicos cada intervalo fijo y escribir al menos `latest.pt`, estado del optimizador, scheduler, RNG, tokenizer, manifiesto, métricas y `recovery_receipt.json`. Antes de cerrar la sesión se debe comprobar que el artefacto exportado puede volver a montarse como input.

## Regla de reanudación

Una sesión nueva no debe “adivinar” el checkpoint. Debe resolver un único artefacto por hash, verificar que el tokenizer y manifiesto coinciden, inspeccionar claves y formas, y sólo entonces cargar pesos y estado del optimizador. Si falta cualquier componente, la ejecución se detiene sin crear una versión aparentemente compatible.

## Regla de evaluación

El holdout debe permanecer separado del entrenamiento y del replay. Se calculan pérdidas EN y ES, tareas de conversación, seguimiento de instrucciones, razonamiento elemental y matemáticas básicas. Un resultado no se promueve por pérdida aislada: también debe conservar salud del router, estabilidad de memoria, ausencia de NaN, generación controlada y equivalencia de reanudación.

## Qué no debe hacerse

No se debe editar un Dataset histórico para “arreglar” una corrida, borrar una salida residual para satisfacer un preflight, seleccionar una copia montada por tamaño, publicar pesos sin revisión ni declarar un modelo funcional porque el proceso alcanzó un número de pasos. Cada corrección debe producir un release nuevo, un hash nuevo y una nota de compatibilidad.

Tampoco se debe afirmar que Aethel tiene inteligencia general, consciencia, IQ humano o razonamiento de frontera. La documentación revolucionaria describe hipótesis de arquitectura; sólo las mediciones comparativas pueden convertir una hipótesis en una decisión técnica.

## Criterio de desbloqueo global

La ruta de entrenamiento se considera desbloqueada únicamente cuando están disponibles simultáneamente: una GPU autorizada, un corpus congelado, un holdout separado, una configuración versionada, una salida persistente, un procedimiento de reanudación y una autorización explícita para iniciar la sesión. Si falta uno, el estado correcto es **BLOCKED**, no “ready”.
