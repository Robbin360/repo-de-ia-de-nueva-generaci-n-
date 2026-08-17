# Plan de entrenamiento continuo en GPU para Aethel

## Decisión de infraestructura

El despliegue web actual no debe ejecutar el preentrenamiento: su contenedor está diseñado para servir la aplicación y usa PyTorch CPU. Una instancia persistente con GPU, almacenamiento externo de checkpoints y reinicio automático es la ruta correcta para una corrida larga. El servicio de cómputo persistente incluido no aporta GPU, por lo que se usaría únicamente para control u observabilidad, no para el entrenamiento.

| Opción | Uso recomendado | Persistencia | Indicador de coste publicado | Decisión operativa |
|---|---|---|---|---|
| Runpod Pod dedicado | Primera corrida Aethel de 100M–1B parámetros | Volumen o almacenamiento de red | RTX A6000 48 GB desde US$0.53/h; A100 80 GB desde US$1.39/h | Opción inicial recomendada por flexibilidad |
| Lambda GPU Instance | Corridas multi-GPU con operación más uniforme | Almacenamiento adjunto entre sesiones | 1× A100 80 GB desde US$2.79/GPU-h | Opción para 4–8 GPU o soporte empresarial |
| Vast.ai con volumen | Exploración de menor coste con recuperación robusta | Volumen local y copia externa | Depende del anfitrión | Alternativa económica, no fuente primaria de checkpoints |

Los precios y disponibilidad cambian. La selección no debe realizarse sin la autorización del propietario de la cuenta y un límite de gasto.

## El modelo Aethel de escala

Aethel no se define como una emulación literal de un cerebro humano. Es un modelo de lenguaje experimental con mecanismos inspirados en funciones cognitivas y controles de seguridad. Mantendrá el núcleo Transformer eficiente del repositorio y escalará cada módulo de manera verificable:

| Módulo | Función implementada | Escalado propuesto |
|---|---|---|
| La Roca | Ruta estable y proyección de identidad | Pesos base protegidos, distilación y anclaje frente a olvido |
| El Líquido | Traza Hebbiana limitada y versionada | Adaptadores de baja dimensión, snapshots auditables y rollback |
| MoE disperso | Expertos top-k con balanceo | 8–64 expertos, 2 activos por token y pérdida de balanceo |
| RoPE y GQA | Posición rotatoria y grupos de KV | Contexto largo con KV-cache y menor coste de atención |
| Espacio de Trabajo Global | Puerta que reúne ruta sólida, líquida y memoria | Selección de hipótesis con telemetría de compuertas |
| Ciclo de Sueño | Replay priorizado | Consolidación periódica, evaluación de regresión y rotación de memoria |
| Neuromodulación | Sorpresa y prioridad | Controla observación, replay y tasa de actualización, sin auto-modificar pesos fuera del optimizador |

La primera meta realista es validar una configuración de **100M–300M parámetros**, no afirmar equivalencia con modelos frontier. Tras pruebas de escalabilidad, datos, pérdidas y benchmarks reproducibles, se podrá decidir si una variante de 1B+ parámetros justifica un clúster multi-GPU.

## Validación FSDP obligatoria

Antes de crear una corrida de la familia `scale-1b`, ejecutar en un host con al menos dos GPU CUDA:

```bash
bash training/run_fsdp_validation.sh
```

La prueba usa `torchrun` con dos procesos, ejecuta FSDP real durante un paso, comprueba que sólo el rango 0 persiste `latest.pt`, reanuda hasta el paso dos y verifica los registros de ambos rangos. En un host sin dos GPU CUDA imprime `SKIPPED`; eso no equivale a una validación aprobada.

## Corpus y gobierno de datos

La ingesta se realizará con un manifiesto versionado, hashes, filtros, deduplicación, detección de PII y particiones separadas de entrenamiento, validación y pruebas. Para un piloto multilingüe y auditable se priorizarán fuentes con documentación explícita de procedencia y licencia. FineWeb-Edu declara licencia ODC-BY y ofrece subconjuntos de muestra; Common Corpus comunica contenido con licencia permisiva, procedencia y cobertura en español; RedPajama-V2 ofrece señales de calidad y deduplicación, pero requiere una revisión jurídica adicional de cada fuente antes de uso productivo.

No se iniciará ninguna descarga masiva hasta que se apruebe el manifiesto, la licencia de cada fuente y el presupuesto de almacenamiento y cómputo.

## Entorno reproducible objetivo

La instancia GPU tendrá CUDA/PyTorch compatible, Python aislado, repositorio fijado a un commit, almacenamiento de objetos para pesos y métricas, y un servicio de supervisión que reinicie la corrida desde el último checkpoint. La salida persistente incluirá: estado de optimizador, estado del programador de tasa, número de paso, RNG de cada worker, pesos, snapshots de El Líquido, memoria episódica, replay y manifiestos de datos.

El script actual se ampliará con BPE versionado, precisión mixta BF16, acumulación de gradiente, FSDP/DDP, guardado atómico, subida a almacenamiento externo y reanudación exacta. Los benchmarks MMLU, HumanEval y GSM8K solo se cargarán cuando se integre su harness, dataset y licencia correspondiente; no habrá resultados sintéticos.

## Referencias

[1]: https://www.runpod.io/pricing "Runpod GPU Cloud Pricing"
[2]: https://lambda.ai/instances "Lambda GPU Instances"
[3]: https://docs.vast.ai/guides/instances/storage/types "Vast.ai Storage Types"
[4]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu "FineWeb-Edu dataset card"
[5]: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2 "RedPajama-V2 dataset card"
[6]: https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open "Common Corpus announcement"
