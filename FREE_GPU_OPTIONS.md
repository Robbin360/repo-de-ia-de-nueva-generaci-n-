# Opciones gratuitas para la primera corrida de Aethel

## Conclusión operativa

Para una primera corrida de entrenamiento real, la opción gratuita más útil es un **Kaggle Notebook con GPU**. Su documentación actual indica acceso gratuito a Tesla P100 y una cuota semanal que normalmente es de hasta 30 horas, aunque puede variar con la demanda. Es suficiente para validar la escalabilidad de Aethel, el tokenizador BPE, MoE, plasticidad y checkpoints en sesiones recuperables; no equivale a un entorno persistente ni a una corrida frontier. [1]

| Plataforma | Recurso gratuito oficial | Límite relevante | Ajuste para Aethel | Decisión |
|---|---|---|---|---|
| Kaggle Notebooks | GPU NVIDIA Tesla P100 | Cuota semanal de hasta 30 h, variable | Entrenamiento temporal por bloques con checkpoints | **Recomendada** |
| Google Colab Free | Recursos GPU/TPU no garantizados | Máxima vida de VM, recursos y límites dinámicos; prohíbe workers distribuidos en el nivel gratuito | Diagnóstico y microcorridas, no entrenamiento continuo | Alternativa secundaria |
| Hugging Face ZeroGPU | RTX Pro 6000 Blackwell compartida bajo demanda | Cuenta gratuita: 5 min de cuota diaria; funciones usualmente de hasta 60 s | Inferencia/demos, no preentrenamiento | No usar para entrenar |

Kaggle no garantiza una GPU concreta ni una sesión persistente. La estrategia correcta es guardar `latest.pt`, checkpoints por paso, métricas JSONL, el manifiesto del corpus y el tokenizador fuera de la sesión. El pipeline de Aethel empaqueta estos artefactos con hashes SHA-256 y los versiona en un Dataset privado de Kaggle configurado mediante `AETHEL_KAGGLE_DATASET`; después se reanuda con `--resume`.

## Protocolo de una corrida gratuita

La primera fase debe operar con un presupuesto temporal explícito: preparar solo un subconjunto aprobado del corpus, entrenar un BPE de 32k tokens, ejecutar bloques de 2–4 horas y copiar los artefactos al final de cada bloque. La configuración inicial recomendada es `dim=512`, `capas=4`, `cabezas=8`, `kv_heads=2`, `expertos=8`, `top_k=2`, `seq_len=1024` y batch efectivo 32. Solo los pesos de los expertos de esta configuración suman aproximadamente 75,497,472 parámetros; el núcleo completo queda cerca del objetivo de 100M, manteniendo margen para los estados del optimizador en una P100 temporal. Antes de ampliar datos, hay que revisar la licencia y la procedencia de cada fuente del manifiesto. Ninguna entrada de corpus viene activada por defecto.

| Fase | Objetivo | Artefactos que deben sobrevivir a la sesión |
|---|---|---|
| Piloto | Verificar datos, tokenizador, pérdida, MoE y memoria | `prepared_manifest.json`, tokenizador y métricas |
| Escala inicial | Entrenar 100M–300M en bloques | `latest.pt`, snapshots de El Líquido, memoria episódica, métricas |
| Evaluación | Ejecutar holdout y harnesses permitidos | Configuración, hashes de datos y resultados crudos |
| Migración | Mover la misma corrida a una GPU persistente | Último checkpoint, RNG, optimizador y manifiestos |

> La oferta gratuita no es una ruta para competir de forma sostenible con modelos frontier. Sirve para eliminar fallos de arquitectura y entrenamiento antes de asumir el coste de GPU persistente y datos a gran escala.

## Referencias

[1]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle Efficient GPU Usage Tips"
[2]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ"
[3]: https://huggingface.co/docs/hub/en/spaces-zerogpu "Hugging Face ZeroGPU"
