# Aethel NextGen — Resultado esperado del piloto Kaggle

**Estado:** receta de validación técnica, no lanzamiento de un modelo de frontera.

## Objetivo del piloto

El cuaderno `aethel-nextgen-bilingual-pilot` valida que el preset acotado puede construir un corpus bilingüe trazable, entrenar con el núcleo Aethel NextGen y conservar artefactos íntegros al terminar o interrumpirse una sesión de Kaggle. No se debe describir su resultado como un modelo final, AGI, sistema consciente o competidor de modelos de frontera.

| Área | Evidencia que debe producir la corrida | Criterio inicial |
|---|---|---|
| Datos | `prepared_manifest.json`, hashes de shards, procedencia, revisión y mezcla por idioma | Sólo fuentes aprobadas; ninguna ruta o peso Aethel V3 |
| Tokenización | `aethel-bpe.json` y metadatos del tokenizador | Cobertura entrenada sobre la mezcla español–inglés del piloto |
| Optimización | `metrics_rank_0.jsonl`, pérdida, tasa de tokens, LR y coste observado | Sin pérdida no finita; pérdida holdout no divergente |
| MoE | `router_health`, cargas de experto y entropía | `healthy=true` tras el calentamiento configurado |
| Memoria | manifiestos de replay y trazas de El Líquido | Eventos de replay y versiones observables, no estado opaco |
| Recuperación | `latest.pt`, checkpoints por paso, configuración y tokenizador empaquetados | Inspección reproducible y reanudación desde el último checkpoint íntegro |

## Interpretación honesta

El piloto puede evidenciar que Aethel procesa una mezcla de español e inglés, que el entrenamiento es estable bajo sus controles y que sus artefactos son reanudables. No mide por sí solo fluidez nativa indistinguible en ambos idiomas, traducción profesional, conocimiento exhaustivo, razonamiento general fiable ni capacidad competitiva frente a modelos de frontera. Dichas afirmaciones requerirían presupuestos de datos y cómputo muy superiores, evaluaciones contaminadas de forma controlada y comparación externa reproducible.

## Política frente al límite de sesión

Kaggle puede terminar una ejecución al alcanzar el límite de tiempo. Por ello, el piloto debe guardar mediante reemplazo atómico un `latest.pt` periódico y checkpoints de paso, vaciar métricas después de cada evento y conservar junto a los pesos la configuración, el paso, la estrategia y una copia del tokenizador. Al recibir una señal de terminación, el runner debe finalizar el paso seguro actual, guardar el último estado íntegro y dejar un marcador de interrupción.

El piloto conserva `latest.pt` como estado completo atómico y, por defecto, las tres copias portátiles más recientes `step_*.pt`. Esta retención acotada limita el crecimiento de la salida y ofrece puntos alternativos de inspección sin sustituir el checkpoint completo.

El modo `notebook-output` conserva esos archivos como salida de una versión comprometida. Para una nueva sesión, se debe crear o actualizar un Dataset privado de artefactos con la salida completa, adjuntarlo al siguiente cuaderno y proporcionar explícitamente la ruta de su `latest.pt` a `AETHEL_RESUME_CHECKPOINT`. El lanzador inspecciona el artefacto y el entrenador rechaza tokenizador o configuración incompatibles; no se puede asumir que el directorio efímero `/kaggle/working` permanecerá entre sesiones.

## Umbrales de decisión

La versión debe considerarse una validación técnica si crea el recibo de persistencia, el checkpoint pasa el inspector reproducible y las métricas no contienen pérdidas no finitas. Si falta alguno de esos artefactos, la siguiente acción es corregir la infraestructura o reanudar; no reportar capacidades del modelo. El paso a una escala superior queda bloqueado hasta completar la evaluación retenida y demostrar estabilidad, retención y coste medido.
