# Resultado D1D — 2026-08-25

## Alcance de la evidencia

Este informe registra únicamente la salida textual compartida por el usuario desde Kaggle. No se abrieron checkpoints, métricas crudas ni artefactos locales protegidos desde este entorno.

La corrida terminó con `D1D_DIAGNOSTIC_COMPLETE` en CUDA, con inicio fresco, seed `17`, `768` pasos y `1.572.864` tokens. La evidencia declara `checkpoint_loaded=false`, `holdout_content_read=false`, `raw_corpus_read=false`, `network_requests=0` y `promotion_authorized=false`.

## Resultado cuantitativo

| Métrica | Resultado | Criterio | Estado |
|---|---:|---:|---|
| Pasos observados | 768 | 768 | Cumple |
| Pasos saludables | 52 | >=117 | No cumple |
| Entropía mínima global | 0.3333333433 | >0.5 | No cumple |
| Desequilibrio máximo global | 0.1875 | <0.3 | Cumple |
| Pérdida media | 9.3994066852 | <=9.53127756 | Cumple |
| Último paso saludable | Sí | — | Señal tardía, no suficiente |
| Entropía mínima en paso 768 | 0.5975525975 | >0.5 | Cumple aisladamente |
| Desequilibrio máximo en paso 768 | 0.1442871094 | <0.3 | Cumple aisladamente |

La clasificación es **`D1D_ROUTER_NOT_IMPROVED`**. El regularizador produjo una mejora tardía observable, pero no estabilidad suficiente durante la ventana completa. Por tanto, esta evidencia no autoriza evaluación holdout, reanudación, promoción, serving ni la afirmación de un modelo funcional.

## Estado operativo

D1D queda cerrada como una intervención no mejorada bajo su criterio global. Una siguiente intervención debe tener un protocolo independiente y no puede seleccionar parámetros después de observar este resultado. El checkpoint y los outputs permanecen fuera del alcance de este informe.
