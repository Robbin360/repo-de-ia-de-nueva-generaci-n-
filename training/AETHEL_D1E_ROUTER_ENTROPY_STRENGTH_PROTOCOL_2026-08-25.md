# D1E — Sonda de fuerza de regularización de entropía

## Estado y límites

> **Estado: `D1E_LOCAL_PROTOCOL_ONLY`.** D1E es una hipótesis local derivada de la señal tardía observada en D1D. Este documento no autoriza subir un Dataset, editar un notebook, seleccionar GPU, ejecutar entrenamiento, cargar checkpoints, abrir holdout ni promover artefactos.

D1D terminó con una media de pérdida dentro del umbral y un último paso saludable, pero sólo 52 de 768 pasos fueron saludables y la entropía mínima global fue `0.3333333433`. Esto indica una señal compatible con mejora tardía, pero no estabilidad suficiente en toda la ventana. D1E prueba una sola modificación: aumentar la magnitud de la señal densa, sin cambiar arquitectura, datos, semilla, duración ni inicialización.

## Hipótesis falsable

> Con el mismo baseline de D1D y un único aumento de `router_entropy_loss_weight` de `0.01` a `0.03`, la señal densa actuará con suficiente magnitud para reducir los episodios de baja entropía durante la ventana completa, manteniendo la pérdida media dentro del umbral predefinido.

El valor `0.03` es una sonda independiente, no un valor validado. No se permite barrer pesos, escoger el mejor resultado después de observarlo ni reutilizar los pesos de D1D. Si D1E falla, cualquier calendario, warmup o cambio de arquitectura deberá tener otro protocolo.

## Diseño controlado

| Control | D1D observado | D1E propuesto |
|---|---:|---:|
| Inicialización | Nueva | Nueva; sin reanudar |
| Datos | Sólo train; holdout sellado | Sólo train; holdout sellado |
| Pasos / tokens | 768 / 1.572.864 | 768 / 1.572.864 |
| Seed | 17 | 17 |
| Arquitectura | 512, 4 capas, GQA 8/2, 8 expertos top-2 | Idéntica |
| `router_aux_loss_weight` | 0.05 | 0.05 |
| `router_entropy_loss_weight` | 0.01 | **0.03** |
| Salida | `aethel-d1d-router-entropy-001-run-v1` | Directorio inédito |
| Checkpoint | No cargado | No cargado |

## Criterios de decisión

D1E conservará la regla global: entropía mínima estrictamente superior a `0.5`, desequilibrio máximo inferior a `0.3`, al menos `117/768` pasos saludables y pérdida media no superior a `9.53127756`. Los cuatro criterios deben cumplirse simultáneamente para clasificar la corrida como `D1E_ROUTER_HEALTHY_CANDIDATE`.

Si falla cualquiera de ellos, la clasificación será `D1E_ROUTER_NOT_IMPROVED`. Un último paso saludable no compensa una ventana global fallida. Una pérdida media aceptable tampoco compensa inestabilidad del router.

## Evidencia requerida

El resumen debe declarar `diagnostic_id=D1E`, release exacto, seed, pasos, tokens, peso auxiliar, peso de entropía, pérdida media, pasos saludables, entropía mínima, desequilibrio máximo, `checkpoint_loaded=false`, `holdout_content_read=false`, `raw_corpus_read=false`, `network_requests=0` y dispositivo real. La corrida debe finalizar con un marcador inequívoco de completitud.

## Puertas

La validación matemática y la integración D1D ya están documentadas. D1E permanece bloqueada para bundle, Kaggle, notebook, GPU y entrenamiento hasta completar una implementación local separada y obtener autorizaciones inmediatas para cada acción externa. No se reutiliza `latest.pt`, no se inspeccionan artefactos de D1D y no se ejecuta holdout.

## No afirmaciones permitidas

Incluso si D1E supera sus criterios de routing, sólo demostraría una mejora experimental de estabilidad del router bajo una configuración pequeña. No demostraría calidad bilingüe, razonamiento general, inteligencia general, rendimiento frontier, servicio productivo ni un modelo comercial listo.
