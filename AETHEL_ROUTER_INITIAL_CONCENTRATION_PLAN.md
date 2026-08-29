# Aethel — Plan para la concentración inicial del router

## Problema observado

En las corridas pequeñas del router MoE, la entropía suave puede mejorar tardíamente mientras la asignación top-k permanece concentrada durante demasiados pasos. El experimento de masa probabilística suave confirmó que una señal sobre la distribución previa a top-k no rompe por sí sola el atractor de asignación dura.

Este documento no declara una solución. Define una secuencia de experimentos controlados antes de ampliar tokens, tamaño de modelo o promoción de checkpoints.

## Hipótesis principal

El router aprende una asignación temprana concentrada porque la decisión top-k amplifica pequeñas diferencias de logits y los expertos favorecidos reciben más gradiente. La intervención debe reducir esa ventaja inicial sin destruir la especialización posterior ni elevar materialmente la pérdida.

## Intervenciones candidatas

| Intervención | Cambio | Riesgo | Criterio de aceptación |
|---|---|---|---|
| Temperatura de router con enfriamiento | Usar una temperatura inicial mayor y reducirla con un calendario fijo | Asignaciones demasiado difusas | Menor concentración inicial sin pérdida media fuera del margen fijado |
| Ruido de logits sólo durante entrenamiento | Añadir ruido reproducible antes de top-k, con amplitud decreciente | Variancia y degradación de convergencia | Mejora de cobertura por ventana y estabilidad entre seeds |
| Capacidad y overflow explícitos | Medir tokens por experto, overflow y descartes antes de cambiar capacidad | Más memoria o carga comunicada | Overflow bajo el límite y cobertura mayor |
| Pérdida de balanceo densa | Mantener una señal sobre probabilidades suaves | Puede no afectar a la decisión dura | Debe mejorar también la asignación top-k, no sólo entropía |
| Especialización gradual | Congelar temporalmente parte de la ventaja de los expertos dominantes | Retrasa especialización útil | Sólo se acepta si mejora salud sin aumentar pérdida |

No se deben combinar varias intervenciones en una misma primera prueba. Cada variante necesita un identificador, seed, salida inédita y criterios definidos antes de observar resultados.

## Protocolo recomendado

1. Ejecutar primero una simulación CPU determinista de logits controlados para verificar cobertura, concentración, overflow y gradientes de la variante.
2. Mantener arquitectura, datos, seed, pasos y acumulación idénticos al baseline de referencia.
3. Medir por ventanas: entropía suave, entropía de asignación top-k, cobertura de expertos, concentración máxima, desequilibrio, overflow, pérdida media y pasos saludables.
4. Clasificar la variante con criterios predefinidos: salud global, no sólo el último paso; pérdida dentro del margen; ausencia de overflow no explicado; y reproducibilidad en una segunda seed sólo después de superar la primera puerta.
5. Cerrar la línea si la variante mejora una métrica aislada pero empeora la salud global o la pérdida.

## Orden de experimentación

La primera sonda recomendada es **temperatura inicial fija con enfriamiento**, porque modifica la selectividad durante la fase de arranque sin añadir un término de pérdida ambiguo. La segunda sería ruido de logits reproducible, sólo si la primera falla. La masa probabilística suave ya fue probada como intervención aislada y no debe repetirse como si fuera nueva.

## Puertas estrictas

No se permite ampliar el corpus, elevar el número de expertos, aumentar tokens, reanudar un checkpoint anterior, abrir holdout, publicar pesos ni activar serving hasta que una variante supere el criterio global de salud del protocolo que la originó. Una mejora tardía o una selección posterior del mejor paso no constituye éxito.

## Estado actual

- `D1D`: `D1D_ROUTER_NOT_IMPROVED`.
- Masa probabilística suave: mejora limitada de distribución, sin romper el atractor top-k.
- Validación CUDA de nuevas variantes: pendiente.
- Entrenamiento de nueva variante: no iniciado.

Este plan es diseño experimental. No demuestra que Aethel tenga razonamiento, bilingüismo, eficiencia superior o un modelo funcional.

## Sonda CPU de temperatura — 2026-08-28

La sonda determinista `engine/test_router_temperature_probe.py` confirmó que dividir logits por una temperatura positiva cambia la entropía suave (`0.931089222` en frío frente a `1.980870485` en caliente), pero conserva exactamente los mismos índices de asignación determinista top-k. Por tanto, la temperatura sola no puede corregir el colapso duro si el router sigue usando `argtopk` sin muestreo, ruido o una señal de entrenamiento sensible a temperatura. La variante no debe promoverse como solución independiente; el siguiente candidato sería ruido reproducible de logits o una política de selección explícitamente estocástica, siempre con una ablación separada.
