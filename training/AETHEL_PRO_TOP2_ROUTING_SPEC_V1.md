# Routing top-2 de Aethel Pro v1

**Estado:** contrato de diseño. El dispatch/combina Triton completo y la validación CUDA siguen pendientes; este documento no declara disponibilidad GPU de producción.

## Secuencia por token

Cada capa MoE proyecta el vector oculto hacia ocho logits de expertos. El router selecciona los dos índices de mayor puntuación, normaliza sus gates sólo entre esos dos expertos y entrega el token a ambas rutas. La salida de cada experto se combina mediante la suma ponderada de sus gates normalizados. Sólo dos expertos quedan activos por token, mientras todos los pesos de expertos siguen presentes en memoria.

| Etapa | Entrada | Salida | Telemetría obligatoria |
|---|---|---|---|
| Router | Vector oculto del token. | Ocho logits. | Entropía y distribución de selección. |
| Top-2 | Logits de router. | Dos expertos e índices. | Carga por experto y tokens descartados. |
| Capacidad | Asignaciones top-2. | Cupos de cada experto. | Overflow, factor de capacidad y orden de despacho. |
| Dispatch | Token + índice de experto. | Buffers agrupados por experto. | Tamaño de buffer y dispositivo. |
| Expertos | Buffers seleccionados. | Dos salidas SwiGLU por token. | Tiempo y errores numéricos. |
| Combina | Salidas + gates. | Vector oculto de capa. | Suma de gates y verificación de forma. |

## Balance y límites

El router debe evitar el colapso hacia pocos expertos. Aethel registra desequilibrio máximo y entropía mínima, y la receta Seed incorpora umbrales de salud que se guardan en métricas. Los umbrales no son una demostración de balance real hasta una corrida CUDA. Si un experto supera su capacidad, el evento debe quedar registrado y el protocolo debe aplicar una política explícita; nunca se oculta como si el token hubiese sido procesado normalmente.

## Contrato Triton

El prefill causal y el dispatch/combina MoE se mantienen bajo `require_triton=True`. Hasta que existan kernels Triton validados numéricamente para esas rutas, la ejecución GPU comercial queda bloqueada. Un fallback PyTorch puede servir sólo a E0 con autorización de laboratorio separada, y sus resultados no cumplen el criterio de producción Pro.

La validación futura compara salida, gradientes, gates, selección top-2, combinación, causalidad, consumo de memoria y rendimiento contra una referencia controlada, con el mismo hardware, semilla y lote.
