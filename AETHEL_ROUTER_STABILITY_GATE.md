# Aethel — puerta de estabilidad del router MoE

**Estado:** diseño y contrato de evaluación; no habilita cambios automáticos en entrenamiento.

## Problema observado

La entropía de la distribución suave y la pérdida auxiliar no garantizan que la selección top-k use suficientes expertos. El experimento CPU de masa probabilística suave mejoró ligeramente la masa probabilística, pero no rompió por sí solo el atractor top-2. Por ello, la salud del router debe observar simultáneamente la distribución suave y la asignación dura que realmente recibe tokens.

## Señales mínimas

| Señal | Definición | Motivo |
|---|---|---|
| Entropía suave | Entropía normalizada de `router_probability` | Detecta concentración antes de top-k |
| Densidad dura | Fracción de tokens asignados por experto | Detecta colapso efectivo |
| Cobertura top-k | Expertos con al menos una asignación / expertos | Detecta expertos nunca usados |
| Desequilibrio | Máxima desviación de densidad respecto a uniforme | Detecta sobrecarga |
| Overflow | Tokens descartados o capacidad excedida | Detecta pérdida silenciosa de capacidad |
| Variación temporal | Peor ventana y percentil alto, no sólo media | Evita ocultar colapso inicial o intermitente |

## Regla de clasificación

Un paso no puede ser saludable si cualquiera de estas condiciones ocurre:

1. La entropía mínima queda bajo el umbral definido.
2. La densidad dura máxima supera el límite.
3. La cobertura top-k cae por debajo del mínimo durante una ventana no trivial.
4. Existe overflow, NaN, inf o telemetría ausente.

Los umbrales deben fijarse antes de mirar el resultado de una nueva corrida. No se permite relajar la puerta para promover un checkpoint que falle.

## Orden de intervención recomendado

Primero se debe corregir la medición para distinguir probabilidad suave de asignación dura. Después se prueban, en ese orden, capacidad por experto y dispatch, jitter sólo durante selección, pérdida auxiliar con peso pequeño y temperatura/entropía calibrada. Cada variante debe compararse con la misma semilla, datos, pasos y presupuesto.

No se recomienda comenzar aumentando el peso de la pérdida auxiliar: puede mejorar una señal agregada mientras el top-k continúa concentrado o puede degradar la pérdida de lenguaje. La masa suave no debe usarse como sustituto de la densidad dura.

## Criterio de promoción

Una variante sólo puede pasar a entrenamiento largo si, en una prueba CPU determinista y posteriormente en una prueba CUDA:

- mantiene la pérdida de lenguaje dentro del margen predefinido del baseline;
- reduce la peor concentración dura y no sólo la media;
- no introduce overflow ni inestabilidad numérica;
- conserva throughput y memoria dentro del presupuesto;
- mantiene reanudación y telemetría reproducibles.

Hasta completar la prueba CUDA, esta puerta es un contrato de aceptación y no evidencia de que el router ya esté estable.

## Relación con la implementación

`engine/router_health.py` ya valida entropía e imbalance normalizados. La siguiente extensión segura debe añadir métricas de asignación dura como campos explícitos, con pruebas de rechazo para cobertura insuficiente y overflow. No se debe cambiar el umbral por defecto ni habilitar una nueva política en el runner sin una ablación registrada.

## Estado actual

- La comparación CPU baseline/ARC no demuestra ultra-eficiencia.
- El experimento de masa suave no elimina el colapso top-2.
- La validación CUDA está pendiente por falta de GPU disponible.
- No se ha promovido ningún checkpoint por esta puerta.
