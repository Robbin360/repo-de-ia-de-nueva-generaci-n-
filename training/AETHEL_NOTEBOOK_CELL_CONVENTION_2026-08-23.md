# Convención de numeración de celdas Kaggle — Aethel

## Regla vigente

Toda celda que se prepare en adelante para el notebook de Aethel debe comenzar dentro del propio código con un bloque visible en este formato:

```python
# =============================================================================
# CELDA <número> — <nombre breve del propósito>
# Propósito: <qué hace y qué no hace>
# Estado: <bloqueada, preparada, habilitable o ejecutada con evidencia>
# =============================================================================
```

El número representa la posición operativa prevista en el notebook, no un resultado experimental. El encabezado debe aclarar si la celda puede tocar Dataset, GPU, pesos, outputs o entrenamiento. Una celda no puede adquirir permisos por estar numerada: las confirmaciones inmediatas y los contratos técnicos siguen siendo obligatorios.

## Aplicación actual

La variante D1B habilitable se identifica como **CELDA 5**. Las celdas que se añadan después deben usar números consecutivos y documentar explícitamente su estado. No se renumeran retrospectivamente las celdas históricas sin una revisión específica del notebook.
