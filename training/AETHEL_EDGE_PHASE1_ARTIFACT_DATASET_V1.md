# Aethel Edge — preservar el checkpoint de fase 1 como input privado

Este paso no descarga ni reempaqueta pesos. En Kaggle, crea un dataset privado desde **Notebook Outputs** de la **Versión 1** ya terminada de `Aethel — Entrenamiento Directo Dataset V1`.

1. Abre **Datasets → New Dataset → Notebook Outputs**.
2. Selecciona la **Versión 1** que terminó en 35.120 segundos.
3. Selecciona sólo la carpeta `aethel-edge-phase-1-183680-v1`, que contiene `latest.pt`, `tokenizer.json`, los snapshots y recibos.
4. Asigna título y slug: `aethel-edge-phase1-artifacts-v1`.
5. Selecciona **Private** y crea el dataset.

No selecciones el TAR exterior, no borres la versión original y no añadas este input al cuaderno de entrenamiento. El dataset se usará únicamente en un cuaderno de evaluación separado.
