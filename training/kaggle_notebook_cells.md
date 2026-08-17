# Celdas para un Kaggle Notebook de Aethel

Configura **Accelerator = GPU** y añade dos datasets privados de entrada: `aethel-source`, que contiene este repositorio sin checkpoints pesados, y `aethel-data`, que contiene el corpus preparado aprobado, el BPE y `evaluation/evaluation_plan.json`. El lanzador rechaza la sesión antes de instalar paquetes si el manifiesto, el holdout, el tokenizador o las referencias de benchmark no están aprobados y accesibles. Después ejecuta estas celdas, una por una.

```bash
%env AETHEL_KAGGLE_DATASET=TU_USUARIO/aethel-artifacts-privado
# Opcional: ruta de un checkpoint empaquetado que contenga model/config/step/tokenizer.
# Un .pth histórico crudo se inspecciona y se rechaza para evitar cargar una arquitectura incompatible.
%env AETHEL_RESUME_CHECKPOINT=
!bash /kaggle/input/aethel-source/training/run_kaggle_aethel.sh
```

Al terminar, el lanzador empaqueta el checkpoint, el tokenizador, las métricas, la memoria y el manifiesto con hashes SHA-256 y los versiona en el dataset privado indicado. El Notebook debe disponer de autenticación de Kaggle API mediante sus secretos; nunca añadas el token al repositorio. No uses la sesión como almacenamiento permanente.

Si la sesión se corta, agrega el último artefacto como input, descomprímelo sobre la misma ruta de salida, establece `AETHEL_RESUME_CHECKPOINT` a su `latest.pt` empaquetado y vuelve a ejecutar. El inspector rechaza pesos crudos que no incluyan configuración y tokenizador verificables; el flag `--resume` restaura después el estado de modelo y optimizador compatible.
