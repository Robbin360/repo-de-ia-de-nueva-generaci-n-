# Política de artefactos de entrenamiento

Los artefactos de entrenamiento certificados se almacenan fuera del repositorio Git. Use `--checkpoint-dir` y `--memory-path` para apuntar a un volumen persistente y controlado.

`aethel_real.pt` permanece en este directorio únicamente como un **artefacto histórico no certificado**. No es evidencia de una corrida Seed, no es apto para reanudar ni debe utilizarse para inferencia, benchmarks o promoción. Su hash, observaciones estáticas y requisitos de eventual auditoría constan en `aethel_real.audit.json`.

No deserialice checkpoints no confiables como parte de una inspección rutinaria. Antes de cargar un archivo histórico en un entorno aislado y autorizado, se deben verificar su hash, procedencia, configuración, paso, tokenizador, estados de optimizador y separación de holdout.
