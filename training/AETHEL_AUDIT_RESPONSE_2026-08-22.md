# Respuesta verificable a la auditoría técnica de Aethel

**Fecha:** 22 de agosto de 2026  
**Ámbito:** contraste local de la auditoría aportada por el usuario con la rama `main` de Aethel.  
**Límite:** esta respuesta no inició Kaggle, GPU, entrenamiento, subida de Dataset ni deserialización de checkpoints.

---

## 1. Juicio técnico

La auditoría tiene razón en su conclusión principal: **Aethel no debe presentarse como un modelo propio entrenado, un runtime GPU industrial o un producto comercial terminado**. El repositorio contiene una base de ingeniería, contratos de seguridad y rutas de entrenamiento preparadas; no contiene una evidencia reproducible de Seed E0 ejecutado en GPU, evaluación bilingüe ni aceptación CUDA/Triton.

El contraste también identificó una diferencia de entorno importante. El checkout auditado no incluía el paquete de Dataset externo al repositorio. En el entorno actual de desarrollo sí existe `/home/ubuntu/aethel-knowledge-corpus-v1-package/`, pero sigue siendo un activo **local, externo a GitHub y no publicado como Dataset privado final**. Su reporte local declara validación sin red, 22 shards y los conteos indicados en la tabla siguiente. Eso respalda que el paquete existe en este host; no prueba que esté disponible en otro checkout, en Kaggle o en un host GPU. [1]

| Evidencia contrastada | Resultado actual | Interpretación permitida |
|---|---|---|
| Paquete bilingüe local | `valid: true`, `network_requests: 0`, 22 shards; train: 19.011 en / 19.012 es; holdout: 989 en / 988 es. | Dataset Seed local verificable en este host; no Dataset Kaggle ni corpus comercial completo. |
| Frontend | `pnpm exec tsc --noEmit` correcto; Vitest: 4 archivos y 9 pruebas correctas. | Controles web y transparencia reproducibles en este checkout. |
| Contratos CPU | Prefill causal, dispatch/combina MoE, bridge Triton y el inspector de host pasaron. | Referencias/guards CPU comprobados; no aceptación CUDA. |
| GPU | `nvidia-smi` no está disponible y Triton informa ausencia de runtime CUDA. | No existe evidencia CUDA; no se debe iniciar Seed aquí. |
| Runtime Rust | Compilación release y 4 pruebas ya verificadas. | Núcleo local compilable; no servicio 24/7. |

> **Regla operativa:** la validez de un activo se expresa junto con su entorno, hash, comando y fecha. Un resultado local no se transfiere automáticamente a GitHub, Kaggle, otro host ni a producción.

---

## 2. Clasificación del archivo `engine/artifacts/aethel_real.pt`

La auditoría detectó correctamente una contradicción entre la política del directorio —los artefactos de entrenamiento deben residir fuera de Git— y el archivo versionado `engine/artifacts/aethel_real.pt`. Una inspección estática confirma un contenedor ZIP de serialización PyTorch de **991.114 bytes**, hash SHA-256 `fa423241ff0d94ea5819e9628c41d16940a4e846c5c625c030a9cbc0a9162122`, con `data.pkl` y quince bloques de almacenamiento tensorial. Esa estructura no demuestra por sí sola configuración, paso, tokenizador, optimizador, procedencia, evaluación o compatibilidad con la receta Seed actual. [2]

El archivo queda clasificado como **histórico, no certificado y no promocionable**. No se ha cargado ni deserializado durante esta auditoría porque un checkpoint PyTorch puede incluir datos serializados que no deben evaluarse fuera de un procedimiento confiable y aislado. Su sidecar versionado registra exclusivamente los metadatos estáticos observados. [3]

Para poder considerarlo como candidato de reanudación, un entorno de auditoría confiable deberá comprobar de forma explícita: hash, claves, tensor count, configuración completa, paso, tokenizador, estados de optimizador, versión de código, procedencia del Dataset y evaluación holdout. Hasta entonces, este archivo no prueba entrenamiento Aethel ni debe ser cargado por los launchers de Seed.

---

## 3. Reproducibilidad de pruebas

La auditoría también señaló una brecha real: `pytest` no está presente en este host y `training/requirements.txt` describe las dependencias del entrenamiento, no las herramientas de prueba. Se añade `training/requirements-test.txt` para que un entorno verificable instale las dependencias del laboratorio junto con `pytest`, sin cambiar el entorno actual ni confundir una dependencia declarada con una prueba ya ejecutada. [4]

La suite de comunicación válida a esta fecha es: **TypeScript correcto; Vitest 4/4 archivos y 9/9 pruebas; cuatro comprobaciones CPU explícitas correctas; pytest no ejecutado; CUDA/Triton de producción no ejecutados.** Cualquier reporte posterior debe conservar esta separación.

---

## 4. Próximo paso seguro

La prioridad no es expandir la arquitectura ni iniciar una GPU. Primero debe preservarse la evidencia: mantener el Dataset congelado sin mutarlo, dejar el artefacto histórico fuera de la ruta de promoción y ejecutar el preflight del host sólo cuando exista una GPU autorizada. La primera ejecución útil seguirá siendo Seed E0 con Dataset trazable, checkpoint atómico, reanudación y evaluación holdout en inglés/español. Si el contrato Triton bloquea la ruta estricta, la alternativa PyTorch exige autorización independiente y sus resultados permanecerán en categoría de laboratorio. [5]

## Referencias

[1]: `/home/ubuntu/aethel-knowledge-corpus-v1-package/package_validation_report.json` y `package_manifest.json` — Validación local y manifiesto de paquete externo a GitHub.  
[2]: `../engine/artifacts/aethel_real.pt` y `../engine/artifacts/README.md` — Contenedor histórico y política de artefactos.  
[3]: `../engine/artifacts/aethel_real.audit.json` — Sidecar de observación estática y clasificación.  
[4]: `requirements.txt` y `requirements-test.txt` — Dependencias declaradas de entrenamiento y pruebas.  
[5]: `AETHEL_SEED_OFFLINE_RUNBOOK_V1.md` y `AETHEL_TRITON_CUDA_ACCEPTANCE_MATRIX_V1.md` — Gates de Seed y aceptación CUDA.
