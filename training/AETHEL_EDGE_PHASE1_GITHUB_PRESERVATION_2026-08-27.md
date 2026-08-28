# Aethel Edge — Preservación de la Fase 1 en GitHub

## Estado confirmado

La primera sesión real de **Aethel Edge** terminó en una versión guardada de Kaggle el 27 de agosto de 2026. La ejecución registrada fue de **35.120 segundos** y produjo aproximadamente **5,49 GB** de salida. Esto confirma ejecución y preservación de artefactos, pero no valida todavía calidad bilingüe, razonamiento, matemáticas ni eficiencia.

| Evidencia | Valor confirmado |
|---|---|
| Checkpoint final | `latest.pt` |
| Paso global final | `183680` |
| Contrato de reanudación | `aethel-training-resume/v2` |
| Reanudación fiel declarada | `true` |
| Snapshots retenidos | `step_00176000.pt`, `step_00180000.pt`, `step_00183680.pt` |
| Hash del tokenizador | `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815` |
| Paquete de preservación | `aethel-edge-phase-1-183680-v1-preservation.tar.gz` |

El dataset privado reutilizable de la carpeta canónica de artefactos se llama `aethel-edge-phase1-artifacts-v1`. Kaggle extrajo además una copia del TAR dentro del dataset; los flujos posteriores deben seleccionar la carpeta canónica `aethel-edge-phase-1-183680-v1` y excluir cualquier ruta que contenga `preservation`.

## Qué se puede publicar en Git normal

El código, los scripts, contratos, pruebas, manifiestos sin datos y documentos de esta plataforma pueden residir en Git normal. **Los pesos, shards, archivos TAR y checkpoints no deben añadirse al historial Git normal.** Además de superar los límites de tamaño, incluirlos haría el repositorio difícil de clonar y no sustituye una preservación de artefactos.

## Opciones compatibles para pesos grandes

| Mecanismo | Límite relevante | Uso recomendado para Edge |
|---|---:|---|
| Git normal | GitHub bloquea archivos mayores de 100 MiB | No usar para `.pt` ni TAR. |
| Git LFS | GitHub Free admite hasta 2 GB por archivo | Válido para cada checkpoint individual sólo si se habilita LFS y se acepta su almacenamiento. |
| GitHub Release assets | Cada asset debe ser menor de 2 GiB | Alternativa para distribuir el checkpoint y TAR sin meterlos en el historial Git. |
| Dataset privado Kaggle | Ya contiene el artefacto canónico | Fuente actual de preservación y entrada para evaluación/reanudación. |

> El repositorio destino `Robbin360/repo-de-ia-de-nueva-generaci-n-` es público. Publicar allí pesos mediante Git LFS o assets de Release los haría accesibles públicamente. Esta acción requiere una confirmación explícita separada.

## Decisión pendiente

La actualización de código y documentación puede prepararse para GitHub. Los pesos Edge **no están presentes en este directorio de trabajo**: la copia canónica está en el dataset privado de Kaggle. Antes de descargarlos o subirlos a GitHub se necesita decidir, de forma explícita, entre mantenerlos privados en Kaggle, habilitar Git LFS o publicarlos como assets de un Release público.

## Fuentes oficiales

1. [GitHub — About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github): GitHub bloquea archivos de Git mayores de 100 MiB.
2. [GitHub — About Git Large File Storage](https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage): GitHub Free admite hasta 2 GB por archivo LFS.
3. [GitHub — About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases): cada asset de un Release debe medir menos de 2 GiB.
