# Contrato V1 de estados y autoridades de Sueño

**Estado:** implementado y probado por CPU. La máquina registra autorizaciones; no ejecuta entrenamiento ni promoción por sí misma.

## Estados

| Estado | Significado | Puede iniciar ajuste |
|---|---|---:|
| `quarantined` | Candidato nuevo y bloqueado. | No. |
| `preflight_pass` | Artefactos compatibles; sigue bloqueado. | No. |
| `authorized` | Existe aprobación de ejecución verificable. | No; sólo habilita la transición de runtime. |
| `running` | Estado reservado para una ejecución externa autorizada. | No lo ejecuta esta máquina. |
| `evaluated` | Hay evidencia de evaluación externa. | No. |
| `promotable` | Cumplió puertas técnicas, pendiente de aprobación humana. | No. |
| `promoted` | Versión aprobada explícitamente. | No cambia La Roca por esta máquina. |
| `rejected` / `rolled_back` | Resultado descartado o reversión terminada. | No. |

## Transiciones críticas

La única ruta de promoción es `quarantined → preflight_pass → authorized → running → evaluated → promotable → promoted`. Cada flecha exige una autoridad específica y una huella de evidencia SHA-256. Un salto, una autoridad distinta o una huella inválida se rechazan.

| Transición | Autoridad requerida |
|---|---|
| `quarantined → preflight_pass` | `preflight-verifier` |
| `preflight_pass → authorized` | `human-execution-approver` |
| `authorized → running` | `runtime-executor` |
| `running → evaluated` | `evaluation-runner` |
| `evaluated → promotable` | `evaluation-reviewer` |
| `promotable → promoted` | `human-promotion-approver` |
| Cualquier `→ rolled_back` permitido | `rollback-operator` |

Cada evento está hash-encadenado al anterior. Por ahora la identidad de autoridad es un contrato de interfaz comprobado por CPU; una integración de producción deberá unir esas etiquetas a controles de acceso y registros de identidad reales.
