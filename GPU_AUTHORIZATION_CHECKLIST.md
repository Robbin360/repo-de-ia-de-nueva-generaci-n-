# Checklist de autorización para una corrida GPU de Aethel

## Principio de control

La preparación técnica **no es autorización**. Ningún agente debe crear una cuenta, aceptar términos, provisionar una GPU, adjuntar almacenamiento, descargar un corpus de pago o iniciar una corrida hasta que el propietario de la cuenta confirme los campos de esta lista en el navegador.

## Puertas obligatorias

| Puerta | Confirmación del propietario | Evidencia que debe conservarse | Estado previo requerido |
|---|---|---|---|
| Cuenta | El usuario inicia sesión en el proveedor elegido con su propia cuenta. | Nombre del proveedor y región visible en la consola. | No se comparte contraseña, token ni datos de pago en el chat. |
| Coste | Se aprueba un máximo de gasto total y una ventana máxima de horas. | Límite de gasto, precio visible y política de apagado. | El precio se vuelve a comprobar en el proveedor; no se usan cifras históricas como compromiso. |
| Hardware | Se aprueba tipo, número y VRAM de GPU. | Topología CUDA real obtenida al iniciar el host. | Para `scale-1b`/FSDP se requieren al menos dos GPU CUDA. |
| Almacenamiento | Se aprueba un volumen persistente y una copia externa de artefactos. | Ruta de `AETHEL_RUN_DIR` y destino de exportación. | Nunca depender sólo del disco efímero de la instancia. |
| Datos | Se aprueba el manifiesto, revisión de licencia, revisión de PII y máximo de documentos. | Revisión fijada, hash de manifiesto y `AETHEL_MAX_DOCUMENTS`. | Todas las fuentes comienzan deshabilitadas en el manifiesto. |
| Modelo | Se aprueba el preset inicial y el límite de pasos/tokens. | Configuración de launcher y hash de checkpoint. | ARC sigue desactivado: el experimento local no superó baseline. |
| Evaluación | Se aprueban holdout y puertas de parada. | Protocolo, tokenizador, semilla y ruta de reportes. | No se muestran scores sin predicciones reales. |

## Orden de ejecución autorizado

Una vez aprobadas las puertas anteriores, el operador sigue esta secuencia. Cada paso se detiene ante un fallo; no hay reintento automático de coste.

```bash
# 1. En el host GPU autorizado, verificar preparación de datos y evaluación sin descargar nada.
python3 training/validate_training_readiness.py \
  --output training/experiments/training_readiness.json

# 2. Con el repositorio en el checkpoint aprobado, validar la topología GPU.
bash training/run_gpu_preflight.sh

# 3. Sólo si el preflight devuelve VERIFIED: definir rutas persistentes y límite de ingesta.
export AETHEL_DATA_DIR=/ruta/persistente/datos-aprobados
export AETHEL_RUN_DIR=/ruta/persistente/aethel-runs
export AETHEL_EVALUATION_CONFIG=/ruta/persistente/evaluation_plan.json
export AETHEL_MAX_DOCUMENTS=<máximo-aprobado>

# 4. Arrancar el piloto documentado y reanudable.
bash training/run_aethel_gpu.sh
```

El launcher ejecuta el validador en modo `--require-approved` antes de instalar dependencias o descargar corpus. Ese modo exige revisiones inmutables, aprobación y habilitación explícita de cada fuente, además de un plan de evaluación aprobado. El preflight exige dos GPU CUDA para la validación FSDP; en un host sin esa topología devuelve `SKIPPED`, que **no** autoriza una corrida multi-GPU. Para el primer piloto de una sola GPU se conserva el mismo manifiesto, almacenamiento persistente y procedimiento de exportación, pero no se declara FSDP validado.

## Criterios de parada

La corrida se pausa si falla el preflight, se supera el límite aprobado, no se puede exportar un checkpoint atómico, el manifiesto no tiene revisión/licencia aprobada, el router MoE pierde salud de forma sostenida o se detecta regresión en la evaluación congelada. Reanudar sólo ocurre desde un checkpoint persistente revisado.

## Datos que el usuario debe confirmar, no publicar

Cuando esté listo para el paso de cuenta, confirme por mensaje el proveedor, el límite máximo total, la ventana de horas y la familia objetivo. El acceso a la consola y cualquier aprovisionamiento se solicitarán mediante confirmación explícita en el navegador.
