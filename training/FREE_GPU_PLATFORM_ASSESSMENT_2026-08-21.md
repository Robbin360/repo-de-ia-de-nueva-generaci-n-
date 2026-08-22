# Evaluación final de plataformas gratuitas de GPU para Aethel

**Autor:** Manus AI  
**Fecha de verificación:** 21 de agosto de 2026  
**Estado:** Decisión de infraestructura para piloto; no autoriza entrenamiento, consumo de cuota ni gasto.

## Decisión

**Kaggle Notebooks** es la plataforma gratuita recomendada para el primer piloto real de Aethel, denominado **Aethel Seed**. Kaggle documenta acceso sin coste a GPU NVIDIA Tesla P100 y una cuota semanal de 30 horas, que puede variar según la demanda y los recursos.[1] La arquitectura y el Dataset ya preparados permiten quitar del camino que causó los errores anteriores: durante el entrenamiento no se descargará corpus desde Internet.

**Google Colab Free** se conserva como alternativa de recuperación, diagnóstico o prueba corta. Google confirma que el acceso a GPU/TPU gratuito no está garantizado, que el tipo de hardware y límites cambian dinámicamente y que la duración máxima de un notebook gratuito puede llegar a 12 horas dependiendo de disponibilidad y patrón de uso.[2]

**Hugging Face ZeroGPU** no se selecciona para entrenar Aethel. Su diseño asigna GPU de manera temporal a una función de una Space; para una cuenta gratuita la cuota incluida es de cinco minutos diarios y las funciones tienen 60 segundos de GPU por defecto.[3] Sí puede servir más adelante para una demostración breve de inferencia, no para optimización de pesos ni reanudación de entrenamiento.

| Plataforma | Acceso gratuito documentado | Tiempo y disponibilidad | Uso aprobado para Aethel |
|---|---|---|---|
| **Kaggle Notebooks** | GPU Tesla P100; cuota de 30 h por semana, sujeta a demanda.[1] | Sesiones temporales y cuota semanal. | **Piloto Aethel Seed**, preflight CUDA/Triton y pruebas breves con checkpoints. |
| **Google Colab Free** | Recursos gratuitos, incluidos GPU/TPU, sin hardware garantizado.[2] | Hasta 12 h en función de disponibilidad y uso; puede terminar por inactividad. | Respaldo de diagnóstico o smoke run reanudable. |
| **Hugging Face ZeroGPU** | GPU compartida por llamada para Spaces; cinco minutos diarios en cuenta gratuita.[3] | 60 s por función por defecto; cola y cuota diaria. | Sólo demo o inferencia corta con pesos ya entrenados. |
| **Google Cloud trial** | Crédito promocional de 300 USD durante 90 días para clientes nuevos.[4] | No es una oferta de GPU gratuita permanente; requiere gestión de cuenta y facturación. | Alternativa posterior, únicamente con autorización explícita. |

## Lección de los fallos anteriores

Los errores observados en Kaggle ocurrieron antes del entrenamiento: HTTP 429/502, respuestas truncadas y fuentes con conteos insuficientes. El problema fue intentar **construir el corpus desde la red dentro de una sesión temporal**. Ese flujo queda descartado.

El flujo permitido parte del paquete local congelado `aethel-knowledge-corpus-v1`: 40.000 documentos de Wikipedia con procedencia, hashes, deduplicación, partición train/holdout y tokenizador BPE entrenado sólo con train. Antes de activar GPU se debe adjuntar como Dataset privado, verificar el manifiesto offline y bloquear cualquier solicitud de red asociada a datos. Durante la sesión sólo se puede cargar, entrenar, evaluar el holdout y persistir artefactos.

> **No se considerará un entrenamiento válido si descarga o reconstruye fuentes de datos durante la sesión GPU.**

## Política de almacenamiento y checkpoints

La GPU y el disco de un notebook temporal no son almacenamiento duradero. Cada checkpoint de Aethel debe contener pesos, estado del optimizador, scheduler, estado aleatorio, tokenizador, manifiesto de Dataset y métricas. Debe escribirse de forma atómica, hashable y verificarse antes de continuar o terminar un bloque.

| Plataforma | Riesgo de almacenamiento | Destino de checkpoint requerido | Recuperación aceptable |
|---|---|---|---|
| **Kaggle** | El workspace de sesión es temporal y no es la única copia válida.[1] | Artefacto/versionado privado accesible por la siguiente sesión, junto con copia local de manifiesto y hashes. | Montar el último artefacto aprobado, verificar hashes y retomar desde el `global_step` confirmado. |
| **Colab Free** | El runtime puede terminar por inactividad o límite de uso.[2] | Almacenamiento externo controlado por el usuario, por ejemplo Drive o un bucket autorizado; mantener copia descargable de emergencia. | Cargar sólo checkpoint con manifiesto coincidente y evidencia de escritura completa. |
| **ZeroGPU** | GPU efímera por función; no está orientado a jobs largos.[3] | No se usarán checkpoints de entrenamiento en este servicio. | No aplica. |
| **Nube con crédito** | El volumen puede persistir, pero depende de la cuenta y del coste posterior al crédito. | Volumen persistente más object storage versionado y controlado. | Restaurar el último checkpoint atómico aprobado antes de volver a lanzar workers. |

La documentación de Kaggle aconseja no usar sesiones batch/commit como mecanismo de checkpoint, ya que ejecutan el notebook completo y son menos eficientes.[1] En Aethel, el commit sirve para conservar el notebook y sus artefactos de salida; la recuperación del entrenamiento depende de checkpoints intermedios guardados y verificados por el runner.

## Secuencia de ejecución propuesta

La plataforma gratuita no se utilizará para intentar Aethel Edge de 2,2 B parámetros. El objetivo inmediato es una evidencia técnica limpia y cuantificable de Aethel Seed.

| Paso | Acción | Evidencia mínima |
|---|---|---|
| 1 | Adjuntar el Dataset privado congelado y validar sin red. | Manifiesto, hashes y reporte `valid=true`. |
| 2 | Ejecutar el preflight GPU. | Dispositivo real, CUDA disponible, precisión seleccionada y contratos Triton/FSDP correctamente bloqueados o ejecutados. |
| 3 | Entrenar Aethel Seed por bloques cortos. | Checkpoints atómicos, JSONL de pérdida y telemetría de router, sin datos simulados. |
| 4 | Evaluar sólo el holdout congelado. | Predicciones, script de evaluación y métricas reproducibles. |
| 5 | Detener y reanudar en otra sesión. | Continuidad de `global_step`, hashes y estado del optimizador. |

El modelo sólo podrá avanzar a Aethel Edge cuando Seed demuestre entrenamiento, retención, recuperación y evaluación reales. La plataforma gratuita mitiga riesgo de ingeniería; no sustituye una infraestructura persistente para un producto comercial.

## Referencias

[1] [Kaggle, *Efficient GPU Usage Tips*](https://www.kaggle.com/docs/efficient-gpu-usage)  
[2] [Google Colab, *Frequently Asked Questions*](https://research.google.com/colaboratory/faq.html)  
[3] [Hugging Face, *Spaces ZeroGPU: Dynamic GPU Allocation for Spaces*](https://huggingface.co/docs/hub/en/spaces-zerogpu)  
[4] [Google Cloud, *Free Trial and Free Tier Services and Products*](https://cloud.google.com/free)
