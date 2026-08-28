# Oracle Cloud Free Tier y Aethel — verificación de infraestructura

**Fecha de consulta:** 26 de agosto de 2026. **Alcance:** determinar si el nivel gratuito de Oracle Cloud Infrastructure puede entrenar Aethel Edge o aportar almacenamiento persistente. Esta nota no crea una cuenta, no solicita recursos y no autoriza gasto.

## Hallazgos oficiales

Oracle distingue entre un crédito promocional de **US$300 válido hasta 30 días** y los recursos **Always Free** que permanecen disponibles dentro de sus límites. [1] [2]

| Recurso Always Free | Límite oficial vigente | Lectura para Aethel |
|---|---:|---|
| Cómputo Arm Ampere A1 | 2 OCPU y 12 GB RAM totales; sólo en región de origen | No contiene GPU; útil para orquestación, validación CPU pequeña o utilidades de preservación, no para entrenar el Edge de ~97 M parámetros. |
| Micro AMD | Hasta dos VMs de 1/8 OCPU y 1 GB RAM | No apto para entrenamiento de modelos. |
| Block Volume | 200 GB totales en región de origen | Puede alojar código, manifiestos y artefactos pequeños/medianos, sujeto a la cuota total. |
| Object Storage | 20 GB totales | Insuficiente como estrategia única si los checkpoints y corpus crecen; podría servir a recibos y metadatos. |

Las fuentes oficiales también advierten que las instancias Always Free inactivas pueden ser reclamadas y que la disponibilidad de formas gratuitas puede faltar temporalmente en una región. [1]

## Decisión técnica

El nivel **Always Free no proporciona GPU**, por lo que no es una ruta para entrenar el primer Edge largo ni el futuro Pro. El crédito promocional puede permitir probar servicios de GPU elegibles durante la ventana de 30 días, pero se consume, es temporal y no equivale a entrenamiento gratuito continuo. Antes de cualquier uso de crédito o instancia pagada habría que comparar GPU concreta, VRAM, región, cuotas y coste, con autorización separada del usuario.

Para el flujo actual, Kaggle sigue siendo la alternativa de cómputo para la fase Edge, y su salida debe guardarse manualmente como Dataset privado al final. Oracle Always Free podría considerarse después como host auxiliar de herramientas ligeras; no sustituye la preservación de checkpoints ni resuelve por sí mismo la falta de GPU.

## Referencias

[1] [Oracle Cloud Infrastructure — Always Free Resources](https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), actualizado el 12 de junio de 2026.

[2] [Oracle Cloud Infrastructure — Free Tier](https://docs.oracle.com/iaas/Content/FreeTier/freetier.htm), actualizado el 29 de junio de 2026.

[3] [Oracle — Cloud Free Tier](https://www.oracle.com/cloud/free/), consultado el 26 de agosto de 2026.
