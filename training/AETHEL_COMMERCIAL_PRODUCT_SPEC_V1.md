# Especificación de producto comercial Aethel v1

## Producto inicial

**Aethel Workspace** es un espacio bilingüe de inteligencia técnica para equipos que necesitan consultar conocimiento privado y aprobado en español e inglés. Su propuesta no es autonomía abierta: combina un modelo propio progresivamente validado, recuperación con fuentes, memoria de sesión controlada y una ruta de adaptación gobernada.

| Componente comercial | Función para el cliente | Límite explícito |
|---|---|---|
| Conversación bilingüe | Explicar, comparar y redactar sobre fuentes autorizadas. | No afirma certeza sin fuente o evaluación. |
| Conocimiento privado | Recuperar documentos permitidos con procedencia. | No mezcla espacios de clientes. |
| Memoria gobernada | Mantener continuidad de sesión y preferencias aprobadas. | TTL, revocación y auditoría; no memoria ilimitada. |
| Aprendizaje controlado | Proponer mejoras LoRA tras revisión. | No cambia producción por sí solo. |
| Operación | Panel de salud, versiones, rollback y métricas. | No presenta métricas inexistentes. |

## Cliente y propuesta de valor

El primer cliente objetivo es un equipo técnico bilingüe con documentación fragmentada: ingeniería, operaciones, cumplimiento técnico, soporte especializado o investigación interna. Aethel debe reducir el tiempo de localizar, comprender y vincular información aprobada, sin obligar al cliente a ceder control sobre sus fuentes o sus cambios de modelo.

## Límites de lanzamiento

El producto no se lanza bajo la afirmación de “cerebro humano”, conciencia, aprendizaje autónomo ilimitado o equivalencia con modelos de frontera. Un piloto requiere un modelo y Dataset propios identificables, evaluación retenida bilingüe, recuperación citable, control de acceso, observabilidad, recuperación ante fallos y una política de soporte. Los valores de coste, latencia y calidad se incorporan cuando existan mediciones reales.

## Criterios de piloto

| Área | Evidencia necesaria |
|---|---|
| Modelo | Checkpoint versionado, licencia revisada y evaluación inglesa/española separada. |
| Datos | Procedencia, derechos de uso, separación de tenants y borrado verificable. |
| Seguridad | Autenticación, autorización y auditoría de accesos. |
| Operación | Observabilidad, backup, rollback y responsable de incidentes. |
| Valor | Casos de uso definidos con resultados del cliente; no testimonios fabricados. |

La primera oferta se limita a un piloto privado y gobernado. La expansión a Edge general o Pro empresarial se decide a partir de métricas y revisión de riesgos, no de promesas de escala.
