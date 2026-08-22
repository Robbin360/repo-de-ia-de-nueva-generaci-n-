# Topología de ejecución concurrente de Aethel v1

**Estado:** diseño operativo. Los workers persistentes y la ruta GPU comercial no están desplegados en este entorno.

## Separación de ritmos

Aethel no fuerza Sólido, Líquido, curiosidad, memoria y Sueño dentro de un único bucle autoregresivo. La respuesta token a token mantiene una ruta corta y determinista; los procesos lentos consumen eventos versionados fuera de esa ruta. Esta separación conserva latencia, permite auditoría y evita que una observación modifique La Roca durante una conversación.

| Dominio | Ritmo | Ejecutor previsto | Escritura permitida | Prohibición |
|---|---|---|---|---|
| La Roca | Token | GPU / runtime de inferencia | Ninguna durante inferencia. | Auto-modificar pesos base. |
| Memoria de trabajo | Sesión | CPU y proceso de sesión. | Estado acotado de la sesión. | Cruzar sesiones sin política. |
| Espacio de Trabajo Global | Token/sesión | GPU/CPU según ruta. | Telemetría y selección de fuentes. | Convertir trazas en cadena de pensamiento expuesta. |
| El Líquido | Episodio | Servicio CPU/Rust previsto. | Eventos con TTL, versión y procedencia. | Escribir en La Roca. |
| Curiosidad | Episodio | Controlador CPU. | Propuestas locales no ejecutables. | Acciones externas o admisión automática a Sueño. |
| Sueño | Lote aislado | Worker de entrenamiento autorizado. | Candidato LoRA en cuarentena. | Leer holdout o promocionar sin P0–P6. |

## Flujo de mensajes

```text
Solicitud → autorización → recuperación permitida → La Roca/GPU → respuesta
                                │                         │
                                └── telemetría ────────────┘
                                                     ↓
                          El Líquido versionado → Curiosidad local
                                                     ↓
                      cola de candidatos (no elegible por defecto)
                                                     ↓
                   Sueño aislado tras aprobación, preflight y evaluación
```

Cada enlace transporta identificadores de sesión, procedencia, tiempo de expiración, versión de política y hash del activo cuando corresponda. Los sistemas de fondo no están autorizados a llamar servicios externos ni a incorporar conocimiento nuevo desde Internet.

## Garantías de concurrencia

La Roca se lee como referencia inmutable por versión. El Líquido utiliza eventos apendibles y snapshots; el servicio de memoria debe emitir un recibo atómico de restauración. Un candidato de Sueño sólo ve un índice de replay aprobado y `train`; no consume holdout. Si un worker falla, la inferencia se degrada a La Roca sin el servicio opcional y registra el incidente, en vez de inventar un estado cognitivo.
