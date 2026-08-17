# Servicio Rust de memoria Aethel — protocolo operativo

El binario `aethel-memory-rust` es un **proceso de protocolo JSONL**: recibe un objeto JSON por línea en `stdin` y devuelve una respuesta JSON por línea en `stdout`. De esta forma, un supervisor de producción puede mantener el proceso aislado de la interfaz y observar cada transición. El componente ha sido compilado y probado localmente; **no está desplegado como daemon 24/7**.

También admite `--socket /run/aethel-memory/memory.sock` para atender el mismo JSONL a través de un socket Unix local. No abre un puerto de red público. La unidad `deploy/aethel-memory.service` aporta un ejemplo de supervisión con reinicio, usuario dedicado, estado privado y permisos restrictivos; es una **plantilla no desplegada**, por lo que no debe interpretarse como una instancia 24/7 activa.

| Operación | Entrada mínima | Resultado | Persistencia |
|---|---|---|---|
| `health` | `{ "op": "health" }` | Estado `READY`, cantidad de registros y bandera de snapshot. | No modifica datos. |
| `remember` | `record` con `id`, `session_id`, vector, saliencia, hash y paso. | Inserta un recuerdo o informa el registro expulsado. | Publica JSONL mediante renombrado atómico. |
| `retrieve` | Vector de consulta y `top_k`. | Recuerdos ordenados por similitud coseno × saliencia. | Sólo lectura. |
| `retrieve_context` | Vector, `top_k` y `max_chars`. | Fragmentos de contexto con `id`, SHA-256, URI opcional y puntuaciones de recuperación. | Sólo lectura. |
| `sleep` | Límite de candidatos de replay. | Reporte de retención y candidatos de consolidación. | Publica el estado consolidado. |
| `snapshot` | Ninguna. | JSONL actual para auditoría o exportación. | Sólo lectura. |

## Invariantes

Cada registro debe tener una dimensión vectorial estable, valores finitos, `salience` en `[0, 1]`, identidad de sesión y `source_sha256`. El contenido recuperable es opcional, pero si existe tiene URI no vacía cuando se declara, entre 1 y 32,768 caracteres y siempre devuelve su hash de origen. Estas restricciones evitan que el servicio trate contenido sin procedencia como memoria recuperable. La capacidad es acotada y la inserción expulsa el registro más antiguo cuando está llena; el ciclo de sueño reordena por saliencia y paso antes de preservar el límite configurado.

## Requisitos antes de operación continua

Para ejecutar este proceso continuamente se necesita un supervisor que reinicie fallos, aplique límites de CPU y memoria, capture logs estructurados y desencadene `sleep` a una frecuencia definida. El snapshot local es suficiente para pruebas; un despliegue persistente debe sustituirlo por almacenamiento duradero con copias de seguridad, cifrado y una política de retención aprobada. El dashboard no debe iniciar este proceso ni afirmar que está activo hasta que el supervisor publique una señal de salud real.

La plantilla de unidad presupone un host persistente autorizado, el binario en `/opt/aethel-memory/aethel-memory-rust`, el usuario/grupo de sistema `aethel` y rutas locales `/var/lib/aethel-memory` y `/run/aethel-memory`. No se instala ni habilita automáticamente desde este proyecto.

> Rust elimina categorías de fallos de memoria mediante sus comprobaciones de propiedad, pero no sustituye la supervisión, las pruebas de recuperación ni la observabilidad de un servicio de larga duración.[1]

## Referencias

[1] [The Rust Programming Language — Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
