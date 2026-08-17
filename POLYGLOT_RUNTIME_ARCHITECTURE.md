# Aethel — Arquitectura políglota objetivo

## Decisión de capas

| Capa | Tecnología objetivo | Estado verificable | Contrato de entrada/salida |
|---|---|---|---|
| Interfaz y conexión | TypeScript, Node.js, React y Tailwind CSS | Implementada en el dashboard. | tRPC/JSON con sesiones, mensajes y telemetría. |
| Laboratorio matemático | Python y PyTorch | Implementado para topología, entrenamiento y exportación. | Corpus/tokenizador → checkpoint `safetensors` + manifiesto. |
| Kernels GPU | Triton + CUDA | Puente implementado; exige CUDA+Triton en el runner GPU por defecto. | Tensores CUDA contiguos → tensores CUDA; equivalencia numérica previa a activación. |
| Inferencia local | Mojo | Contrato diseñado; runtime no instalado ni validado. | Pesos exportados + tokens → logits/token siguiente. |
| Córtex y memoria 24/7 | Rust + Candle | Núcleo JSONL de memoria local compilado y probado; aún no hay un daemon desplegado ni Candle integrado. | Eventos de conversación → embeddings/metadatos → recuperación y consolidación. |

> Python se reserva para investigación, entrenamiento y exportación. La inferencia de usuario y las memorias persistentes no se declararán migradas hasta que Mojo y Rust ejecuten sus pruebas de interoperabilidad.

## Contratos que deben permanecer estables

El artefacto de modelo exportado debe llevar pesos, versión del tokenizador, configuración, licencia de corpus y SHA-256. El runtime de inferencia local debe aceptar `prefill(tokens)` y `decode(token, kv_cache)`, devolviendo logits, caché y telemetría de tiempo. El servicio Rust debe almacenar registros con identificador de sesión, vector, fuente, fecha, política de retención y versión de consolidación; no debe guardar ni recuperar una "memoria" sin trazabilidad.

## Regla de Triton

La ejecución en GPU de Aethel exige Triton por defecto. El puente actual fusiona SwiGLU y aborta si se solicita producción CUDA sin Triton. Triton es un lenguaje y compilador de programación paralela orientado a kernels DNN personalizados, y su modelo por bloques está diseñado para favorecer localidad y paralelismo.[1] [2] Las siguientes rutas sólo podrán activarse tras pruebas de equivalencia y benchmark en la GPU objetivo: atención causal/FlashAttention, selección y dispersión MoE, y actualización de El Líquido. El fallback PyTorch sólo existe mediante una opción explícita de laboratorio, nunca como sustituto silencioso de producción.

## Activación futura

La portabilidad de pesos a Mojo y Candle exige una prueba de igualdad de logits frente al checkpoint PyTorch, una prueba de generación con KV-cache, y una medición por hardware de tokens/s, latencia p50/p95 y memoria. El núcleo Rust ya admite `health`, `remember`, `retrieve`, `sleep` y `snapshot` en JSONL, con publicación atómica de snapshots y restauración local. Para operación 24/7 todavía requiere un supervisor persistente, almacenamiento S3 o base vectorial autorizada, métricas y una política explícita de retención. Rust aporta comprobaciones de propiedad en compilación, pero no permite afirmar por sí solo que un servicio "jamás" fallará; se requiere además supervisión, límites de recursos, backups y pruebas de recuperación.[3]

## Referencias

[1] [Triton Documentation](https://triton-lang.org/)

[2] [Triton Programming Guide — Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)

[3] [The Rust Programming Language — Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
