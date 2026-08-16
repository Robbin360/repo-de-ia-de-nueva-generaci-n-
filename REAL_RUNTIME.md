# Ejecución real de Aethel

Aethel no genera telemetría sintética. El chat usa el LLM real integrado en el servidor. El entrenador lanza `engine/train_real.py`, que importa el modelo PyTorch de `engine/aethel_model.py`, ejecuta optimización real y emite eventos JSON de pérdida, tokens, dispositivo y VRAM.

## Requisitos del motor

| Requisito | Uso |
|---|---|
| Python 3.10 o superior | Ejecutar el proceso de entrenamiento |
| PyTorch | Construir el Transformer Aethel y optimizar pesos |
| Memoria suficiente | Depende de `dim`, `capas`, `expertos` y `pasos` |
| GPU CUDA opcional | Acelerar el entrenamiento y reportar VRAM real |

En un entorno Node-only sin Python o PyTorch, el endpoint devuelve un error explícito y el panel muestra `NOT_CONNECTED`; no se sustituyen las métricas por valores inventados. Los checkpoints se escriben en `engine/artifacts/` cuando el proceso completa correctamente.

## Endpoints operativos

`training.start` inicia el proceso PyTorch con los hiperparámetros configurados. `training.status` devuelve únicamente los eventos emitidos por ese proceso. `engine.status` expone telemetría únicamente mientras existe un job activo. El chat `chat.send` usa el LLM real y persiste cada turno en la tabla `aethel_chat_messages`.

Los benchmarks muestran los nombres de Aethel, GPT-4, Llama y Mixtral, junto con MMLU, HumanEval y GSM8K, pero las puntuaciones permanecen en blanco hasta que se carguen resultados verificables de ejecuciones reales.
