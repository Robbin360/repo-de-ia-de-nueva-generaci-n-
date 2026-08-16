# Diseño técnico de Aethel NextGen

## Hallazgos de fuentes primarias

La capa Sparsely-Gated Mixture-of-Experts activa una combinación escasa de subredes por entrada mediante una red de gating entrenable. La contribución central es separar capacidad total de cómputo activo, permitiendo aumentar parámetros sin aumentar proporcionalmente el coste por token. La fuente advierte que el enrutamiento introduce retos algorítmicos y de rendimiento, por lo que Aethel deberá registrar balanceo de expertos, capacidad y tokens descartados, no solo el número nominal de expertos.

Fuente: [Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538).

## Decisiones de arquitectura

Aethel NextGen se implementará como un sistema híbrido y medible, no como una afirmación de consciencia o cerebro humano. El núcleo será un Transformer eficiente con RoPE y GQA, bloques Sparse MoE con balanceo explícito, y un estado recurrente ligero para contexto prolongado. La memoria se separará en memoria de trabajo acotada, memoria episódica recuperable y memoria semántica consolidada. El aprendizaje continuo tendrá replay de ejemplos, regularización de cambios y evaluación de regresión antes de promover nuevos pesos.

La adaptación durante uso se limitará inicialmente a memoria externa, preferencias y adaptadores pequeños versionados. No se actualizarán los pesos base de forma automática a partir de cualquier conversación: esa ruta puede provocar olvido catastrófico, contaminación de datos y degradación silenciosa. Cada cambio deberá quedar asociado a un checkpoint, una métrica y una posibilidad de rollback.

## Límites de escala

Un modelo pequeño entrenado en CPU no puede competir honestamente con modelos frontier en conocimiento general sin datos, cómputo, memoria y evaluación comparables. El objetivo verificable de esta iteración será lograr una arquitectura eficiente, reproducible y extensible; las comparaciones mostrarán resultados únicamente cuando provengan de ejecuciones reales.

## Referencias

[1]: https://arxiv.org/abs/1701.06538 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
