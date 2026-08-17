# Fuentes primarias: cómputo adaptativo para Aethel

## Mixture-of-Depths

Raposo et al. proponen restringir a un número fijo `k` los tokens que atraviesan atención y MLP en una capa, eligiendo las identidades mediante routing top-k. El presupuesto total de cómputo permanece conocido aunque cambie por token y capa. El resumen declara desempeño comparable al baseline con FLOPs equivalentes y, en posentrenamiento, hasta 50% más velocidad de muestreo en sus configuraciones evaluadas. Aethel usará esta idea como una **hipótesis a medir**, no como promesa de aceleración.

URL: https://arxiv.org/abs/2404.02258

## Mixture-of-Recursions

Bae et al. combinan reutilización de un bloque recurrente y routing que asigna distintas profundidades de recursión a cada token. El artículo describe una frontera de eficiencia en sus experimentos, pero su diseño completo requeriría reentrenamiento y una implementación adicional. Aethel adopta únicamente el principio de presupuesto explícito y refinamiento condicionado.

URL: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b08bbf8b420faa6eeb4020720582ec7-Abstract-Conference.html

## Refinamiento adaptativo interno

Mathur et al. describen un Transformer que refina internamente estados de atención con cómputo de prueba adaptativo. Su resultado se reporta en benchmarks de encoder y no prueba que funcione en el decodificador autoregresivo de Aethel. Se utiliza como motivación para un módulo residual recurrente, con profundidad limitada y telemetría.

URL: https://arxiv.org/abs/2507.13569

## Criterio de uso en Aethel

Una variante debe mantener compatibilidad de checkpoint cuando el módulo está desactivado, registrar fracción de refinamiento y pasos efectivos, y superar o igualar al baseline en pérdida validada bajo un presupuesto de cómputo acordado. Si aumenta la pérdida o no reduce trabajo medible, se rechaza.
