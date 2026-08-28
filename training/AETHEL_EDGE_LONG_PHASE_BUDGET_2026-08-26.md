# Aethel Edge — presupuesto del primer tramo largo

## Objetivo y límites

Esta fase entrena el mismo núcleo Edge de 4 capas, dimensión 512 y MoE de 8 expertos con top-2. No es una promoción ni una afirmación de razonamiento, bilingüismo o eficiencia comparativa. Parte desde inicialización fresca y conserva el perfil de router que alcanzó 446/768 pasos saludables en la sonda jitter.

Kaggle admite como máximo una sesión de 12 horas según el requisito operativo del usuario. La corrida no debe consumir todo ese margen: el empaquetado, la validación de artefactos y el guardado manual de versión requieren tiempo dentro de la misma sesión. Por ello, el límite operativo inicial es **9 horas y 30 minutos**; cualquier terminación debe alcanzar una frontera de actualizador y emitir un checkpoint recuperable.

## Proyección, no garantía

La sonda jitter midió 6.664,31 tokens/s como promedio. A esa tasa, 10 horas equivaldrían aproximadamente a 234.292 micro-pasos de 1.024 tokens. Para proteger contra variación de rendimiento, replay, validaciones y empaquetado, la primera sesión se presupuestará a 5.500 tokens/s durante 9,5 horas: aproximadamente **183.691 micro-pasos** y **188 millones de tokens** procesados.

El número se redondeará hacia abajo al múltiplo de `gradient_accumulation=16`: **183.680 micro-pasos**, 11.480 actualizaciones de AdamW y 188.088.320 tokens procesados. La telemetría actual contabiliza micro-pasos; no se presentarán dichos pasos como actualizaciones del optimizador.

## Horizonte y checkpoints

La fase inicial tendrá un horizonte global inmutable, separado del límite de cada sesión. La primera propuesta es un horizonte de **734.720 micro-pasos** (cuatro sesiones de 183.680), equivalente a 752.353.280 tokens procesados antes de cualquier evaluación de ampliación. La tasa de aprendizaje seguirá ese horizonte global y no se reiniciará al reanudar.

Cada sesión deberá guardar `latest.pt` completo cada 4.000 micro-pasos y al terminar. Cada `latest.pt` incluye pesos, AdamW, scaler, RNG, estado cognitivo mutable, estado de referencia de regularización, contrato de datos y el paso global. Se conservarán tres snapshots portátiles sin optimizador. Antes de que el usuario abandone la sesión, el flujo validará y empaquetará el checkpoint, escribirá la compuerta de guardado y requerirá `Save Version` manual.

## Puertas de continuación

Se detiene o no se reanuda automáticamente si hay pérdida no finita, incompatibilidad de tokenizador/configuración/manifiesto, falta de paquete de preservación o ausencia de checkpoint reanudable. La salud del router se registra por paso, pero una proporción alta de pasos saludables no promociona el modelo. Tras cada sesión preservada se ejecutarán inspección de checkpoint, generación mínima y holdout separado antes de decidir el tramo siguiente.
