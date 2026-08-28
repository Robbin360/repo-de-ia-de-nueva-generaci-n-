# D1D — Regularización de entropía densa del router MoE

## Estado y frontera

> **Estado: `D1D_LOCAL_PROTOCOL_ONLY`.** D1D es una hipótesis nueva diseñada después del cierre `D1C_ROUTER_NOT_IMPROVED`. Está validada inicialmente sólo por una prueba matemática CPU y todavía no está integrada en el entrenador ni autorizada para Kaggle, notebook, GPU, Dataset, entrenamiento o artefactos.

D1D no reanuda D1A, D1B, D1C ni E0. No carga checkpoints, no abre holdout, no evalúa el corpus fuera de los límites de una futura corrida train-only y no habilita promoción, serving o repetición automática. El protocolo no constituye autorización operativa.

## Motivación

D1C aumentó el peso del término de balanceo auxiliar a `0.05`, pero sólo produjo 67/768 pasos saludables y falló tres de cuatro criterios predefinidos. El término de balanceo existente depende de densidad de tokens y probabilidad agregada; puede ser demasiado indirecto o escaso para corregir rápidamente una distribución suave colapsada.

D1D introduce una señal densa directamente sobre las probabilidades suaves del router **antes de top-k**. Para una fila de probabilidades `p` con `E` expertos, el helper calcula:

\[
L_{ent}(p)=\frac{1}{\log E}\;\frac{1}{T}\sum_{t=1}^{T}\sum_{e=1}^{E}p_{t,e}\log(\max(p_{t,e},10^{-9})).
\]

Esta cantidad es la **entropía negativa normalizada**. Su rango ideal es `[-1, 0]`: una distribución uniforme da `-1`, mientras que una distribución concentrada se aproxima a `0`. Al minimizar `L_ent`, el optimizador recibe una señal que favorece mayor entropía y reduce la concentración. El término no sustituye al balanceo de carga; se suma como regularizador independiente y pequeño.

## Hipótesis falsable

> Manteniendo el baseline de D1C V3-R1 y añadiendo únicamente una regularización densa de entropía con coeficiente `router_entropy_loss_weight=0.01`, la salud del router mejorará materialmente frente a D1C sin producir un deterioro material de la pérdida media.

El valor `0.01` es una primera sonda conservadora, no una constante validada. No se deben barrer varios pesos dentro de la misma corrida ni escoger el resultado favorable después de observar métricas. Un cambio de peso será un experimento posterior separado.

## Diseño controlado

| Control | D1C V3-R1 de referencia | D1D preparado |
|---|---:|---:|
| Inicialización | Nueva | Nueva; sin reanudar |
| Datos | Sólo train; holdout sellado | Sólo train; holdout sellado |
| Pasos / tokens | 768 / 1.572.864 | 768 / 1.572.864 |
| Seed / ventana | 17 / misma ventana | 17 / misma ventana |
| Arquitectura | 512, 4 capas, GQA 8/2, 8 expertos top-2 | Idéntica |
| `router_bias_step` / límite | 0.05 / 0.5 | 0.05 / 0.5 |
| `router_aux_loss_weight` | 0.05 | 0.05 |
| `router_entropy_loss_weight` | 0 | **0.01** |
| Señal nueva | Ninguna | Entropía negativa normalizada sobre `router_probability` densa |
| Runtime | CUDA/PyTorch experimental, fallback aceptado sólo por perfil | Igual, sin activar Triton automáticamente |
| Salida | La salida histórica no se reutiliza | Directorio y nombre de salida inéditos |

La integración debe conservar el gradiente hacia los logits del router y calcular la señal sobre la matriz `[tokens, expertos]` de probabilidades suaves. No es válido calcularla sólo sobre los dos expertos seleccionados por top-k, porque eso elimina precisamente la señal densa contra los expertos ignorados.

## Criterios de clasificación

D1D usará los mismos criterios de salud para conservar comparabilidad: entropía mínima estrictamente superior a `0.5` y desequilibrio máximo inferior a `0.3`. El resultado sólo podrá considerarse **apoyado para revisión documental posterior** si además mejora materialmente D1C en pasos saludables y mantiene la pérdida media dentro de un umbral predefinido antes de ejecutar.

Como regla operativa inicial, se exige al menos `117/768` pasos saludables, entropía mínima `> 0.5`, desequilibrio máximo `< 0.3` y pérdida media no superior a `9.53127756`, que equivale a un máximo de 1 % sobre la media observada de D1C V3-R1 (`9.43690848`). Si falla cualquier criterio, la clasificación será `D1D_ROUTER_NOT_IMPROVED`; no habrá ajuste oportunista ni nueva corrida automática.

Los valores de D1C son una referencia documental compartida por el usuario. D1D no abre ni inspecciona outputs o checkpoints de D1C y no los utiliza como pesos de inicialización.

## Evidencia mínima admisible

La futura corrida debe emitir un resumen determinista con identificador `D1D`, pasos observados, tokens finales, pérdida media, entropía mínima, desequilibrio máximo, pasos saludables, `router_entropy_loss_weight`, seed, release exacto, estado de checkpoint cargado, estado de holdout, uso de red y runtime. El resumen debe confirmar `checkpoint_loaded=false`, `holdout_content_read=false` y `network_requests=0`.

No se considerará evidencia suficiente un gráfico sin contrato, una captura parcial, una métrica agregada sin parámetros o una salida que no demuestre el inicio nuevo y el aislamiento train-only.

## Puertas de autorización

Todas las siguientes puertas permanecen cerradas en el estado actual:

| Puerta | Estado actual |
|---|---|
| Validación matemática CPU | **Abierta**: contrato corregido y prueba pasada |
| Integración en trainer | **Cerrada**: pendiente de revisión y prueba local |
| Bundle V5 de código | **Cerrada**: aún no preparado |
| Subida o nueva versión privada en Kaggle | **Cerrada**: requiere autorización inmediata específica |
| Edición de notebook / nueva CELDA 11 | **Cerrada**: requiere autorización separada |
| Selección o uso de GPU | **Cerrada**: requiere autorización separada |
| Corrida D1D train-only | **Cerrada**: requiere autorización final inmediata |
| Save Version y artefactos | **Cerrada**: requieren autorización propia |
| Holdout, reanudación, promoción y serving | **Cerradas** y fuera del alcance D1D |

La GPU T4 activa en una sesión de Kaggle no equivale por sí sola a autorización para ejecutar D1D. Ninguna acción externa se realiza desde este entorno local.

## Secuencia prevista

Primero se corrige y prueba la expectativa matemática CPU. Después se integra el término en el entrenador, se añade una prueba local de composición de pérdidas y se comprueba que el valor por defecto no cambia cuando el coeficiente es cero. Luego se genera un bundle V5 de código sin corpus, pesos, checkpoints ni métricas crudas. Sólo tras la revisión del bundle se solicitará autorización para subirlo como nueva versión privada.

Después, en una autorización independiente, se preparará la CELDA 11 bloqueada para resolver el release exacto. Una segunda autorización permitirá reemplazarla o abrir sus puertas. Finalmente se solicitará una autorización inmediata y separada para ejecutar exactamente una corrida D1D train-only de 768 pasos, con inicialización nueva y salida inédita. Si el resumen clasifica D1D como no mejorada, la línea se cierra; no se ajusta el peso sin un protocolo nuevo.

## No afirmaciones permitidas

D1D no demuestra todavía que Aethel tenga un modelo funcional, calidad bilingüe, razonamiento, inteligencia general, rendimiento frontier ni capacidad de serving. Incluso una corrida saludable sólo demostraría que una intervención concreta mejoró métricas de routing bajo una configuración experimental y un presupuesto pequeño. La promoción exige evaluaciones y puertas posteriores que permanecen fuera de este protocolo.

## Resultado D1D — corrida train-only real

La corrida autorizada se ejecutó en CUDA con inicio nuevo, seed `17`, `768` pasos y `1.572.864` tokens observados. No cargó checkpoint, no leyó contenido holdout y registró `network_requests=0`. El peso de entropía fue `0.01` y el término histórico auxiliar permaneció en `0.05`.

| Métrica | Criterio | Resultado D1D | Decisión |
|---|---:|---:|---|
| Pasos saludables | `>=117/768` | `52/768` | No cumple |
| Entropía mínima global | `>0.5` | `0.3333333433` | No cumple |
| Desequilibrio máximo global | `<0.3` | `0.1875` | Cumple |
| Pérdida media | `<=9.53127756` | `9.3994066852` | Cumple |
| Último paso: entropía mínima | `>0.5` | `0.5975525975` | Cumple aisladamente |
| Último paso: desequilibrio máximo | `<0.3` | `0.1442871094` | Cumple aisladamente |

La clasificación formal es **`D1D_ROUTER_NOT_IMPROVED`**. El último paso fue saludable, pero no sustituye los criterios definidos sobre la corrida completa: sólo 52 de 768 pasos fueron saludables y la entropía mínima global quedó por debajo de `0.5`. La media de pérdida sí quedó dentro del umbral, por lo que el fracaso se atribuye a estabilidad temporal insuficiente del router, no a un deterioro de la pérdida bajo este protocolo.

La intervención produjo una señal positiva tardía, pero no evidencia suficiente para promover el checkpoint, ejecutar holdout, reanudar, publicar, servir ni declarar un modelo funcional. D1D queda cerrada como experimento no mejorado en su criterio global. Cualquier nueva variante de peso, calendario o regularización requiere un protocolo posterior independiente; no se autoriza ajuste oportunista sobre esta corrida.
