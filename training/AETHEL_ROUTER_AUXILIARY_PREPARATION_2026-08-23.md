# Preparación local del control auxiliar MoE — 23 de agosto de 2026

## Alcance y estado

Este documento registra una preparación **local y no ejecutada**. No abre corpus, shards, holdout, outputs ni checkpoints; tampoco selecciona GPU, actualiza un Dataset, modifica un notebook, usa Kaggle, inicia una corrida, evalúa, promueve o sirve un modelo.

Los diagnósticos D1A y D1B conservaron la misma contribución auxiliar del router: el núcleo sumaba `0.01 * aux_loss` a la pérdida de lenguaje. Por tanto, la comparación D1A/D1B sólo informa el cambio de `router_bias_step`; no mide el efecto de ese peso auxiliar ni permite concluir que deba cambiarse.

## Preparación A2

La constante histórica se hizo explícita como `router_aux_loss_weight` en `NextGenConfig` y en la interfaz del entrenador. El valor predeterminado sigue siendo **0.01**, por lo que esta preparación no cambia D1A, D1B ni una futura corrida que omita el argumento. `engine/router_auxiliary.py` valida que el peso sea finito y no negativo, y aplica la suma de forma que conserva la operación tensorial del núcleo. `training/test_router_auxiliary_contract.py` verifica el valor histórico, el caso cero explícito y los rechazos de configuración inválida sin usar PyTorch ni datos protegidos.

| Hecho local | Implicación permitida | Implicación prohibida |
|---|---|---|
| La ponderación 0.01 está parametrizada y cubierta por prueba determinista. | Una hipótesis futura puede nombrar el parámetro con exactitud. | No demuestra que un peso distinto mejore el router. |
| D1A y D1B usaron 0.01 de forma constante. | Su diferencia no identifica el efecto de la pérdida auxiliar. | No convierte D1B en base para elegir un valor nuevo. |
| El contrato rechaza pesos negativos, no finitos o ambiguos. | Evita una configuración silenciosamente inválida. | No autoriza código fuente externo, GPU ni una ejecución. |

## Puerta posterior

Antes de proponer siquiera una nueva corrida, debe existir una revisión documental independiente que formule una sola hipótesis falsable, fije un valor concreto distinto de 0.01, conserve los restantes controles comparables y describa el criterio de descarte. Esa revisión requerirá autorización separada para cualquier modificación de release, Dataset de código, notebook, GPU o ejecución. D2, D3, holdout, outputs/checkpoints, promoción y serving siguen bloqueados.

## A3 — Dirección de la señal auxiliar

**A3 está completada sólo en local.** La fórmula `n_experts * sum(density * probability)` se extrajo a `router_balance_auxiliary_loss`. Una prueba CPU con tensores sintéticos deterministas verifica que, con densidad completamente concentrada, el gradiente descendente reduce el logit del experto dominante y eleva el del otro; con densidad uniforme, la señal no prefiere ningún experto. Esta es una comprobación algebraica del signo de la señal, no evidencia de que su magnitud histórica 0.01 sea suficiente ni de que una modificación mejore D1A/D1B.
