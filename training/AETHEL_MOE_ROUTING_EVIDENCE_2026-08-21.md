# Evidencia externa para routing MoE top-2

Este registro conserva fuentes consultadas para contextualizar la implementación de routing top-2 de Aethel Pro. No constituye una validación del modelo Aethel ni de sus kernels.

| Fuente | Aporte usado | URL |
|---|---|---|
| Google Research, *Mixture-of-Experts with Expert Choice Routing* | Compara GShard top-2 con otros enfoques de routing y trata el balance de expertos como problema de sistema. | https://research.google/blog/mixture-of-experts-with-expert-choice-routing/ |
| Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* | Describe capacidad por experto y pérdida auxiliar de balanceo en un Transformer sparse. | https://www.jmlr.org/papers/v23/21-0998.html |
| Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* | Fuente del patrón de gating top-2 al que se hace referencia. | https://arxiv.org/abs/2006.16668 |

La implementación local actual usa top-2, una pérdida auxiliar de balanceo y un sesgo de router lento. El dispatch/combinación GPU con Triton sigue bloqueado hasta tener una ruta completa y validada.
