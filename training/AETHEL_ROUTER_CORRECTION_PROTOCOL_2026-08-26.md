# Protocolo de corrección mínima del router MoE

## Evidencia que motiva la corrección

La corrida directa V1 completó 768 pasos y redujo la pérdida de 10,4948 a 8,1372, pero sólo 43 pasos (5,60 %) cumplieron la salud global del router. La clasificación no proviene de un error del resumidor: cada paso usa la peor entropía de sus cuatro capas y exige entropía mínima de 0,50 junto con desequilibrio máximo de 0,30.

El mínimo de entropía fue 0,333333, el valor que aparece cuando el top-2 se concentra en un único par de expertos. El máximo desequilibrio de 0,1875 quedó bajo el límite; por tanto el bloqueo se debe a concentración repetida de selección, no a una puerta de desequilibrio excesivamente estricta.

## Corrección única

El sesgo adaptativo de balanceo pasa a intervenir solamente en la selección top-k. Los pesos con que se combinan las salidas expertas, la pérdida auxiliar de carga y la regularización de entropía se calculan con los logits crudos del router. Esta separación evita que un controlador sin gradiente modifique la función de mezcla o reduzca la señal de entrenamiento del router.

La decisión se alinea con el enfoque de balanceo sin pérdida auxiliar, donde el sesgo ajusta la asignación y no cambia los pesos de combinación del modelo. No se modifica la arquitectura, el número de expertos, top-2, Dataset v1, pasos, precisión, semilla, checkpoint histórico ni se ajustan umbrales para convertir artificialmente un fallo en éxito.

## Criterios definidos antes de GPU

La corrida correctiva partirá de inicialización fresca y tendrá el mismo presupuesto de 768 pasos. Se clasificará como mejora únicamente si: (a) completa sin OOM ni pérdida no finita; (b) mantiene el checkpoint separado de V1; (c) supera 43 pasos saludables; y (d) no cae por debajo de la pérdida final V1 de 8,1372184753. La promoción sigue bloqueada salvo que la evidencia adicional demuestre estabilidad global, no sólo mejora parcial.

## Límites

La corrección no demuestra por sí misma razonamiento, bilingüismo, matemáticas, ultra-eficiencia comparada, Triton, consolidación completa de Sueño o runtime Rust/Mojo. Es una intervención aislada para restaurar estabilidad MoE antes de escalar.

## Referencias

[1] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. https://arxiv.org/abs/2101.03961

[2] DeepSeek-V3 Technical Report, sección Auxiliary-Loss-Free Load Balancing. https://arxiv.org/abs/2412.19437
