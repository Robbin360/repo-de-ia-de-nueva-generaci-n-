# Contrato V1 de preflight de Sueño

**Estado:** validado por CPU. Esta puerta comprueba manifiestos; no abre los shards del corpus, no crea un optimizador, no realiza ajuste LoRA y no habilita GPU.

## Entradas y condición de paso

El preflight recibe cuatro artefactos versionados: manifiesto de La Roca, manifiesto del candidato LoRA aislado, manifiesto de replay en cuarentena y manifiesto del Dataset offline. El hash de La Roca debe coincidir con el padre declarado por el candidato y por replay.

| Entrada | Comprobaciones | Rechazo si |
|---|---|---|
| La Roca | Es una referencia sin LoRA y tiene SHA-256 válido. | No es referencia estable. |
| Candidato | Conserva el hash base de La Roca; no inició entrenamiento, optimizador, promoción, holdout ni acciones externas. | La rama se desligó de La Roca o habilitó una capacidad prohibida. |
| Replay | Tiene hash íntegro, aprobaciones ya verificadas y todos los permisos de ajuste en `false`. | Fue modificado, contiene duplicados o toca holdout. |
| Dataset | Es offline, tiene train/holdout bilingües y tokenizador derivado sólo de `train`. | Falta un split, el tokenizer mezcla holdout o falta una huella. |

## Salida

Una pasada correcta produce `quarantined_preflight_pass`, no “entrenamiento autorizado”. La salida conserva `eligible_for_training=false`, `eligible_for_promotion=false`, `optimizer_creation_enabled=false`, `holdout_access_enabled=false` y `requires_runtime_authorization=true`.

> El preflight sólo prueba que los manifiestos son compatibles y que sus bloqueos declarados permanecen activos. Antes de un ajuste real deberán verificarse de nuevo los hashes de archivos, el presupuesto, las credenciales de autorización y la política de ejecución.

## Integración de estado

Un reporte íntegro puede registrar **solamente** la transición `quarantined → preflight_pass` del candidato identificado por ese mismo reporte. El enlace comprueba hash del reporte, `candidate_id` y hash de La Roca; no admite estados posteriores, no crea un optimizador y no inicia runtime.
