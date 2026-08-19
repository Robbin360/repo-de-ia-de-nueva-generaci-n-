# Observaciones de ejecución Kaggle

- El cuaderno `aethel-nextgen-bilingual-pilot` está en modo edición con una sesión Draft activa.
- La celda corregida muestra `project_gutenberg` en la salida de copias descartadas y llama al launcher al final.
- Se ejecutó únicamente la celda actual mediante `Run current cell`; no se pulsó `Run All` ni `Save Version`.
- Durante las revisiones, el botón permaneció como `Cancel Run`, por lo que la ejecución seguía activa.
- El uso de output pasó aproximadamente de 778.8 MiB a 891.1 MiB. No se observó todavía la línea final de selección ni la puerta de conteo en la vista disponible.
- No se debe iniciar otra ejecución ni cancelar la actual hasta obtener la salida final.

## Reintento con V10 y selección verificada

- La celda V10 fue pegada en el cuaderno y se ejecutó tras autorización explícita, sin usar `Run All` ni `Save Version`.
- Kaggle detectó tres bundles y seleccionó `/kaggle/input/datasets/felixtremigual/aethel-nextgen-source/oxvQlQKMIBZuWdXS`.
- La salida verificó `project_gutenberg=True` y `olc_pd_books=True` para el bundle elegido, mientras descartó `pKXqovDfRmpcogEs` y `FtHvAkAfWJNYrbtR` con `project_gutenberg=False`.
- La sesión permanece en `Draft Session Running` con `Cancel Run` visible. La consola no ha publicado aún conteos por idioma ni una excepción posterior a la selección.
- El panel de salida contiene `/kaggle/working` expandible; no se realizó ninguna acción destructiva sobre los artefactos.

## Recarga del editor durante la preparación

- Tras una recarga de la vista, Kaggle mostró `Your Notebook is now running in the cloud`, seguido de `Editor Loading…` y `Session is starting…`.
- La recarga eliminó temporalmente del panel la salida anterior, pero no se pulsó `Run All`, `Save Version` ni una segunda ejecución.
- La observación pendiente sigue siendo la puerta real de conteos; no se infiere éxito de preparación o entrenamiento a partir del estado de la interfaz.

## Estado más reciente

- Tras la recarga, Kaggle muestra `Draft Session Starting…` y el botón `Run All` volvió a estar habilitado; no se pulsó.
- El cuaderno y el Dataset privado `aethel-nextgen-source` siguen presentes, pero la consola sólo muestra el aviso genérico de notebook en la nube y no conserva la salida anterior en la vista.
- No se puede afirmar que la preparación continúe ni que haya finalizado hasta que la sesión termine de inicializar y publique nuevo estado o registros.

## Reintentos de fuentes inglesas

- La consola de la sesión recuperada mostró reintentos reales de `olc-pd-books-en` y `project-gutenberg-en` después de HTTP 502, ambos con espera inicial de 4,0 s.
- En el siguiente minuto de observación no apareció ni un conteo final ni una excepción terminal; tampoco se ejecutó ninguna acción de interrupción, reinicio, guardado o nueva ejecución.
