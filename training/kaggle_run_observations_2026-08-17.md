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
