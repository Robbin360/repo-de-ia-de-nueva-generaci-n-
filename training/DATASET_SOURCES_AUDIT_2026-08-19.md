# Auditoría de fuentes para el Dataset Aethel NextGen

## Fuentes verificadas antes de adquisición

| Fuente | Uso previsto | Hallazgo verificable | URL |
|---|---|---|---|
| Wikimedia Downloads | Conocimiento general bilingüe y proyectos educativos | Wikimedia publica copias completas de wikis públicas como contenido y metadatos; aplica una política de User-Agent y límite de conexiones. | https://dumps.wikimedia.org/ |
| Meta-Wiki: Data dumps | Contrato técnico de exportaciones | La documentación define exportaciones de contenido y archivos `pages-articles` con texto de revisiones actuales de páginas de artículos. | https://meta.wikimedia.org/wiki/Data_dumps/What%27s_available_for_download |
| Project Gutenberg | Literatura y prosa inglesa de dominio público | La política oficial se revisará por archivo y procedencia antes de incluir textos; no se adoptará contenido por mera disponibilidad de API. | https://www.gutenberg.org/policy/collection-development-policy.html |
| Kaikki / Wiktionary | Léxico, definiciones y ejemplos bilingües | La inclusión requiere descarga fuente, validación de campos y revisión de la licencia declarada en el artefacto obtenido. | https://kaikki.org/dictionary.html |

## Decisión de adquisición

El Dataset se construirá por etapas reproducibles y no dependerá de API de filas durante la corrida GPU. Cada fuente materializada incluirá URL, fecha/revisión, licencia declarada, SHA-256, conteo, idioma y rol de currículo. El entrenamiento no se inicia hasta que el manifiesto de datos certifique los mínimos por idioma y dominio.

## Hallazgos de índices oficiales (2026-08-19)

Los índices `latest` de Wikimedia exponen artefactos `pages-articles-multistream` y sus índices para `eswiki` y `enwiki`. Los dumps completos superan el presupuesto operativo de un piloto Kaggle, por lo que la adquisición inicial utilizará fragmentos multistream oficiales y documentará exactamente los rangos, fechas y hashes descargados. Esta decisión conserva procedencia primaria sin depender de la API de filas de Hugging Face durante el entrenamiento.

Las rutas verificadas el 2026-08-19 fueron `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2` y `https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2`. El primer fragmento inglés cubre páginas 1–41.242 y el español 1–159.400. Los SHA-256 de los artefactos realmente descargados se incluirán en el manifiesto materializado, no se presumirán desde el índice.

Las cabeceras oficiales consultadas el 2026-08-19 indican soporte `Accept-Ranges: bytes` para ambos fragmentos. El artefacto inglés declara 299.138.062 bytes y `Last-Modified: Wed, 05 Aug 2026 18:41:29 GMT`; el español declara 319.479.582 bytes y `Last-Modified: Tue, 04 Aug 2026 10:34:45 GMT`. Por ello la descarga reanudable del materializador es aplicable sin añadir una API externa.
