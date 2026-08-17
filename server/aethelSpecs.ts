export type AethelPreset = {
  name: string;
  parametersTotal: number;
  parametersMillions: number;
  activeExpertFraction: string;
  estimatedOptimizerGib: number;
  context: number;
  experts: number;
  activeExperts: number;
  moeHiddenDim: number;
  adaptiveRefinementSteps: number;
  adaptiveRefinementParameters: number;
};

type PresetDefinition = {
  vocabSize: number;
  dim: number;
  layers: number;
  heads: number;
  kvHeads: number;
  experts: number;
  activeExperts: number;
  maxSeqLen: number;
  adaptiveRefinementSteps?: number;
};

export type AethelRuntimeSnapshot = {
  mode: string;
  status: string;
  config?: Record<string, unknown>;
};

const PRESETS: Record<string, PresetDefinition> = {
  "pilot-100m": { vocabSize: 32_000, dim: 512, layers: 4, heads: 8, kvHeads: 2, experts: 8, activeExperts: 2, maxSeqLen: 1024 },
  "research-300m": { vocabSize: 32_000, dim: 768, layers: 8, heads: 12, kvHeads: 3, experts: 8, activeExperts: 2, maxSeqLen: 2048 },
  "scale-1b": { vocabSize: 32_000, dim: 1024, layers: 16, heads: 16, kvHeads: 4, experts: 8, activeExperts: 2, maxSeqLen: 2048 },
  "adaptive-research-300m": { vocabSize: 32_000, dim: 768, layers: 8, heads: 12, kvHeads: 3, experts: 8, activeExperts: 2, maxSeqLen: 2048, adaptiveRefinementSteps: 2 },
};

function calculatePreset(name: string, values: PresetDefinition): AethelPreset {
  const moeHiddenDim = 256 * Math.ceil((8 * values.dim / 3) / 256);
  const tiedEmbedding = values.vocabSize * values.dim;
  const attention = (2 * values.dim * values.dim) + (2 * values.dim * values.dim * values.kvHeads / values.heads);
  const moe = (values.dim * values.experts) + (3 * values.experts * values.dim * moeHiddenDim);
  const core = tiedEmbedding + values.layers * (attention + moe + 2 * values.dim) + values.dim;
  const nextgen = (values.dim * values.dim + values.dim) + (values.dim * values.dim) + (values.dim + 1) + (9 * values.dim + 3) + (values.dim * values.dim) + (6 * values.dim * values.dim + 6 * values.dim) + (2 * values.dim) + (values.dim * values.dim);
  const adaptiveRefinementSteps = values.adaptiveRefinementSteps ?? 0;
  const adaptiveRefinementParameters = adaptiveRefinementSteps ? (6 * values.dim * values.dim + 9 * values.dim + 1) : 0;
  const parametersTotal = core + nextgen + adaptiveRefinementParameters;
  return {
    name,
    parametersTotal,
    parametersMillions: Number((parametersTotal / 1_000_000).toFixed(2)),
    activeExpertFraction: `${values.activeExperts}/${values.experts}`,
    estimatedOptimizerGib: Number((parametersTotal * 12 / 1024 ** 3).toFixed(2)),
    context: values.maxSeqLen,
    experts: values.experts,
    activeExperts: values.activeExperts,
    moeHiddenDim,
    adaptiveRefinementSteps,
    adaptiveRefinementParameters,
  };
}

export function getAethelSpecification() {
  const presets = Object.entries(PRESETS).map(([name, values]) => calculatePreset(name, values));
  return {
    presets,
    memory: {
      working: "Estado recurrente GRU por sesión; se reinicia de forma explícita.",
      episodic: "Estados y tokens persistidos en JSONL, recuperados por similitud coseno ponderada.",
      semantic: "Prototipos vectoriales persistidos y fusionados por similitud; no se les asignan etiquetas inventadas.",
      consolidation: "Replay estratificado, limitado y diverso en el Ciclo de Sueño.",
    },
    reasoning: {
      protocol: ["recuperación", "integración", "refinamiento presupuestado", "predicción"],
      observable: "Se registran las fuentes de memoria, los pesos del Espacio de Trabajo Global y la telemetría del refinamiento cuando ARC está habilitado.",
      privacy: "No se expone cadena de pensamiento interna; se presentan resultados, evidencia y límites verificables.",
    },
    adaptiveCompute: "ARC es una variante experimental apagada por defecto. Registra estados seleccionados, pasos efectivos y dificultad; sólo se adopta tras comparar pérdida, router, latencia y memoria contra el baseline en GPU.",
    technology: [
      "**Python + PyTorch:** laboratorio temporal para topología, entrenamiento, evaluación y exportación de pesos; implementado.",
      "**Triton + CUDA:** puente SwiGLU fusionado integrado; el runner GPU exige Triton por defecto y sólo activa esta ruta con CUDA real.",
      "**Rust + Candle:** núcleo Rust de memoria trazable compilado y probado; el servicio 24/7 y la exportación Candle todavía no están desplegados.",
      "**Mojo:** contrato de inferencia token a token definido; el runtime Mojo aún no está instalado ni validado con pesos Aethel.",
      "**TypeScript / Node.js:** ecosistema, evaluaciones, gateway y dashboard; implementado.",
      "**C++:** conexión nativa futura mencionada por la interfaz original; no hay fuentes C++ en la clonación auditada.",
      "**C#:** no se encontraron archivos ni referencias verificables en la clonación auditada; requiere confirmar otra rama o documento.",
    ],
    limitations: [
      "Las capacidades requieren entrenamiento con corpus y evaluación real.",
      "No hay puntuaciones MMLU, GSM8K o HumanEval hasta ejecutar sus harnesses con predicciones reales.",
      "FSDP y la familia scale-1b requieren al menos dos GPU CUDA reales y validación distribuida.",
      "Triton no se ha benchmarkeado aún en una GPU objetivo; Mojo y el servicio Rust 24/7 no se declararán disponibles hasta ejecutar pruebas de interoperabilidad y operación persistente.",
      "ARC no es una ganancia de rendimiento demostrada: sus conteos no sustituyen mediciones de FLOPs, latencia o VRAM.",
    ],
  };
}

export function formatAethelSpecificationForChat(runtime: AethelRuntimeSnapshot) {
  const specification = getAethelSpecification();
  const rows = specification.presets.map(item => `| ${item.name} | ${item.parametersMillions.toFixed(2)} M | ${item.activeExpertFraction} | ${item.context.toLocaleString("es-ES")} | ${item.estimatedOptimizerGib.toFixed(2)} GiB |`).join("\n");
  const activeConfig = runtime.config && Object.keys(runtime.config).length ? Object.entries(runtime.config).map(([key, value]) => `${key}=${String(value)}`).join(", ") : "sin proceso activo";
  const technology = specification.technology.map(item => `- ${item}`).join("\n");
  return `## Ficha técnica verificable de Aethel NextGen

### Configuración activa
- **Modo cognitivo:** \`${runtime.mode}\`
- **Motor:** \`${runtime.status}\`
- **Configuración de proceso:** ${activeConfig}

| Familia | Parámetros totales | Expertos activos | Contexto | AdamW estimado |
|---|---:|---:|---:|---:|
${rows}

### Cómputo adaptativo experimental
${specification.adaptiveCompute}

### Memoria implementada
- **Trabajo:** ${specification.memory.working}
- **Episódica:** ${specification.memory.episodic}
- **Semántica:** ${specification.memory.semantic}
- **Consolidación:** ${specification.memory.consolidation}

### Razonamiento observable
El protocolo operativo admite **recuperación → integración → refinamiento presupuestado → predicción** cuando ARC está habilitado. ${specification.reasoning.observable} ${specification.reasoning.privacy}

### Stack tecnológico del repositorio original
${technology}

### Límites actuales
${specification.limitations.map(item => `- ${item}`).join("\n")}`;
}
