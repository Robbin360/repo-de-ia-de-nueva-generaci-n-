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
};

export type AethelRuntimeSnapshot = {
  mode: string;
  status: string;
  config?: Record<string, unknown>;
};

const PRESETS = {
  "pilot-100m": { vocabSize: 32_000, dim: 512, layers: 4, heads: 8, kvHeads: 2, experts: 8, activeExperts: 2, maxSeqLen: 1024 },
  "research-300m": { vocabSize: 32_000, dim: 768, layers: 8, heads: 12, kvHeads: 3, experts: 8, activeExperts: 2, maxSeqLen: 2048 },
  "scale-1b": { vocabSize: 32_000, dim: 1024, layers: 16, heads: 16, kvHeads: 4, experts: 8, activeExperts: 2, maxSeqLen: 2048 },
} as const;

function calculatePreset(name: string, values: (typeof PRESETS)[keyof typeof PRESETS]): AethelPreset {
  const moeHiddenDim = 256 * Math.ceil((8 * values.dim / 3) / 256);
  const tiedEmbedding = values.vocabSize * values.dim;
  const attention = (2 * values.dim * values.dim) + (2 * values.dim * values.dim * values.kvHeads / values.heads);
  const moe = (values.dim * values.experts) + (3 * values.experts * values.dim * moeHiddenDim);
  const core = tiedEmbedding + values.layers * (attention + moe + 2 * values.dim) + values.dim;
  const nextgen = (values.dim * values.dim + values.dim) + (values.dim * values.dim) + (values.dim + 1) + (9 * values.dim + 3) + (values.dim * values.dim) + (6 * values.dim * values.dim + 6 * values.dim) + (2 * values.dim) + (values.dim * values.dim);
  const parametersTotal = core + nextgen;
  return { name, parametersTotal, parametersMillions: Number((parametersTotal / 1_000_000).toFixed(2)), activeExpertFraction: `${values.activeExperts}/${values.experts}`, estimatedOptimizerGib: Number((parametersTotal * 12 / 1024 ** 3).toFixed(2)), context: values.maxSeqLen, experts: values.experts, activeExperts: values.activeExperts, moeHiddenDim };
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
      protocol: ["recuperación", "integración", "predicción"],
      observable: "Se registran las fuentes de memoria y los pesos del Espacio de Trabajo Global.",
      privacy: "No se expone cadena de pensamiento interna; se presentan resultados, evidencia y límites verificables.",
    },
    technology: [
      "**Python + PyTorch:** investigación, entrenamiento, memoria y evaluación; implementado.",
      "**Triton + CUDA:** kernels fusionados de GPU, incluido SwiGLU; implementado en el repositorio original y pendiente de integración GPU en este runner.",
      "**Rust + Candle:** runtime de inferencia de producción documentado en `rust_engine/`; previsto para exportación del modelo.",
      "**TypeScript / Node.js:** ecosistema, evaluaciones, gateway y dashboard; implementado.",
      "**C++:** conexión nativa futura mencionada por la interfaz original; no hay fuentes C++ en la clonación auditada.",
      "**C#:** no se encontraron archivos ni referencias verificables en la clonación auditada; requiere confirmar otra rama o documento.",
    ],
    limitations: ["Las capacidades requieren entrenamiento con corpus y evaluación real.", "No hay puntuaciones MMLU, GSM8K o HumanEval hasta ejecutar sus harnesses con predicciones reales.", "FSDP y la familia scale-1b requieren al menos dos GPU CUDA reales y validación distribuida."],
  };
}

export function formatAethelSpecificationForChat(runtime: AethelRuntimeSnapshot) {
  const specification = getAethelSpecification();
  const rows = specification.presets.map(item => `| ${item.name} | ${item.parametersMillions.toFixed(2)} M | ${item.activeExpertFraction} | ${item.context.toLocaleString("es-ES")} | ${item.estimatedOptimizerGib.toFixed(2)} GiB |`).join("\n");
  const activeConfig = runtime.config && Object.keys(runtime.config).length ? Object.entries(runtime.config).map(([key, value]) => `${key}=${String(value)}`).join(", ") : "sin proceso activo";
  const technology = specification.technology.map(item => `- ${item}`).join("\n");
  return `## Ficha técnica verificable de Aethel NextGen\n\n### Configuración activa\n- **Modo cognitivo:** \`${runtime.mode}\`\n- **Motor:** \`${runtime.status}\`\n- **Configuración de proceso:** ${activeConfig}\n\n| Familia | Parámetros totales | Expertos activos | Contexto | AdamW estimado |\n|---|---:|---:|---:|---:|\n${rows}\n\n### Memoria implementada\n- **Trabajo:** ${specification.memory.working}\n- **Episódica:** ${specification.memory.episodic}\n- **Semántica:** ${specification.memory.semantic}\n- **Consolidación:** ${specification.memory.consolidation}\n\n### Razonamiento observable\nEl protocolo operativo es **recuperación → integración → predicción**. ${specification.reasoning.observable} ${specification.reasoning.privacy}\n\n### Stack tecnológico del repositorio original\n${technology}\n\n### Límites actuales\n${specification.limitations.map(item => `- ${item}`).join("\n")}`;
}
