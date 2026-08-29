export const E0_V8_AUDIT = {
  experiment: "Seed E0 V8",
  release: "e0-v8-canonical-cuda-device-check",
  status: "COMPLETADO / EXPERIMENTAL",
  step: 4992,
  checkpoint: "latest.pt + 3 snapshots",
  checkpointState: "inspeccionado; reanudable bajo contrato",
  holdout: {
    en: { segments: 256, loss: 7.771877005696297, perplexity: 2372.921097307039 },
    es: { segments: 256, loss: 10.774345595389605, perplexity: 47779.19608484159 },
  },
  router: "UNHEALTHY / requiere diagnóstico posterior",
  runtime: "No desplegado en esta plataforma",
  limits: [
    "No es un benchmark comparativo.",
    "No acredita utilidad ni capacidad comercial.",
    "El chat continúa usando un LLM de plataforma.",
    "El checkpoint no se carga, mueve ni promociona desde este panel.",
  ],
} as const;
