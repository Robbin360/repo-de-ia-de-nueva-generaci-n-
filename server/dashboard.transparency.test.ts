import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const homeSource = readFileSync(resolve(process.cwd(), "client/src/pages/Home.tsx"), "utf8");

describe("dashboard transparency", () => {
  it("labels calculated designs as theoretical and denies an Aethel checkpoint", () => {
    expect(homeSource).toContain("configuración piloto calculada");
    expect(homeSource).toContain("no existe aún un checkpoint Aethel entrenado");
    expect(homeSource).toContain("Configuración de diseño");
    expect(homeSource).toContain("No hay pesos entrenados, benchmarks ni telemetría Aethel disponibles aún");
  });

  it("does not expose a dashboard control that claims to start real Aethel training", () => {
    expect(homeSource).toContain("El dashboard no inicia procesos de entrenamiento");
    expect(homeSource).toContain("Sin telemetría de modelo propio");
    expect(homeSource).not.toContain("Iniciar Aethel NextGen");
    expect(homeSource).not.toContain("TRAINING LOSS / REAL PROCESS");
  });

  it("does not present the platform LLM or a standby service as a trained Aethel runtime", () => {
    expect(homeSource).toContain("LLM de plataforma conectado · Aethel Seed sin entrenar");
    expect(homeSource).toContain("modelo propio no iniciado");
    expect(homeSource).toContain("No se muestran valores cero como métricas");
    expect(homeSource).not.toContain("LLM real conectado · runtime bajo demanda");
    expect(homeSource).not.toContain("runtime standby");
  });
});
