import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const homeSource = readFileSync(resolve(process.cwd(), "client/src/pages/Home.tsx"), "utf8");

describe("dashboard transparency", () => {
  it("distinguishes calculated design from the audited experimental E0 checkpoint", () => {
    expect(homeSource).toContain("configuración piloto calculada");
    expect(homeSource).toContain("Seed E0 V8 produjo un checkpoint experimental auditado");
    expect(homeSource).toContain("E0 V8 auditado");
    expect(homeSource).toContain("No se cargan pesos, ni se presenta este resultado experimental como benchmark o producto");
  });

  it("does not expose a dashboard control that claims to start real Aethel training", () => {
    expect(homeSource).toContain("No inicia, reanuda, carga ni promociona checkpoints");
    expect(homeSource).toContain("no hay corrida activa");
    expect(homeSource).not.toContain("Iniciar Aethel NextGen");
    expect(homeSource).not.toContain("TRAINING LOSS / REAL PROCESS");
  });

  it("does not present the platform LLM or a standby service as a trained Aethel runtime", () => {
    expect(homeSource).toContain("LLM de plataforma · Seed E0 V8 auditado · runtime no desplegado");
    expect(homeSource).toContain("runtime no desplegado");
    expect(homeSource).toContain("sin serving propio");
    expect(homeSource).not.toContain("LLM real conectado · runtime bajo demanda");
    expect(homeSource).not.toContain("runtime standby");
  });
});
