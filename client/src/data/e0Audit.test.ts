import { describe, expect, it } from "vitest";
import { E0_V8_AUDIT } from "./e0Audit";

describe("E0_V8_AUDIT", () => {
  it("conserva sólo la evidencia auditada del Seed E0 V8", () => {
    expect(E0_V8_AUDIT.status).toBe("COMPLETADO / EXPERIMENTAL");
    expect(E0_V8_AUDIT.step).toBe(4992);
    expect(E0_V8_AUDIT.holdout.en.segments).toBe(256);
    expect(E0_V8_AUDIT.holdout.es.segments).toBe(256);
    expect(E0_V8_AUDIT.router).toContain("UNHEALTHY");
  });

  it("declara límites que impiden presentar E0 como producto o benchmark", () => {
    expect(E0_V8_AUDIT.limits).toContain("No es un benchmark comparativo.");
    expect(E0_V8_AUDIT.limits).toContain("No acredita utilidad ni capacidad comercial.");
    expect(E0_V8_AUDIT.runtime).toBe("No desplegado en esta plataforma");
  });
});
