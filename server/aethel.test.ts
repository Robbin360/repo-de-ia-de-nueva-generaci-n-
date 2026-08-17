import { describe, expect, it, vi } from "vitest";

const chatStore = vi.hoisted(() => ({ saved: [] as Array<Record<string, unknown>> }));

vi.mock("./db", () => ({
  getChatHistory: vi.fn(async () => []),
  saveChatMessage: vi.fn(async (message: Record<string, unknown>) => { chatStore.saved.push(message); }),
}));

vi.mock("./_core/llm", () => ({ invokeLLM: vi.fn() }));

import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const ctx = { user: null, req: {} as TrpcContext["req"], res: {} as TrpcContext["res"] } as TrpcContext;

describe("Aethel real runtime", () => {
  it("does not report an online engine without an active process", async () => {
    const result = await appRouter.createCaller(ctx).engine.status();
    expect(["NOT_CONNECTED", "STARTING", "RUNNING"]).toContain(result.status);
    if (result.status === "NOT_CONNECTED") {
      expect(result.tokensPerSecond).toBeNull();
      expect(result.loss).toBeNull();
      expect(result.experts).toBeNull();
    }
  });

  it("does not fabricate benchmark scores", async () => {
    const result = await appRouter.createCaller(ctx).benchmarks.summary();
    expect(result.metrics).toEqual(["MMLU", "HumanEval", "GSM8K"]);
    expect(result.models.map(model => model.name)).toEqual(["Aethel", "GPT-4", "Llama", "Mixtral"]);
    expect(result.models.every(model => model.mmlu === null && model.humaneval === null && model.gsm8k === null)).toBe(true);
    expect(result.note).toContain("No hay resultados verificables");
  });

  it("expone presupuestos calculados y límites cognitivos verificables", async () => {
    const result = await appRouter.createCaller(ctx).aethel.specification();
    expect(result.presets.map(item => item.name)).toEqual(["pilot-100m", "research-300m", "scale-1b", "adaptive-research-300m"]);
    expect(result.presets.map(item => item.parametersMillions)).toEqual([97.16, 344.34, 1192.68, 347.88]);
    expect(result.presets.at(-1)?.adaptiveRefinementParameters).toBe(3545857);
    expect(result.memory.semantic).toContain("Prototipos vectoriales");
    expect(result.reasoning.protocol).toEqual(["recuperación", "integración", "refinamiento presupuestado", "predicción"]);
    expect(result.adaptiveCompute).toContain("apagada por defecto");
    expect(result.technology.join(" ")).toContain("Triton + CUDA");
    expect(result.technology.join(" ")).toContain("Rust + Candle");
    expect(result.technology.join(" ")).toContain("Mojo");
    expect(result.technology.join(" ")).toContain("contrato de inferencia token a token definido");
    expect(result.technology.join(" ")).toContain("C#");
    expect(result.limitations.some(item => item.includes("GPU CUDA"))).toBe(true);
  });

  it("devuelve la ficha técnica desde el flujo de chat sin invocar un LLM", async () => {
    chatStore.saved.length = 0;
    const result = await appRouter.createCaller(ctx).chat.send({ sessionId: "aethel-spec-test", message: "¿Cuántos parámetros y qué memoria tiene?", architectureMode: "hybrid_aethel" });
    expect(result.model).toBe("Aethel specification engine");
    expect(result.reply).toContain("Configuración activa");
    expect(result.reply).toContain("hybrid_aethel");
    expect(result.reply).toContain("97.16 M");
    expect(result.reply).toContain("347.88 M");
    expect(result.reply).toContain("Cómputo adaptativo experimental");
    expect(result.reply).toContain("NOT_CONNECTED");
    expect(result.reply).toContain("Stack tecnológico del repositorio original");
    expect(result.reply).toContain("Triton + CUDA");
    expect(result.reply).toContain("Rust + Candle");
    expect(result.reply).toContain("Mojo");
    expect(chatStore.saved).toHaveLength(2);
  });
});
