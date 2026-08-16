import { describe, expect, it } from "vitest";
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
});
