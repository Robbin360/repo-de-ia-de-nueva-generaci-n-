import { describe, expect, it } from "vitest";
import { appRouter, DASHBOARD_TRAINING_BLOCK_MESSAGE } from "./routers";
import type { TrpcContext } from "./_core/context";

function createContext(): TrpcContext {
  return {
    user: null,
    req: { protocol: "https", headers: {} } as TrpcContext["req"],
    res: {} as TrpcContext["res"],
  };
}

describe("training dashboard guard", () => {
  it("rejects unauthorised dashboard training rather than spawning a local process", async () => {
    const caller = appRouter.createCaller(createContext());
    await expect(caller.training.nextgenStart({ dim: 256, layers: 2, experts: 4, learningRate: 0.0003, steps: 10, seqLen: 128, batchSize: 2 }))
      .rejects.toMatchObject({ code: "PRECONDITION_FAILED", message: DASHBOARD_TRAINING_BLOCK_MESSAGE });
  });
});
