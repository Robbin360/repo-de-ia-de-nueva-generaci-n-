import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { invokeLLM } from "./_core/llm";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { getChatHistory, saveChatMessage } from "./db";
import { formatAethelSpecificationForChat, getAethelSpecification } from "./aethelSpecs";

const architectureModes = ["hybrid_aethel", "sparse_moe", "mamba_ssm", "test_time_compute"] as const;
export const DASHBOARD_TRAINING_BLOCK_MESSAGE = "El entrenamiento desde el dashboard está bloqueado. Aethel Seed sólo puede iniciarse mediante el runbook offline, Dataset congelado, GPU autorizada y gates Triton explícitos.";
const aethelSystemPrompt = `Eres Aethel V3, un sistema bio-mimético de inteligencia artificial construido como laboratorio experimental. Responde en español salvo que el usuario pida otro idioma. Explica tus respuestas con precisión y transparencia: tu arquitectura combina atención RoPE para posición contextual, GQA para eficiencia de memoria y Sparse MoE con expertos especializados. Tu identidad cognitiva se organiza en cinco pilares exactos: La Roca (memoria estable), El Líquido (plasticidad adaptativa), Ciclo de Sueño (consolidación), Neuromodulación (curiosidad y sorpresa) y Espacio de Trabajo Global (síntesis de hipótesis). El razonamiento que puedes describir es un protocolo observable de recuperación, integración y predicción; no muestres ni afirmes una cadena de pensamiento interna. No afirmes que tienes conciencia ni inventes resultados de benchmarks. Si una métrica no proviene de un proceso activo, di que no está disponible.`;

function requestsAethelSpecification(message: string) {
  const normalized = message.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  return ["parametro", "ficha tecnica", "especificacion", "capacidad", "memoria", "razonamiento", "cuanto mide", "cuantos parametros"].some(term => normalized.includes(term));
}

function activeRuntimeSnapshot(mode: string) {
  return { mode, status: "NOT_CONNECTED", config: undefined };
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part: any) => part?.text ?? "").join("");
  return "";
}

function rejectDashboardTraining(): never {
  throw new TRPCError({ code: "PRECONDITION_FAILED", message: DASHBOARD_TRAINING_BLOCK_MESSAGE });
}

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => { const cookieOptions = getSessionCookieOptions(ctx.req); ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 }); return { success: true } as const; }),
  }),
  chat: router({
    send: publicProcedure.input(z.object({ sessionId: z.string().min(1).max(96), message: z.string().min(1).max(12000), architectureMode: z.enum(architectureModes).default("hybrid_aethel") })).mutation(async ({ input, ctx }) => {
      const history = await getChatHistory(input.sessionId, 18);
      await saveChatMessage({ sessionId: input.sessionId, userId: ctx.user?.id, role: "user", content: input.message, architectureMode: input.architectureMode, tokensProcessed: Math.ceil(input.message.length / 4) });
      if (requestsAethelSpecification(input.message)) {
        const reply = formatAethelSpecificationForChat(activeRuntimeSnapshot(input.architectureMode));
        const tokens = Math.ceil((input.message.length + reply.length) / 4);
        await saveChatMessage({ sessionId: input.sessionId, userId: ctx.user?.id, role: "assistant", content: reply, architectureMode: input.architectureMode, tokensProcessed: tokens, metadata: JSON.stringify({ model: "Aethel specification engine", pillars: 5, source: "calculated" }) });
        return { reply, tokensProcessed: tokens, model: "Aethel specification engine" };
      }
      const response = await invokeLLM({ messages: [{ role: "system", content: `${aethelSystemPrompt}\nModo activo: ${input.architectureMode}.` }, ...history.reverse().map(item => ({ role: item.role as "user" | "assistant", content: item.content })), { role: "user", content: input.message }] });
      const reply = contentToText(response.choices?.[0]?.message?.content) || "El LLM real no devolvió contenido en esta iteración.";
      const tokens = Math.ceil((input.message.length + reply.length) / 4);
      await saveChatMessage({ sessionId: input.sessionId, userId: ctx.user?.id, role: "assistant", content: reply, architectureMode: input.architectureMode, tokensProcessed: tokens, metadata: JSON.stringify({ model: response.model ?? "built-in-llm", pillars: 5 }) });
      return { reply, tokensProcessed: tokens, model: response.model ?? "Aethel LLM Core" };
    }),
    history: publicProcedure.input(z.object({ sessionId: z.string().min(1).max(96) })).query(({ input }) => getChatHistory(input.sessionId, 60)),
  }),
  aethel: router({
    specification: publicProcedure.query(() => getAethelSpecification()),
  }),
  engine: router({
    status: publicProcedure.query(() => ({ status: "NOT_CONNECTED" as const, kernel: "Aethel Seed runtime", telemetry: "unavailable", message: "No hay un proceso Aethel activo. El dashboard no inicia entrenamiento; usa el runbook offline con GPU autorizada.", tokensPerSecond: null, loss: null, vram: null, kvCache: null, experts: null, config: null })),
  }),
  training: router({
    start: publicProcedure.input(z.object({ dim: z.number().int().min(64).max(1024), layers: z.number().int().min(1).max(8), experts: z.number().int().min(1).max(8), learningRate: z.number().min(0.000001).max(0.01), steps: z.number().int().min(1).max(1000).default(20) })).mutation(() => rejectDashboardTraining()),
    nextgenStart: publicProcedure.input(z.object({ dim: z.number().int().min(64).max(1024), layers: z.number().int().min(1).max(8), experts: z.number().int().min(1).max(8), learningRate: z.number().min(0.000001).max(0.01), steps: z.number().int().min(1).max(100000).default(1000), seqLen: z.number().int().min(16).max(512).default(128), batchSize: z.number().int().min(1).max(16).default(2) })).mutation(() => rejectDashboardTraining()),
    status: publicProcedure.input(z.object({ jobId: z.string().uuid() })).query(() => ({ status: "NOT_CONNECTED" as const, metrics: [], error: DASHBOARD_TRAINING_BLOCK_MESSAGE })),
  }),
  benchmarks: router({
    summary: publicProcedure.query(() => ({ metrics: ["MMLU", "HumanEval", "GSM8K"], models: [{ name: "Aethel", mmlu: null, humaneval: null, gsm8k: null, accent: true }, { name: "GPT-4", mmlu: null, humaneval: null, gsm8k: null }, { name: "Llama", mmlu: null, humaneval: null, gsm8k: null }, { name: "Mixtral", mmlu: null, humaneval: null, gsm8k: null }], note: "No hay resultados verificables cargados. Ejecuta evaluaciones reales para poblar esta matriz; no se muestran valores inventados." })),
  }),
});

export type AppRouter = typeof appRouter;
