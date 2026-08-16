import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { invokeLLM } from "./_core/llm";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { getChatHistory, saveChatMessage } from "./db";

type Job = { id: string; status: "STARTING" | "RUNNING" | "COMPLETED" | "FAILED"; process: ChildProcessWithoutNullStreams; metrics: Record<string, unknown>[]; error?: string; output?: string };
const jobs = new Map<string, Job>();
const architectureModes = ["hybrid_aethel", "sparse_moe", "mamba_ssm", "test_time_compute"] as const;
const aethelSystemPrompt = `Eres Aethel V3, un sistema bio-mimético de inteligencia artificial construido como laboratorio experimental. Responde en español salvo que el usuario pida otro idioma. Explica tus respuestas con precisión y transparencia: tu arquitectura combina atención RoPE para posición contextual, GQA para eficiencia de memoria y Sparse MoE con expertos especializados. Tu identidad cognitiva se organiza en cinco pilares exactos: La Roca (memoria estable), El Líquido (plasticidad adaptativa), Ciclo de Sueño (consolidación), Neuromodulación (curiosidad y sorpresa) y Espacio de Trabajo Global (síntesis de hipótesis). No afirmes que tienes conciencia ni inventes resultados de benchmarks. Si una métrica no proviene de un proceso activo, di que no está disponible.`;

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part: any) => part?.text ?? "").join("");
  return "";
}

function startTraining(input: { dim: number; layers: number; experts: number; learningRate: number; steps: number }) {
  const script = join(process.cwd(), "engine", "train_real.py");
  if (!existsSync(script)) throw new Error("El ejecutor PyTorch real no está instalado en este entorno.");
  const id = randomUUID();
  const child = spawn("python3", [script, "--dim", String(input.dim), "--layers", String(input.layers), "--experts", String(input.experts), "--learning-rate", String(input.learningRate), "--steps", String(input.steps)], { cwd: process.cwd() });
  const job: Job = { id, status: "STARTING", process: child, metrics: [] };
  jobs.set(id, job);
  let buffer = "";
  child.stdout.on("data", chunk => {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      try {
        const event = JSON.parse(line) as Record<string, unknown>;
        if (event.type === "metric") { job.status = "RUNNING"; job.metrics.push(event); }
        if (event.type === "complete") { job.status = "COMPLETED"; job.output = String(event.output); }
        if (event.type === "error") { job.status = "FAILED"; job.error = String(event.error); }
      } catch { /* ignore non-JSON framework logs */ }
    }
  });
  child.stderr.on("data", chunk => { job.error = chunk.toString().slice(-2000); });
  child.on("error", error => { job.status = "FAILED"; job.error = error.message; });
  child.on("close", code => { if (code !== 0 && job.status !== "FAILED") { job.status = "FAILED"; job.error ??= `El proceso terminó con código ${code}`; } });
  return id;
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
      const response = await invokeLLM({ messages: [{ role: "system", content: `${aethelSystemPrompt}\nModo activo: ${input.architectureMode}.` }, ...history.reverse().map(item => ({ role: item.role as "user" | "assistant", content: item.content })), { role: "user", content: input.message }] });
      const reply = contentToText(response.choices?.[0]?.message?.content) || "El LLM real no devolvió contenido en esta iteración.";
      const tokens = Math.ceil((input.message.length + reply.length) / 4);
      await saveChatMessage({ sessionId: input.sessionId, userId: ctx.user?.id, role: "assistant", content: reply, architectureMode: input.architectureMode, tokensProcessed: tokens, metadata: JSON.stringify({ model: response.model ?? "built-in-llm", pillars: 5 }) });
      return { reply, tokensProcessed: tokens, model: response.model ?? "Aethel LLM Core" };
    }),
    history: publicProcedure.input(z.object({ sessionId: z.string().min(1).max(96) })).query(({ input }) => getChatHistory(input.sessionId, 60)),
  }),
  engine: router({
    status: publicProcedure.query(() => {
      const active = Array.from(jobs.values()).find(job => job.status === "RUNNING" || job.status === "STARTING");
      if (!active) return { status: "NOT_CONNECTED" as const, kernel: "Aethel PyTorch runtime", telemetry: "unavailable", message: "No hay un proceso Aethel activo. Inicia un entrenamiento real para recibir telemetría.", tokensPerSecond: null, loss: null, vram: null, kvCache: null, experts: null };
      const latest = active.metrics.at(-1) as { tokens?: number; elapsed?: number; loss?: number; vram?: number; experts?: number[] | null; kv_cache?: number | null } | undefined;
      return { status: active.status, jobId: active.id, kernel: "Aethel PyTorch runtime", telemetry: "process", tokensPerSecond: typeof latest?.tokens === "number" && typeof latest.elapsed === "number" && latest.elapsed > 0 ? Math.round(latest.tokens / latest.elapsed) : null, loss: latest?.loss ?? null, vram: latest?.vram ?? null, kvCache: latest?.kv_cache ?? null, experts: latest?.experts ?? null };
    }),
  }),
  training: router({
    start: publicProcedure.input(z.object({ dim: z.number().int().min(64).max(1024), layers: z.number().int().min(1).max(8), experts: z.number().int().min(1).max(8), learningRate: z.number().min(0.000001).max(0.01), steps: z.number().int().min(1).max(1000).default(20) })).mutation(({ input }) => ({ jobId: startTraining(input), status: "STARTING" as const })),
    status: publicProcedure.input(z.object({ jobId: z.string().uuid() })).query(({ input }) => { const job = jobs.get(input.jobId); if (!job) return { status: "NOT_FOUND" as const, metrics: [], error: "Proceso no encontrado en este servidor." }; return { status: job.status, metrics: job.metrics, error: job.error, output: job.output }; }),
  }),
  benchmarks: router({
    summary: publicProcedure.query(() => ({ metrics: ["MMLU", "HumanEval", "GSM8K"], models: [{ name: "Aethel", mmlu: null, humaneval: null, gsm8k: null, accent: true }, { name: "GPT-4", mmlu: null, humaneval: null, gsm8k: null }, { name: "Llama", mmlu: null, humaneval: null, gsm8k: null }, { name: "Mixtral", mmlu: null, humaneval: null, gsm8k: null }], note: "No hay resultados verificables cargados. Ejecuta evaluaciones reales para poblar esta matriz; no se muestran valores inventados." })),
  }),
});

export type AppRouter = typeof appRouter;
