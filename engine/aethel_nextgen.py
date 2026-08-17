"""Aethel NextGen alineado con la arquitectura documentada del repositorio.

El núcleo conserva los mecanismos verificables del proyecto: La Roca, El Líquido,
MoE disperso, RoPE, GQA, ultra-eficiencia, Ciclo de Sueño, Neuromodulación y
Espacio de Trabajo Global. No afirma consciencia; expone estados medibles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aethel_model import AethelConfig, AethelModel


@dataclass
class NextGenConfig:
    vocab_size: int = 256
    dim: int = 384
    layers: int = 4
    heads: int = 8
    kv_heads: int = 2
    experts: int = 4
    active_experts: int = 2
    max_seq_len: int = 2048
    memory_slots: int = 128
    replay_capacity: int = 2048
    memory_decay: float = 0.995
    router_bias_step: float = 0.05
    router_bias_limit: float = 0.5
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_freeze_base: bool = True

    def model_config(self) -> AethelConfig:
        return AethelConfig(vocab_size=self.vocab_size, dim=self.dim, n_layers=self.layers, n_heads=self.heads, n_kv_heads=self.kv_heads, n_experts=self.experts, active_experts=self.active_experts, max_seq_len=self.max_seq_len, router_bias_step=self.router_bias_step, router_bias_limit=self.router_bias_limit)


class LaRoca(nn.Module):
    """Ruta sólida: representa el conocimiento estable y la identidad base."""

    def __init__(self, dim: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.stable_projection = nn.Linear(dim, dim, bias=False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.stable_projection(observation) + self.anchor.to(observation.device, observation.dtype)


class ElLiquido(nn.Module):
    """Ruta líquida: plasticidad rápida mediante traza Hebbiana versionada."""

    def __init__(self, dim: int, decay: float, snapshot_path: str | Path | None = None):
        super().__init__()
        self.decay = decay
        self.plastic_projection = nn.Linear(dim, dim, bias=False)
        self.register_buffer("hebbian_trace", torch.zeros(1, dim), persistent=False)
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.version = 0

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.plastic_projection(observation) + self.hebbian_trace.to(observation.device, observation.dtype)

    @torch.no_grad()
    def observe(self, state: torch.Tensor, salience: float) -> None:
        normalized = F.normalize(state.detach().float(), dim=-1).mean(dim=0, keepdim=True)
        self.hebbian_trace.mul_(self.decay).add_(normalized * float(max(0.0, min(1.0, salience))) * (1.0 - self.decay))
        self.version += 1
        if self.snapshot_path:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"version": self.version, "salience": float(salience), "trace_norm": float(self.hebbian_trace.norm().cpu()), "decay": self.decay}
            with self.snapshot_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\\n")

    def manifest(self) -> dict:
        return {"version": self.version, "decay": self.decay, "trace_norm": float(self.hebbian_trace.norm().detach().cpu()), "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None}


class Neuromodulacion(nn.Module):
    """Calcula sorpresa y prioridad de aprendizaje sin mutar pesos por accidente."""

    def __init__(self, dim: int):
        super().__init__()
        self.signal = nn.Linear(dim, 1)

    def forward(self, observation: torch.Tensor, loss: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        intrinsic = torch.sigmoid(self.signal(observation)).mean()
        surprise = loss.detach().float() if loss is not None else intrinsic.detach()
        priority = torch.clamp(0.5 * intrinsic.detach() + 0.5 * torch.sigmoid(surprise), 0.0, 1.0)
        return priority, surprise


class EspacioTrabajoGlobal(nn.Module):
    """Integra hipótesis sólida, líquida y recuperada en un contexto top-down."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 3, 3)
        self.output = nn.Linear(dim, dim, bias=False)
        self.last_weights = {"La Roca": 0.0, "El Líquido": 0.0, "Memoria recuperada": 0.0}

    def forward(self, rock: torch.Tensor, liquid: torch.Tensor, recalled: torch.Tensor) -> torch.Tensor:
        candidates = torch.stack([rock, liquid, recalled], dim=1)
        weights = F.softmax(self.gate(torch.cat([rock, liquid, recalled], dim=-1)), dim=-1).unsqueeze(-1)
        mean_weights = weights.detach().float().mean(dim=0).squeeze(-1).cpu().tolist()
        self.last_weights = {"La Roca": float(mean_weights[0]), "El Líquido": float(mean_weights[1]), "Memoria recuperada": float(mean_weights[2])}
        return self.output((candidates * weights).sum(dim=1))


class CicloDeSueno:
    """Consolidación explícita: mantiene replay y manifiestos sin actualizar pesos solo."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.replay: list[dict] = []
        self.consolidation_step = 0

    def consolidate(self, state: torch.Tensor, tokens: list[int], priority: float) -> None:
        self.consolidation_step += 1
        signature = sum((index + 1) * token for index, token in enumerate(tokens[:32])) % 1_000_003 if tokens else 0
        self.replay.append({"state": state.detach().float().flatten().tolist(), "tokens": tokens, "priority": float(priority), "age": self.consolidation_step, "signature": signature})
        # Conserva recuerdos salientes y diversidad aproximada de secuencias, no solo los más recientes.
        buckets: dict[int, dict] = {}
        for item in sorted(self.replay, key=lambda candidate: candidate["priority"] * 0.7 + min(candidate["age"], 10_000) * 1e-7, reverse=True):
            buckets.setdefault(item["signature"], item)
        self.replay = sorted(buckets.values(), key=lambda item: item["priority"], reverse=True)[: self.capacity]

    def manifest(self) -> dict:
        unique_signatures = len({item["signature"] for item in self.replay})
        return {"replay_records": len(self.replay), "capacity": self.capacity, "unique_signatures": unique_signatures, "consolidation_step": self.consolidation_step}

    def sample_pairs(self, seq_len: int, batch_size: int, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Devuelve pares autoregresivos reales del buffer sin inventar ejemplos."""
        candidates = [item["tokens"] for item in self.replay if len(item["tokens"]) >= seq_len + 1]
        if not candidates:
            return None
        selected = [candidates[index % len(candidates)] for index in range(batch_size)]
        stacked = torch.tensor([tokens[: seq_len + 1] for tokens in selected], dtype=torch.long, device=device)
        return stacked[:, :-1], stacked[:, 1:]


class MemoriaEpisodica:
    """Memoria persistente recuperable y auditable en JSONL."""

    def __init__(self, path: str | Path, dim: int, capacity: int = 2048):
        self.path = Path(path)
        self.dim = dim
        self.capacity = capacity
        self.records: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(json.loads(line))
            self.records = self.records[-capacity:]

    def add(self, state: torch.Tensor, tokens: list[int], salience: float) -> None:
        self.records.append({"state": state.detach().float().flatten().tolist(), "tokens": tokens, "salience": float(salience)})
        self.records = self.records[-self.capacity:]

    def retrieve(self, query: torch.Tensor, k: int = 4) -> Optional[torch.Tensor]:
        recalled, _ = self.retrieve_with_trace(query, k)
        return recalled

    def retrieve_with_trace(self, query: torch.Tensor, k: int = 4) -> tuple[Optional[torch.Tensor], dict]:
        if not self.records:
            return None, {"source": "episodic", "available": 0, "selected": 0, "top_similarity": None}
        vectors = torch.tensor([r["state"] for r in self.records], dtype=torch.float32, device=query.device)
        scores = F.normalize(vectors, dim=-1) @ F.normalize(query.detach().float(), dim=-1).squeeze(0)
        values, indices = torch.topk(scores, min(k, len(self.records)))
        weights = F.softmax(values, dim=0).unsqueeze(-1)
        recalled = (vectors[indices] * weights).sum(dim=0, keepdim=True).to(query.dtype)
        return recalled, {"source": "episodic", "available": len(self.records), "selected": int(indices.numel()), "top_similarity": float(values[0].detach().cpu())}

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MemoriaSemantica:
    """Prototipos persistentes de estados recurrentes; no inventa etiquetas ni texto."""

    def __init__(self, path: str | Path, dim: int, capacity: int = 128, merge_threshold: float = 0.92):
        self.path = Path(path)
        self.dim = dim
        self.capacity = capacity
        self.merge_threshold = merge_threshold
        self.records: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(json.loads(line))
            self.records = self.records[-capacity:]

    def add(self, state: torch.Tensor, salience: float) -> None:
        vector = F.normalize(state.detach().float().flatten(), dim=0).cpu()
        if self.records:
            vectors = torch.tensor([record["state"] for record in self.records], dtype=torch.float32)
            scores = F.normalize(vectors, dim=-1) @ vector
            value, index = torch.max(scores, dim=0)
            if float(value) >= self.merge_threshold:
                record = self.records[int(index)]
                count = int(record.get("observations", 1))
                merged = F.normalize((torch.tensor(record["state"], dtype=torch.float32) * count + vector * max(float(salience), 0.01)) / (count + max(float(salience), 0.01)), dim=0)
                record.update({"state": merged.tolist(), "observations": count + 1, "salience": max(float(record.get("salience", 0.0)), float(salience))})
                return
        self.records.append({"state": vector.tolist(), "observations": 1, "salience": float(salience)})
        self.records = self.records[-self.capacity:]

    def retrieve_with_trace(self, query: torch.Tensor, k: int = 2) -> tuple[Optional[torch.Tensor], dict]:
        if not self.records:
            return None, {"source": "semantic", "available": 0, "selected": 0, "top_similarity": None}
        vectors = torch.tensor([record["state"] for record in self.records], dtype=torch.float32, device=query.device)
        scores = F.normalize(vectors, dim=-1) @ F.normalize(query.detach().float(), dim=-1).squeeze(0)
        values, indices = torch.topk(scores, min(k, len(self.records)))
        weights = F.softmax(values, dim=0).unsqueeze(-1)
        recalled = (vectors[indices] * weights).sum(dim=0, keepdim=True).to(query.dtype)
        observations = sum(int(self.records[int(index)].get("observations", 1)) for index in indices.detach().cpu().tolist())
        return recalled, {"source": "semantic", "available": len(self.records), "selected": int(indices.numel()), "top_similarity": float(values[0].detach().cpu()), "observations": observations}

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def manifest(self) -> dict:
        observations = sum(int(record.get("observations", 1)) for record in self.records)
        return {"semantic_records": len(self.records), "capacity": self.capacity, "observations": observations, "path": str(self.path), "merge_threshold": self.merge_threshold}


class WorkingMemory(nn.Module):
    """Estado de trabajo recurrente que se reinicia explícitamente por sesión."""

    def __init__(self, dim: int):
        super().__init__()
        self.update = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, observation: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
        if state is None:
            state = torch.zeros_like(observation)
        return self.norm(self.update(observation, state))


class LoRALinear(nn.Module):
    """Adaptador de bajo rango que conserva la proyección base sin modificarla."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA requiere rango positivo")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adaptation = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + adaptation


class AethelNextGen(nn.Module):
    """La arquitectura documentada compuesta en un núcleo entrenable."""

    def __init__(self, config: NextGenConfig, memory_path: str | Path = "engine/artifacts/nextgen/episodic_memory.jsonl"):
        super().__init__()
        self.config = config
        self.core = AethelModel(config.model_config())
        self.rock = LaRoca(config.dim)
        memory_file = Path(memory_path)
        self.liquid = ElLiquido(config.dim, config.memory_decay, memory_file.with_name("liquid_versions.jsonl"))
        self.neuromodulation = Neuromodulacion(config.dim)
        self.workspace = EspacioTrabajoGlobal(config.dim)
        self.sleep = CicloDeSueno(config.replay_capacity)
        self.memory = MemoriaEpisodica(memory_path, config.dim, config.replay_capacity)
        self.semantic_memory = MemoriaSemantica(memory_file.with_name("semantic_memory.jsonl"), config.dim, config.memory_slots)
        self.working_memory = WorkingMemory(config.dim)
        self.memory_to_core = nn.Linear(config.dim, config.dim, bias=False)
        self.register_buffer("memory_state", torch.zeros(1, config.dim), persistent=False)
        self.lora_config: dict | None = None
        self.last_metrics: dict = {"memory_hits": 0, "memory_records": len(self.memory.records), "semantic_records": len(self.semantic_memory.records), "replay_records": 0, "pillar": "Aethel NextGen"}
        if config.lora_rank:
            self.enable_lora(config.lora_rank, config.lora_alpha, config.lora_freeze_base)

    def enable_lora(self, rank: int = 8, alpha: float = 16.0, freeze_base: bool = True) -> dict:
        """Añade LoRA a Q/K/V/O y SwiGLU de cada experto; es opt-in y auditable."""
        if self.lora_config is not None:
            raise RuntimeError("Los adaptadores LoRA ya están habilitados")
        if freeze_base:
            for parameter in self.parameters():
                parameter.requires_grad = False
        replaced = 0
        for layer in self.core.layers:
            for name in ("wq", "wk", "wv", "wo"):
                setattr(layer.attention, name, LoRALinear(getattr(layer.attention, name), rank, alpha))
                replaced += 1
            for expert in layer.feed_forward.experts:
                for name in ("w1", "w2", "w3"):
                    setattr(expert, name, LoRALinear(getattr(expert, name), rank, alpha))
                    replaced += 1
        self.lora_config = {"rank": rank, "alpha": alpha, "freeze_base": freeze_base, "targets": replaced}
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {**self.lora_config, "parameters_total": total, "parameters_trainable": trainable, "trainable_fraction": trainable / total}

    def reset_session(self) -> None:
        self.memory_state = torch.zeros_like(self.memory_state)
        self.last_metrics = {"memory_hits": 0, "memory_records": len(self.memory.records), "semantic_records": len(self.semantic_memory.records), "replay_records": len(self.sleep.replay), "pillar": "Aethel NextGen"}

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        observation = self.core.tok_embeddings(tokens).mean(dim=1)
        working = self.working_memory(observation, self.memory_state.expand(tokens.shape[0], -1))
        rock = self.rock(working)
        liquid = self.liquid(working)
        episodic, episodic_trace = self.memory.retrieve_with_trace(working[:1])
        semantic, semantic_trace = self.semantic_memory.retrieve_with_trace(working[:1])
        recalled_sources = [state for state in [episodic, semantic] if state is not None]
        if not recalled_sources:
            recalled = torch.zeros_like(working)
            memory_hit = 0
        else:
            memory_hit = 1
            recalled = torch.stack(recalled_sources, dim=0).mean(dim=0).reshape(1, -1).expand(working.size(0), -1)
        global_context = self.workspace(rock, liquid, recalled)
        self.memory_state = (working + global_context).mean(dim=0, keepdim=True).detach()
        logits, aux_loss, _ = self.core(tokens, memory_state=self.memory_to_core(global_context))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) + 0.01 * aux_loss
        priority, surprise = self.neuromodulation(working, loss)
        self.last_metrics.update({"memory_hits": self.last_metrics.get("memory_hits", 0) + memory_hit, "memory_records": len(self.memory.records), "semantic_records": len(self.semantic_memory.records), "replay_records": len(self.sleep.replay), "aux_loss": float(aux_loss.detach().cpu()), "neuromodulation": float(priority.cpu()), "surprise": float(surprise.cpu()), "reasoning_trace": {"protocol": ["recuperación", "integración", "predicción"], "episodic": episodic_trace, "semantic": semantic_trace, "workspace_weights": dict(self.workspace.last_weights), "internal_chain_of_thought_exposed": False}, "pillars": {"La Roca": True, "El Líquido": True, "Ciclo de Sueño": True, "Neuromodulación": True, "Espacio de Trabajo Global": True}})
        return logits, loss, dict(self.last_metrics)

    @torch.no_grad()
    def observe(self, tokens: torch.Tensor, salience: float = 1.0) -> dict:
        state = self.memory_state.detach().cpu()
        self.liquid.observe(state, salience)
        self.memory.add(state, tokens[0].detach().cpu().tolist(), salience)
        self.semantic_memory.add(state, salience)
        self.sleep.consolidate(state, tokens[0].detach().cpu().tolist(), salience)
        self.memory.flush()
        self.semantic_memory.flush()
        self.last_metrics.update({"memory_records": len(self.memory.records), "semantic_records": len(self.semantic_memory.records), "replay_records": len(self.sleep.replay), "liquid_version": self.liquid.version})
        return dict(self.last_metrics)

    def regularization_loss(self, reference: dict[str, torch.Tensor], coefficient: float = 1e-4) -> torch.Tensor:
        penalty = torch.zeros((), device=self.memory_state.device)
        for name, parameter in self.named_parameters():
            if name in reference:
                penalty = penalty + (parameter - reference[name].to(parameter.device)).pow(2).mean()
        return coefficient * penalty

    def export_memory_manifest(self) -> dict:
        return {"episodic_records": len(self.memory.records), "episodic": {"capacity": self.memory.capacity, "path": str(self.memory.path)}, "semantic": self.semantic_memory.manifest(), "replay": self.sleep.manifest(), "liquid": self.liquid.manifest(), "reasoning_protocol": ["recuperación", "integración", "predicción"]}
