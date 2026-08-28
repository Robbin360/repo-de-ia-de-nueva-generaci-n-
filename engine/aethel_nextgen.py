"""Aethel NextGen alineado con la arquitectura documentada del repositorio.

El núcleo conserva los mecanismos verificables del proyecto: La Roca, El Líquido,
MoE disperso, RoPE, GQA, ultra-eficiencia, Ciclo de Sueño, Neuromodulación y
Espacio de Trabajo Global. No afirma consciencia; expone estados medibles.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aethel_model import AethelConfig, AethelModel
from router_auxiliary import add_router_auxiliary_loss, validate_router_aux_loss_weight, validate_router_jitter_noise


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
    router_aux_loss_weight: float = 0.01
    router_entropy_loss_weight: float = 0.0
    router_jitter_noise: float = 0.0
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_freeze_base: bool = True
    require_triton: bool = False
    adaptive_refinement_steps: int = 0
    adaptive_refinement_threshold: float = 0.35
    adaptive_compute_penalty: float = 0.0
    curiosity_enabled: bool = True
    curiosity_risk_block_threshold: float = 0.70

    def model_config(self) -> AethelConfig:
        return AethelConfig(vocab_size=self.vocab_size, dim=self.dim, n_layers=self.layers, n_heads=self.heads, n_kv_heads=self.kv_heads, n_experts=self.experts, active_experts=self.active_experts, max_seq_len=self.max_seq_len, router_bias_step=self.router_bias_step, router_bias_limit=self.router_bias_limit, router_jitter_noise=self.router_jitter_noise, require_triton=self.require_triton)


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

    def __init__(self, dim: int, decay: float, snapshot_path: str | Path | None = None, curiosity_path: str | Path | None = None, curiosity_ttl: int = 64):
        super().__init__()
        self.decay = decay
        self.plastic_projection = nn.Linear(dim, dim, bias=False)
        self.register_buffer("hebbian_trace", torch.zeros(1, dim), persistent=False)
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.curiosity_path = Path(curiosity_path) if curiosity_path else None
        self.curiosity_ttl = curiosity_ttl
        self.curiosity_events = 0
        self.version = 0

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.plastic_projection(observation) + self.hebbian_trace.to(observation.device, observation.dtype)

    @torch.no_grad()
    def observe(self, state: torch.Tensor, salience: float, curiosity: Optional[dict] = None) -> None:
        normalized = F.normalize(state.detach().float(), dim=-1).mean(dim=0, keepdim=True)
        # ``hebbian_trace`` es un buffer del módulo y se mueve junto al modelo.
        # La traza debe actualizarse en su propio dispositivo/dtype; las copias CPU
        # sólo corresponden a persistencia episódica, semántica y de sueño.
        normalized = normalized.to(device=self.hebbian_trace.device, dtype=self.hebbian_trace.dtype)
        self.hebbian_trace.mul_(self.decay).add_(normalized * float(max(0.0, min(1.0, salience))) * (1.0 - self.decay))
        self.version += 1
        if self.snapshot_path:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"version": self.version, "salience": float(salience), "trace_norm": float(self.hebbian_trace.norm().cpu()), "decay": self.decay}
            with self.snapshot_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\\n")
        if curiosity is not None:
            self.record_curiosity(curiosity)

    def record_curiosity(self, curiosity: dict) -> None:
        """Guarda una propuesta líquida, no una orden ni datos de entrenamiento."""
        if self.curiosity_path is None:
            return
        self.curiosity_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "liquid_version": self.version,
            "source": "local_curiosity_telemetry",
            "ttl_observations": self.curiosity_ttl,
            "eligible_for_sleep": False,
            "action": str(curiosity.get("action", "observe_only")),
            "priority": float(curiosity.get("priority", 0.0)),
            "blocked": bool(curiosity.get("blocked", False)),
            "requires_approval": bool(curiosity.get("requires_approval", False)),
            "reasons": list(curiosity.get("reasons", [])),
            "signals": dict(curiosity.get("signals", {})),
            "external_action_enabled": False,
        }
        with self.curiosity_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.curiosity_events += 1

    def manifest(self) -> dict:
        return {"version": self.version, "decay": self.decay, "trace_norm": float(self.hebbian_trace.norm().detach().cpu()), "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None, "curiosity_events": self.curiosity_events, "curiosity_path": str(self.curiosity_path) if self.curiosity_path else None, "curiosity_ttl": self.curiosity_ttl}


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


@dataclass(frozen=True)
class CuriositySignals:
    """Señales ya medidas para priorizar aprendizaje; no representan emociones."""

    uncertainty: float
    novelty: float
    contradiction: float
    expected_progress: float
    risk: float = 0.0
    cost: float = 0.0
    permitted: bool = True


@dataclass(frozen=True)
class CuriosityDecision:
    """Resultado serializable y no ejecutable del controlador de curiosidad."""

    priority: float
    action: str
    blocked: bool
    requires_approval: bool
    reasons: tuple[str, ...]
    signals: CuriositySignals

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "action": self.action,
            "blocked": self.blocked,
            "requires_approval": self.requires_approval,
            "reasons": list(self.reasons),
            "signals": {
                "uncertainty": self.signals.uncertainty,
                "novelty": self.signals.novelty,
                "contradiction": self.signals.contradiction,
                "expected_progress": self.signals.expected_progress,
                "risk": self.signals.risk,
                "cost": self.signals.cost,
                "permitted": self.signals.permitted,
            },
            "external_action_enabled": False,
        }


class CuriosityController:
    """Convierte señales en una propuesta local, nunca en una acción externa.

    La prioridad combina incertidumbre, novedad, contradicción y progreso esperado.
    El término de progreso evita que la novedad o el error puro conviertan ruido
    impredecible en una solicitud de entrenamiento. La decisión más fuerte que este
    objeto puede emitir es ``propose_replay``; su admisión al Sueño requiere una
    política independiente y aprobación explícita.
    """

    def __init__(
        self,
        risk_block_threshold: float = 0.70,
        retrieve_threshold: float = 0.25,
        clarify_threshold: float = 0.45,
        replay_threshold: float = 0.60,
        min_progress_for_replay: float = 0.20,
        assessment_capacity: int = 2048,
    ):
        thresholds = (risk_block_threshold, retrieve_threshold, clarify_threshold, replay_threshold, min_progress_for_replay)
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("Los umbrales de curiosidad deben estar entre 0 y 1")
        if not retrieve_threshold <= clarify_threshold <= replay_threshold:
            raise ValueError("Los umbrales de curiosidad deben ser crecientes")
        if assessment_capacity < 1:
            raise ValueError("La capacidad de evaluaciones de curiosidad debe ser positiva")
        self.risk_block_threshold = risk_block_threshold
        self.retrieve_threshold = retrieve_threshold
        self.clarify_threshold = clarify_threshold
        self.replay_threshold = replay_threshold
        self.min_progress_for_replay = min_progress_for_replay
        self.assessment_capacity = assessment_capacity
        self.assessments: list[CuriosityDecision] = []
        self.uncertainty_by_context: dict[str, float] = {}

    def _append(self, decision: CuriosityDecision) -> CuriosityDecision:
        self.assessments.append(decision)
        self.assessments = self.assessments[-self.assessment_capacity:]
        return decision

    def observe_progress(self, context_key: str, uncertainty: float) -> float:
        """Mide reducción local de incertidumbre para un contexto no textual.

        El valor es una señal de telemetría en ``[0, 1]``: cero para un
        contexto nuevo o cuando la incertidumbre aumenta. No actualiza pesos,
        no persiste datos de usuario y no habilita admisión a Sueño.
        """
        key = str(context_key).strip()[:128]
        if not key:
            raise ValueError("El contexto de progreso no puede estar vacío")
        current = self._bounded(uncertainty, "uncertainty")
        previous = self.uncertainty_by_context.get(key)
        if previous is None and len(self.uncertainty_by_context) >= self.assessment_capacity:
            self.uncertainty_by_context.pop(next(iter(self.uncertainty_by_context)))
        self.uncertainty_by_context[key] = current
        return 0.0 if previous is None else max(0.0, min(1.0, previous - current))

    @staticmethod
    def _bounded(value: float, name: str) -> float:
        numeric = float(value)
        if not torch.isfinite(torch.tensor(numeric)):
            raise ValueError(f"La señal de curiosidad {name} debe ser finita")
        return max(0.0, min(1.0, numeric))

    def assess(self, signals: CuriositySignals) -> CuriosityDecision:
        normalized = CuriositySignals(
            uncertainty=self._bounded(signals.uncertainty, "uncertainty"),
            novelty=self._bounded(signals.novelty, "novelty"),
            contradiction=self._bounded(signals.contradiction, "contradiction"),
            expected_progress=self._bounded(signals.expected_progress, "expected_progress"),
            risk=self._bounded(signals.risk, "risk"),
            cost=self._bounded(signals.cost, "cost"),
            permitted=bool(signals.permitted),
        )
        reasons: list[str] = []
        blocked = not normalized.permitted or normalized.risk >= self.risk_block_threshold
        if not normalized.permitted:
            reasons.append("sin_permiso")
        if normalized.risk >= self.risk_block_threshold:
            reasons.append("riesgo_alto")
        if blocked:
            decision = CuriosityDecision(0.0, "blocked", True, True, tuple(reasons), normalized)
            return self._append(decision)

        raw_priority = (
            0.20 * normalized.uncertainty
            + 0.20 * normalized.novelty
            + 0.20 * normalized.contradiction
            + 0.40 * normalized.expected_progress
            - 0.50 * normalized.risk
            - 0.10 * normalized.cost
        )
        # Mantiene alguna capacidad de investigar, pero reduce la atracción de ruido
        # cuando no hay evidencia de que el error sea aprendible.
        learnability = 0.25 + 0.75 * normalized.expected_progress
        priority = max(0.0, min(1.0, raw_priority * learnability))
        if normalized.expected_progress < self.min_progress_for_replay and normalized.uncertainty >= 0.60:
            reasons.append("incertidumbre_sin_progreso_demostrado")
        if priority >= self.replay_threshold and normalized.expected_progress >= self.min_progress_for_replay:
            action = "propose_replay"
            reasons.append("candidato_replay_requiere_revision")
        elif priority >= self.clarify_threshold:
            action = "ask_clarification"
            reasons.append("reducir_incertidumbre_sin_entrenar")
        elif priority >= self.retrieve_threshold:
            action = "retrieve_local"
            reasons.append("contrastar_memoria_local")
        else:
            action = "observe_only"
            reasons.append("evidencia_insuficiente_para_actuar")
        decision = CuriosityDecision(priority, action, False, action == "propose_replay", tuple(reasons), normalized)
        return self._append(decision)

    def manifest(self) -> dict:
        return {
            "assessments": len(self.assessments),
            "assessment_capacity": self.assessment_capacity,
            "progress_contexts": len(self.uncertainty_by_context),
            "longitudinal_progress_enabled": True,
            "risk_block_threshold": self.risk_block_threshold,
            "actions": {action: sum(item.action == action for item in self.assessments) for action in ("blocked", "observe_only", "retrieve_local", "ask_clarification", "propose_replay")},
            "external_action_enabled": False,
        }


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


class RefinamientoAdaptativo(nn.Module):
    """Profundidad adicional con presupuesto explícito; sólo procesa estados seleccionados.

    El router es deliberadamente simple y sus conteos son telemetría, no una promesa
    de aceleración: el resultado debe compararse en la GPU objetivo antes de adoptarse.
    """

    def __init__(self, dim: int, max_steps: int, threshold: float):
        super().__init__()
        if max_steps < 1:
            raise ValueError("El refinamiento adaptativo requiere al menos un paso")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("El umbral adaptativo debe estar entre 0 y 1")
        self.max_steps = max_steps
        self.threshold = threshold
        self.difficulty = nn.Linear(dim, 1)
        nn.init.zeros_(self.difficulty.weight)
        nn.init.zeros_(self.difficulty.bias)
        self.cell = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.last_metrics = {"enabled": True, "selected": 0, "batch": 0, "effective_token_steps": 0, "fraction": 0.0, "mean_difficulty": 0.0}

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor]:
        difficulty = torch.sigmoid(self.difficulty(state)).squeeze(-1)
        selected = torch.nonzero(difficulty >= self.threshold, as_tuple=False).flatten()
        refined = state
        if selected.numel():
            for _ in range(self.max_steps):
                selected_state = refined.index_select(0, selected)
                proposal = self.norm(self.cell(selected_state, selected_state))
                # La mezcla conserva gradiente hacia el router; la selección discreta
                # sigue determinando el trabajo materializado y se registra aparte.
                blend = difficulty.index_select(0, selected).unsqueeze(-1).to(proposal.dtype)
                proposal = selected_state + blend * (proposal - selected_state)
                refined = refined.index_copy(0, selected, proposal)
        selected_count = int(selected.numel())
        effective_steps = selected_count * self.max_steps
        metrics = {
            "enabled": True,
            "max_steps": self.max_steps,
            "threshold": self.threshold,
            "selected": selected_count,
            "batch": int(state.size(0)),
            "effective_token_steps": effective_steps,
            "fraction": selected_count / max(1, int(state.size(0))),
            "mean_difficulty": float(difficulty.detach().mean().cpu()),
        }
        self.last_metrics = metrics
        # Penaliza la probabilidad del router, aunque la decisión de ejecución sea discreta.
        return refined, metrics, difficulty.mean()


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
        config.router_aux_loss_weight = validate_router_aux_loss_weight(config.router_aux_loss_weight)
        config.router_entropy_loss_weight = validate_router_aux_loss_weight(config.router_entropy_loss_weight)
        config.router_jitter_noise = validate_router_jitter_noise(config.router_jitter_noise)
        self.config = config
        self.core = AethelModel(config.model_config())
        self.rock = LaRoca(config.dim)
        memory_file = Path(memory_path)
        self.liquid = ElLiquido(config.dim, config.memory_decay, memory_file.with_name("liquid_versions.jsonl"), memory_file.with_name("curiosity_events.jsonl"))
        self.neuromodulation = Neuromodulacion(config.dim)
        self.curiosity = CuriosityController(config.curiosity_risk_block_threshold) if config.curiosity_enabled else None
        self.workspace = EspacioTrabajoGlobal(config.dim)
        self.sleep = CicloDeSueno(config.replay_capacity)
        self.memory = MemoriaEpisodica(memory_path, config.dim, config.replay_capacity)
        self.semantic_memory = MemoriaSemantica(memory_file.with_name("semantic_memory.jsonl"), config.dim, config.memory_slots)
        self.working_memory = WorkingMemory(config.dim)
        self.adaptive_refinement = RefinamientoAdaptativo(config.dim, config.adaptive_refinement_steps, config.adaptive_refinement_threshold) if config.adaptive_refinement_steps else None
        self.memory_to_core = nn.Linear(config.dim, config.dim, bias=False)
        self.register_buffer("memory_state", torch.zeros(1, config.dim), persistent=False)
        self._pending_memory_state: torch.Tensor | None = None
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
        # Mantiene la identidad del buffer registrado y, con ella, su dispositivo.
        # Una reasignación puede sustituir el tensor que `Module.to(...)` ya movió.
        self.memory_state.zero_()
        self._pending_memory_state = None
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
        if self.adaptive_refinement is not None:
            global_context, adaptive_metrics, adaptive_probability = self.adaptive_refinement(global_context)
        else:
            adaptive_metrics = {"enabled": False, "max_steps": 0, "selected": 0, "batch": int(tokens.size(0)), "effective_token_steps": 0, "fraction": 0.0, "mean_difficulty": 0.0}
            adaptive_probability = torch.zeros((), device=tokens.device)
        # `memory_state` participa como entrada de la memoria de trabajo. Mutarlo
        # dentro de este forward invalidaría el tensor guardado por autograd; se
        # deja una actualización desconectada pendiente para confirmarla después de
        # backward en el entrenador.
        next_memory_state = (working + global_context).mean(dim=0, keepdim=True)
        self._pending_memory_state = next_memory_state.detach().to(device=self.memory_state.device, dtype=self.memory_state.dtype)
        logits, aux_loss, _ = self.core(tokens, memory_state=self.memory_to_core(global_context))
        entropy_loss = self.core.last_router_entropy_loss
        loss = None
        if targets is not None:
            base_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = add_router_auxiliary_loss(base_loss, aux_loss, weight=self.config.router_aux_loss_weight)
            loss = loss + self.config.router_entropy_loss_weight * entropy_loss
            if self.config.adaptive_compute_penalty and self.adaptive_refinement is not None:
                loss = loss + self.config.adaptive_compute_penalty * adaptive_probability
        priority, surprise = self.neuromodulation(working, loss)
        semantic_similarity = semantic_trace.get("top_similarity")
        novelty = 0.50 if semantic_similarity is None else max(0.0, min(1.0, 1.0 - (float(semantic_similarity) + 1.0) / 2.0))
        curiosity = None
        if self.curiosity is not None:
            uncertainty = float(torch.sigmoid(surprise.detach()).cpu())
            token_head = tokens.detach().to(dtype=torch.int64).flatten()[:32]
            positions = torch.arange(1, token_head.numel() + 1, device=tokens.device, dtype=torch.int64)
            context_key = f"token_context:{int((token_head * positions).sum().cpu())}:{token_head.numel()}"
            progress = self.curiosity.observe_progress(context_key, uncertainty)
            # La contradicción factual sigue en cero hasta incorporar una fuente
            # verificable; el progreso sólo captura reducción de incertidumbre local.
            curiosity = self.curiosity.assess(CuriositySignals(uncertainty=uncertainty, novelty=novelty, contradiction=0.0, expected_progress=progress, risk=0.0, cost=0.0, permitted=True)).to_dict()
        self.last_metrics.update({"memory_hits": self.last_metrics.get("memory_hits", 0) + memory_hit, "memory_records": len(self.memory.records), "semantic_records": len(self.semantic_memory.records), "replay_records": len(self.sleep.replay), "aux_loss": float(aux_loss.detach().cpu()), "router_entropy_loss": float(entropy_loss.detach().cpu()), "router_entropy_loss_weight": self.config.router_entropy_loss_weight, "router_jitter_noise": self.config.router_jitter_noise, "router_selection_jitter_noise": self.core.layers[0].feed_forward.last_routing_stats.get("selection_jitter_noise", 0.0), "neuromodulation": float(priority.cpu()), "surprise": float(surprise.cpu()), "curiosity": curiosity, "adaptive_compute": adaptive_metrics, "reasoning_trace": {"protocol": ["recuperación", "integración", "refinamiento presupuestado", "predicción"], "episodic": episodic_trace, "semantic": semantic_trace, "workspace_weights": dict(self.workspace.last_weights), "internal_chain_of_thought_exposed": False}, "pillars": {"La Roca": True, "El Líquido": True, "Ciclo de Sueño": True, "Neuromodulación": True, "Curiosidad funcional": self.curiosity is not None, "Espacio de Trabajo Global": True}})
        return logits, loss, dict(self.last_metrics)

    @torch.no_grad()
    def commit_memory_state(self) -> None:
        """Aplica la transición de memoria después de backward sin romper autograd."""
        if self._pending_memory_state is not None:
            self.memory_state.copy_(self._pending_memory_state)
            self._pending_memory_state = None

    @torch.no_grad()
    def observe(self, tokens: torch.Tensor, salience: float = 1.0) -> dict:
        liquid_state = self.memory_state.detach()
        self.liquid.observe(liquid_state, salience, self.last_metrics.get("curiosity"))
        persistent_state = liquid_state.cpu()
        self.memory.add(persistent_state, tokens[0].detach().cpu().tolist(), salience)
        self.semantic_memory.add(persistent_state, salience)
        self.sleep.consolidate(persistent_state, tokens[0].detach().cpu().tolist(), salience)
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
        return {"episodic_records": len(self.memory.records), "episodic": {"capacity": self.memory.capacity, "path": str(self.memory.path)}, "semantic": self.semantic_memory.manifest(), "replay": self.sleep.manifest(), "liquid": self.liquid.manifest(), "curiosity": self.curiosity.manifest() if self.curiosity is not None else {"enabled": False, "external_action_enabled": False}, "reasoning_protocol": ["recuperación", "integración", "predicción"]}

    def export_resume_runtime_state(self) -> dict:
        """Exporta el estado mutable que ``state_dict`` no incluye por diseño.

        El payload se guarda dentro de ``latest.pt`` al finalizar una frontera segura
        de optimizador. No contiene corpus ni habilita acciones externas.
        """
        curiosity: dict | None = None
        if self.curiosity is not None:
            curiosity = {
                "assessments": [decision.to_dict() for decision in self.curiosity.assessments],
                "uncertainty_by_context": dict(self.curiosity.uncertainty_by_context),
            }
        return {
            "schema": "aethel-nextgen-runtime-state/v1",
            "memory_state": self.memory_state.detach().cpu().clone(),
            "liquid": {
                "hebbian_trace": self.liquid.hebbian_trace.detach().cpu().clone(),
                "version": int(self.liquid.version),
                "curiosity_events": int(self.liquid.curiosity_events),
            },
            "episodic_records": copy.deepcopy(self.memory.records),
            "semantic_records": copy.deepcopy(self.semantic_memory.records),
            "sleep": {
                "replay": copy.deepcopy(self.sleep.replay),
                "consolidation_step": int(self.sleep.consolidation_step),
            },
            "curiosity": curiosity,
        }

    @torch.no_grad()
    def restore_resume_runtime_state(self, state: dict) -> None:
        """Restaura únicamente un estado previamente emitido por Aethel.

        La restauración valida forma, capacidades y esquema antes de mutar buffers o
        memorias. Un checkpoint antiguo sin este bloque queda explícitamente no apto
        para reanudación fiel, aunque sus pesos aún puedan inspeccionarse.
        """
        if state.get("schema") != "aethel-nextgen-runtime-state/v1":
            raise ValueError("El checkpoint no contiene un estado runtime Aethel compatible para reanudar.")
        memory_state = state.get("memory_state")
        liquid = state.get("liquid")
        sleep = state.get("sleep")
        if not isinstance(memory_state, torch.Tensor) or tuple(memory_state.shape) != tuple(self.memory_state.shape):
            raise ValueError("La forma de memory_state del checkpoint no coincide con el modelo activo.")
        if not isinstance(liquid, dict) or not isinstance(liquid.get("hebbian_trace"), torch.Tensor):
            raise ValueError("El checkpoint no contiene la traza líquida requerida para reanudar.")
        if tuple(liquid["hebbian_trace"].shape) != tuple(self.liquid.hebbian_trace.shape):
            raise ValueError("La forma de la traza líquida no coincide con el modelo activo.")
        episodic = state.get("episodic_records")
        semantic = state.get("semantic_records")
        if not isinstance(episodic, list) or not isinstance(semantic, list) or len(episodic) > self.memory.capacity or len(semantic) > self.semantic_memory.capacity:
            raise ValueError("Las memorias del checkpoint exceden la capacidad configurada o son inválidas.")
        if not isinstance(sleep, dict) or not isinstance(sleep.get("replay"), list) or len(sleep["replay"]) > self.sleep.capacity:
            raise ValueError("El replay del checkpoint excede la capacidad configurada o es inválido.")
        self.memory_state.copy_(memory_state.to(device=self.memory_state.device, dtype=self.memory_state.dtype))
        self.liquid.hebbian_trace.copy_(liquid["hebbian_trace"].to(device=self.liquid.hebbian_trace.device, dtype=self.liquid.hebbian_trace.dtype))
        self.liquid.version = int(liquid.get("version", 0))
        self.liquid.curiosity_events = int(liquid.get("curiosity_events", 0))
        self.memory.records = copy.deepcopy(episodic)
        self.semantic_memory.records = copy.deepcopy(semantic)
        self.sleep.replay = copy.deepcopy(sleep["replay"])
        self.sleep.consolidation_step = int(sleep.get("consolidation_step", 0))
        curiosity = state.get("curiosity")
        if curiosity is not None:
            if self.curiosity is None or not isinstance(curiosity, dict):
                raise ValueError("El estado de curiosidad no coincide con la configuración activa.")
            assessments = curiosity.get("assessments", [])
            contexts = curiosity.get("uncertainty_by_context", {})
            if not isinstance(assessments, list) or not isinstance(contexts, dict) or len(assessments) > self.curiosity.assessment_capacity or len(contexts) > self.curiosity.assessment_capacity:
                raise ValueError("El estado de curiosidad excede la capacidad configurada o es inválido.")
            restored: list[CuriosityDecision] = []
            for item in assessments:
                if not isinstance(item, dict) or not isinstance(item.get("signals"), dict):
                    raise ValueError("Una evaluación de curiosidad del checkpoint es inválida.")
                signals = CuriositySignals(**item["signals"])
                restored.append(CuriosityDecision(float(item["priority"]), str(item["action"]), bool(item["blocked"]), bool(item["requires_approval"]), tuple(str(reason) for reason in item.get("reasons", [])), signals))
            self.curiosity.assessments = restored
            self.curiosity.uncertainty_by_context = {str(key): float(value) for key, value in contexts.items()}
