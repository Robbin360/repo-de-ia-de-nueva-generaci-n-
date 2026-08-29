import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from triton_bridge import causal_decode_attention, fused_swiglu, moe_dispatch_combine_reference, top2_router
from router_auxiliary import router_balance_auxiliary_loss, router_entropy_regularization_loss

@dataclass
class AethelConfig:
    vocab_size: int = 100288  # tiktoken cl100k_base alineado
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    n_experts: int = 8
    active_experts: int = 2
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    router_bias_step: float = 0.01
    router_bias_limit: float = 0.25
    router_jitter_noise: float = 0.0
    require_triton: bool = False


def enforce_triton_prefill_contract(*, require_triton: bool, is_cuda: bool, is_decode: bool) -> None:
    """Impide que producción use SDPA para prefill cuando Triton es obligatorio.

    La decodificación token a token dispone de una ruta Triton separada. El
    prefill causal por bloques todavía no tiene un kernel GPU validado, por lo
    que en modo de producción debe fallar de manera explícita, no degradarse
    silenciosamente a SDPA.
    """
    if require_triton and is_cuda and not is_decode:
        raise RuntimeError(
            "Aethel exige un kernel Triton causal validado para prefill CUDA; "
            "SDPA no está permitido mientras esa ruta no exista."
        )


def enforce_triton_moe_dispatch_contract(*, require_triton: bool, is_cuda: bool) -> None:
    """Bloquea el loop de expertos PyTorch cuando la producción exige Triton.

    Top-2 Triton sólo selecciona expertos: no reemplaza el scatter, cómputo
    agrupado y combina. El modo estricto debe esperar un kernel completo y
    validado en GPU, no declarar esa ruta parcial como dispatch MoE.
    """
    if require_triton and is_cuda:
        raise RuntimeError(
            "Aethel exige un kernel Triton validado de dispatch/combina MoE; "
            "el bucle PyTorch de expertos no está permitido en producción."
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputa las frecuencias complejas para Rotary Position Embeddings (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica incrustaciones rotacionales (RoPE) sobre las consultas (Q) y llaves (K)."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # Broadcast sobre Batch y Heads
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, require_triton: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.require_triton = require_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(fused_swiglu(self.w1(x), self.w2(x), require_triton=self.require_triton and x.is_cuda))

class SparseMoE(nn.Module):
    def __init__(self, config: AethelConfig):
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_experts
        self.top_k = config.active_experts
        
        self.gate = nn.Linear(config.dim, config.n_experts, bias=False)
        
        # Dimensión oculta escalada para SwiGLU (estilo Llama / Mixtral)
        hidden_dim = int(8 * config.dim / 3)
        # Alineación con Tensor Cores (múltiplo de 256)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.experts = nn.ModuleList([SwiGLU(config.dim, hidden_dim, config.require_triton) for _ in range(config.n_experts)])
        self.last_load = [0.0] * config.n_experts
        self.last_entropy_loss = torch.zeros((), dtype=torch.float32)
        self.register_buffer("router_bias", torch.zeros(config.n_experts), persistent=True)
        self.register_buffer("load_ema", torch.full((config.n_experts,), 1.0 / config.n_experts), persistent=True)
        self.router_bias_step = config.router_bias_step
        self.router_bias_limit = config.router_bias_limit
        self.router_jitter_noise = config.router_jitter_noise
        self.last_routing_stats = {"entropy": 0.0, "max_load": 0.0, "imbalance": 0.0, "bias": [0.0] * config.n_experts, "selection_jitter_noise": 0.0}

    @torch.no_grad()
    def _update_load_balancer(self, tokens_per_expert: torch.Tensor) -> None:
        """Ajuste lento sin pérdida auxiliar adicional; no altera gradientes del router."""
        self.load_ema.mul_(0.95).add_(tokens_per_expert.detach().to(self.load_ema) * 0.05)
        target = 1.0 / self.n_experts
        correction = target - self.load_ema
        self.router_bias.add_(self.router_bias_step * correction).clamp_(-self.router_bias_limit, self.router_bias_limit)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.shape[0]
        enforce_triton_moe_dispatch_contract(
            require_triton=self.experts[0].require_triton,
            is_cuda=x.is_cuda,
        )
        
        # El sesgo sin pérdida sólo decide qué expertos pueden competir. Las
        # probabilidades, los pesos de combinación y las pérdidas se calculan
        # sobre logits crudos para que el controlador de balanceo no altere la
        # semántica del modelo ni amortigüe su señal de gradiente.
        router_logits = self.gate(x_flat)
        selection_scores = router_logits + self.router_bias.to(x_flat.dtype)
        # Ruido sólo en entrenamiento y sólo en la selección: rompe empates
        # tempranos del top-k sin contaminar las probabilidades densas, los pesos
        # de combinación, la entropía ni la inferencia determinista.
        active_jitter = self.router_jitter_noise if self.training else 0.0
        if active_jitter:
            selection_scores = selection_scores + torch.randn_like(selection_scores) * active_jitter
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Selección Top-K. El kernel Triton conserva el gradiente del router
        # usando PyTorch durante entrenamiento y acelera top-2 en inferencia.
        if not self.training and self.top_k == 2:
            _, selected_experts = top2_router(selection_scores, require_triton=self.experts[0].require_triton and x.is_cuda)
        else:
            _, selected_experts = torch.topk(selection_scores, self.top_k, dim=-1)
        topk_weights = router_probs.gather(dim=-1, index=selected_experts)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
        
        # Referencia explícita de la semántica que deberá preservar el kernel
        # Triton de capacidad/dispatch/combina. En CUDA estricto, el contrato
        # anterior bloquea esta ruta PyTorch antes de alcanzar este punto.
        final_hidden_states = moe_dispatch_combine_reference(
            x_flat,
            selected_experts,
            topk_weights,
            list(self.experts),
        )
            
        # --- PÉRDIDA AUXILIAR CORREGIDA DE BALANCEO DE CARGA ---
        # Fracción de tokens asignados a cada experto
        tokens_per_expert = torch.bincount(selected_experts.flatten(), minlength=self.n_experts).float() / (num_tokens * self.top_k)
        self.last_load = [round(float(value) * 100, 4) for value in tokens_per_expert.detach().cpu()]
        if self.training:
            self._update_load_balancer(tokens_per_expert)
        entropy = -(tokens_per_expert * tokens_per_expert.clamp_min(1e-9).log()).sum() / math.log(self.n_experts)
        self.last_routing_stats = {
            "entropy": float(entropy.detach().cpu()),
            "max_load": float(tokens_per_expert.max().detach().cpu()),
            "imbalance": float((tokens_per_expert - (1.0 / self.n_experts)).abs().mean().detach().cpu()),
            # Señales de asignación dura: no deben confundirse con la entropía
            # de las probabilidades suaves del router.
            "hard_coverage": float((tokens_per_expert > 0).float().mean().detach().cpu()),
            "hard_max_density": float(tokens_per_expert.max().detach().cpu()),
            "hard_min_density": float(tokens_per_expert.min().detach().cpu()),
            "bias": [float(value) for value in self.router_bias.detach().cpu()],
            "selection_jitter_noise": float(active_jitter),
        }
        # Probabilidad promedio asignada a cada experto por el router
        router_prob_per_expert = router_probs.mean(dim=0)
        # Pérdida auxiliar = n_experts * sum(density * prob)
        aux_loss = router_balance_auxiliary_loss(tokens_per_expert, router_prob_per_expert)
        self.last_entropy_loss = router_entropy_regularization_loss(router_probs)
            
        return final_hidden_states.view(batch_size, seq_len, dim), aux_loss

class Attention(nn.Module):
    def __init__(self, config: AethelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.require_triton = config.require_triton

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Manejo de KV Cache para inferencia autorregresiva O(1)
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)
        # El primer prefill no contiene entradas previas, pero debe devolver
        # claves/valores cuando el llamador solicitó conservar el contexto.
        new_kv_cache = (k, v) if use_kv_cache else None

        # Expansión GQA (Grouped-Query Attention)
        k_rep = torch.repeat_interleave(k, self.n_rep, dim=2)
        v_rep = torch.repeat_interleave(v, self.n_rep, dim=2)

        q_trans = q.transpose(1, 2)
        k_trans = k_rep.transpose(1, 2)
        v_trans = v_rep.transpose(1, 2)

        # Triton acelera el paso de decodificación con KV-cache. El prefill
        # causal por bloques aún no tiene kernel GPU validado y se bloquea
        # explícitamente si la configuración exige Triton en producción.
        is_decode = kv_cache is not None and T == 1 and mask is None
        enforce_triton_prefill_contract(
            require_triton=self.require_triton,
            is_cuda=x.is_cuda,
            is_decode=is_decode,
        )
        if is_decode:
            output = causal_decode_attention(q_trans, k_trans, v_trans, require_triton=self.require_triton and x.is_cuda)
        else:
            output = F.scaled_dot_product_attention(
                q_trans, k_trans, v_trans,
                attn_mask=mask,
                is_causal=(mask is None and kv_cache is None and T > 1)
            )
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output), new_kv_cache

class TransformerBlock(nn.Module):
    def __init__(self, config: AethelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SparseMoE(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, use_kv_cache)
        h = x + attn_out
        ff_out, aux_loss = self.feed_forward(self.ffn_norm(h))
        out = h + ff_out
        return out, aux_loss, new_kv_cache

class AethelModel(nn.Module):
    def __init__(self, config: AethelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.last_expert_loads = [0.0] * config.n_experts
        self.last_router_entropy_loss = torch.zeros((), dtype=torch.float32)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights entre embedding y proyector de salida (opcional/estándar)
        self.tok_embeddings.weight = self.output.weight

        # Registrar freqs_cis como buffer no persistente para movimiento automático de device
        freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Inicialización formal de pesos (Varianza escalada según profundidad)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        tokens: torch.Tensor, 
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        if memory_state is not None:
            h = h + memory_state.unsqueeze(1).to(h.dtype)
        
        # Slice de freqs_cis correspondiente a las posiciones actuales
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        
        total_aux_loss = 0.0
        total_router_entropy_loss = 0.0
        cache_requested = kv_caches is not None
        new_kv_caches = [] if cache_requested else None

        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, aux_loss, new_cache = layer(h, freqs_cis, kv_cache=cache_i, use_kv_cache=cache_requested)
            total_aux_loss += aux_loss
            total_router_entropy_loss += layer.feed_forward.last_entropy_loss
            self.last_expert_loads = layer.feed_forward.last_load
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)
            
        self.last_router_entropy_loss = total_router_entropy_loss
        h = self.norm(h)
        output = self.output(h)
        return output, total_aux_loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: torch.Tensor, 
        max_new_tokens: int = 50, 
        temperature: float = 0.7, 
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generación autorregresiva de texto con KV Cache activado."""
        was_training = self.training
        self.eval()
        B, T = prompt_tokens.shape
        tokens = prompt_tokens.clone()
        
        # Inicializar KV Caches vacías para cada capa
        kv_caches = [None] * len(self.layers)
        
        # Prefill phase
        logits, _, kv_caches = self.forward(tokens, start_pos=0, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]
        
        for cur_pos in range(T, T + max_new_tokens):
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_probs[sorted_indices_to_remove] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=-1)
            
            # Decode step: solo pasamos el nuevo token
            logits, _, kv_caches = self.forward(next_token, start_pos=cur_pos, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]

        if was_training:
            self.train()

        return tokens

if __name__ == "__main__":
    print("================================================================")
    print(" 🚀 VERIFICANDO ARQUITECTURA AETHEL (MOE + GQA + ROPE + KV-CACHE)")
    print("================================================================")
    config = AethelConfig()
    config.n_layers = 2  # Reducido para prueba rápida
    config.dim = 512
    config.n_heads = 8
    config.n_kv_heads = 2
    
    model = AethelModel(config)
    
    # 1. Test Forward Pass
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 16))
    logits, aux_loss, _ = model(dummy_tokens)
    print(f"✅ Forward Pass Exitoso!")
    print(f"   Logits Shape: {logits.shape}")
    print(f"   Aux Loss: {aux_loss.item():.4f}")
    
    # 2. Test Autoregressive Generation with KV Cache
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    print(f"✅ Generación Autorregresiva con KV Cache Exitosa!")
    print(f"   Prompt Tokens ({prompt.shape[1]}): {prompt[0].tolist()}")
    print(f"   Generated Tokens ({generated.shape[1]}): {generated[0].tolist()}")
