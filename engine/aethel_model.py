import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

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
        
        self.experts = nn.ModuleList([SwiGLU(config.dim, hidden_dim) for _ in range(config.n_experts)])
        self.last_load = [0.0] * config.n_experts
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.shape[0]
        
        # Logits del Router
        router_logits = self.gate(x_flat)
        # Probabilidades completas del router (para pérdida auxiliar exacta)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Selección Top-K
        topk_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
        
        final_hidden_states = torch.zeros(
            (num_tokens, dim), dtype=x.dtype, device=x.device
        )
        
        # Máscara de enrutamiento
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.n_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
                
            current_state = x_flat[top_x]
            current_hidden_states = expert_layer(current_state) * topk_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
            
        # --- PÉRDIDA AUXILIAR CORREGIDA DE BALANCEO DE CARGA ---
        # Fracción de tokens asignados a cada experto
        tokens_per_expert = torch.bincount(selected_experts.flatten(), minlength=self.n_experts).float() / (num_tokens * self.top_k)
        self.last_load = [round(float(value) * 100, 4) for value in tokens_per_expert.detach().cpu()]
        # Probabilidad promedio asignada a cada experto por el router
        router_prob_per_expert = router_probs.mean(dim=0)
        # Pérdida auxiliar = n_experts * sum(density * prob)
        aux_loss = self.n_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
            
        return final_hidden_states.view(batch_size, seq_len, dim), aux_loss

class Attention(nn.Module):
    def __init__(self, config: AethelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        new_kv_cache = (k, v) if kv_cache is not None else None

        # Expansión GQA (Grouped-Query Attention)
        k_rep = torch.repeat_interleave(k, self.n_rep, dim=2)
        v_rep = torch.repeat_interleave(v, self.n_rep, dim=2)

        q_trans = q.transpose(1, 2)
        k_trans = k_rep.transpose(1, 2)
        v_trans = v_rep.transpose(1, 2)

        # FlashAttention vía PyTorch scaled_dot_product_attention
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache)
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
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Slice de freqs_cis correspondiente a las posiciones actuales
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        
        total_aux_loss = 0.0
        new_kv_caches = [] if kv_caches is not None else None

        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, aux_loss, new_cache = layer(h, freqs_cis, kv_cache=cache_i)
            total_aux_loss += aux_loss
            self.last_expert_loads = layer.feed_forward.last_load
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)
            
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
