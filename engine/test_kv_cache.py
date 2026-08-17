"""Comprueba que el prefill materialice KV-cache y conserve logits autoregresivos."""
from __future__ import annotations

import torch

from aethel_model import AethelConfig, AethelModel


def main() -> None:
    torch.manual_seed(17)
    model = AethelModel(AethelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=1, n_experts=2, active_experts=1, max_seq_len=16))
    model.eval()
    tokens = torch.tensor([[4, 5, 6, 7, 8, 9]], dtype=torch.long)
    with torch.no_grad():
        full_logits, _, _ = model(tokens)
        prefill_logits, _, caches = model(tokens[:, :4], kv_caches=[None] * 2)
        cached_first, _, first_caches = model(tokens[:, 4:5], start_pos=4, kv_caches=caches)
        cached_second, _, next_caches = model(tokens[:, 5:], start_pos=5, kv_caches=first_caches)
    assert caches is not None and first_caches is not None and next_caches is not None
    assert all(cache is not None and cache[0].shape[1] == 4 for cache in caches)
    assert all(cache is not None and cache[0].shape[1] == 6 for cache in next_caches)
    assert torch.allclose(prefill_logits, full_logits[:, :4], rtol=1e-4, atol=1e-5)
    assert torch.allclose(cached_first, full_logits[:, 4:5], rtol=1e-4, atol=1e-5)
    assert torch.allclose(cached_second, full_logits[:, 5:], rtol=1e-4, atol=1e-5)
    print("kv_cache_autoregressive OK")


if __name__ == "__main__":
    main()
