from router_logit_noise_probe import probe


result = probe(seed=17, sigma=0.2)
assert result["deterministic_repeat"] is True
assert 0.0 <= result["noisy_coverage"] <= 1.0
assert 0.0 <= result["noisy_concentration"] <= 1.0
assert probe(seed=17, sigma=0.2) == result
assert probe(seed=17, sigma=0.0)["clean_coverage"] == result["clean_coverage"]
print("router_logit_noise_probe: OK")
