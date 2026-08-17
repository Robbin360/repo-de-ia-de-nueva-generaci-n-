#!/usr/bin/env bash
# Preflight real de Aethel: no emula GPU ni transforma requisitos incumplidos en éxito.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import torch

print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": torch.cuda.device_count(),
    "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
}))
PY

if ! python3 - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1)
PY
then
  echo '{"status":"SKIPPED","reason":"requires_at_least_two_cuda_gpus","next":"No se ejecutaron Triton ni FSDP"}'
  exit 0
fi

echo "[1/2] Validando kernel Triton implementado contra PyTorch en CUDA"
bash training/run_triton_gpu_validation.sh

echo "[2/2] Validando FSDP: entrenamiento, checkpoint de rango 0 y reanudación"
cd "$ROOT/engine"
python3 test_fsdp_gpu_e2e.py

echo '{"status":"VERIFIED","checks":["triton_swiglu","fsdp_checkpoint_resume"]}'
