#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/engine"

python3 test_triton_gpu.py

echo "La validación anterior cubre únicamente el kernel SwiGLU integrado. La atención causal y el routing/dispatch MoE requieren kernels Triton adicionales y sus propias pruebas GPU antes de declararse disponibles."
