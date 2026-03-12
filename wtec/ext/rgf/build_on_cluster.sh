#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
BLAS_BACKEND=none

if ! command -v mpicc >/dev/null 2>&1; then
  echo "mpicc is required to build wtec_rgf_runner" >&2
  exit 1
fi

if [[ -n "${MKLROOT:-}" ]]; then
  BLAS_BACKEND=mkl
elif command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -qi openblas; then
  BLAS_BACKEND=openblas
fi

echo "Building wtec_rgf_runner with BLAS_BACKEND=${BLAS_BACKEND}" >&2
make -C "$ROOT" clean >/dev/null 2>&1 || true
make -C "$ROOT" MPICC=mpicc BLAS_BACKEND="$BLAS_BACKEND"
"$ROOT/build/wtec_rgf_runner" --probe
