#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##
# Run bench_prefill under ncu for a given engine, capturing a full .ncu-rep.
# Usage: run_ncu_prefill.sh <engine> <out.ncu-rep> [extra ncu args...]
# engine: modular_max | tridao | flashinfer
# Runs --no-kineto so the attention kernel launches once and syncs; ncu attaches
# with --target-processes all (bazel run spawns a launcher process tree).
set -euo pipefail

engine="${1:?engine required: modular_max|tridao|flashinfer}"
out="${2:?output .ncu-rep path required}"
shift 2
extra=("$@")

cd ~/main

# q_len=28160, cache_len=0 => kv_len=28160 (pure prefill). batch=2, deepseek-v2 GQA.
common_args=( --engine "$engine" --batch_size 2 --q_len 28160 --cache_len 0 \
  --num_q_heads 32 --num_kv_heads 4 --head_dim 128 --causal 1 --window_size -1 \
  --num_iters 1 --no-kineto )

case "$engine" in
  modular_max) kname="regex:mha" ;;
  tridao)      kname="regex:kernel_cutlass.*[Ff]lash|[Ff]lashAttent|fmha" ;;
  flashinfer)  kname="regex:Sm100Fmha|fmha|[Ff]lash" ;;
  *) echo "unknown engine $engine" >&2; exit 1 ;;
esac

echo "=== ncu profile: engine=$engine -> $out ==="
# Build the binary once (bazel run triggers build), then ncu wraps the bazel
# launcher with --target-processes all so the python child is profiled.
set +e
./bazelw run //max/kernels/benchmarks/misc/comparison:bench_prefill -- \
  "${common_args[@]}" >/dev/null 2>&1
build_rc=$?
set -e
if [ $build_rc -ne 0 ]; then
  echo "bazel build/run of bench_prefill failed (rc=$build_rc); cannot profile" >&2
  exit $build_rc
fi

# Now run under ncu. Use the bazel run wrapper as the app; ncu attache to all.
ncu --target-processes all \
  --set full \
  --clock-control base \
  --kernel-name "$kname" \
  --launch-skip 0 --launch-count 1 \
  --replay-mode kernel \
  --import-source yes \
  "${extra[@]}" \
  -o "$out" \
  -- ./bazelw run //max/kernels/benchmarks/misc/comparison:bench_prefill -- \
     "${common_args[@]}"
echo "NCU_EXIT:$? out=$out"
