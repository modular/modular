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
# Build and profile the SM100 FP8 index prefill scorer on the B200 Coder box.
#
# Produces a single-invocation .ncu-rep with the full metric set and
# source-correlation (Mojo built --debug-level=line-tables), against the shipped
# 2-CTA/SM, Q-resident, 1-consumer-WG (128 q-scales in registers) path. The
# resulting kernel launches at block (256,1,1) -- 1 producer + 1 consumer
# warpgroup. A 1-CTA/2-consumer-WG variant would launch at ~384 threads.
#
# See the ncu-analyze skill for caveats on warp-specialized kernel attribution:
# trust kernel duration (with --clock-control none --cache-control none), tensor
# pipe %, instruction mix, and the GMEM predicted-vs-measured sanity check;
# distrust stall attribution and occupancy commentary on this producer/consumer
# split.
#
# Run on the B200:  bash bench_fp8_index_prefill_ncu.sh [extra ncu flags]
set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
NCU=${NCU:-ncu}
TARGET=//max/kernels/benchmarks:gpu/nn/bench_fp8_index_prefill
# Kernel symbol the scorer routes the nh=32 prefill shape to.
SYM_REGEX='fp8_index_score_prefill_sm100'
# GLM-5.2 pure-prefill: batch=1, 2048 new tokens, no cached prefix, causal.
# Long enough to clear the nh=32 prefill route's 448-token-tile gate.
BIN_ARGS=(--batch_size=1 --seq_len=2048 --cache_len=0 --max_num_keys=2048)
OUT=${OUT:-fp8_index_prefill_shipped_ltables.ncu-rep}
NCU_FLAGS=("$@")

cd "$REPO"

# Line tables for `--import-source`; the default opt build carries none.
./bazelw build "$TARGET" --mojocopt=--debug-level --mojocopt=line-tables

# Pin the SM clock below the power-limited ceiling for comparable durations.
# The ncu-analyze skill notes setup-gpu-clock.sh may not take on a power-capped
# board; --clock-control none below keeps the run honest about that.
if ! sudo utils/setup-gpu-clock.sh 2>/dev/null; then
  echo "WARN: clock pin via setup-gpu-clock.sh did not confirm; pinning manually"
  sudo nvidia-smi -lgc 1100,1100
fi

BIN=$(readlink -f bazel-bin/max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill)

# `--run_benchmark=False` launches the kernel once; ncu does its own replay.
"$NCU" \
  --set full \
  --kernel-name "regex:${SYM_REGEX}" \
  --clock-control none --cache-control none \
  --import-source yes \
  ${NCU_FLAGS[@]+"${NCU_FLAGS[@]}"} \
  -o "$OUT" \
  "$BIN" "${BIN_ARGS[@]}" --run_benchmark=False

echo "Profile written to $OUT (run: ncu --import $OUT)"
# Restore the clock when done so you don't leave the box pinned.
sudo nvidia-smi -rgc 2>/dev/null || true
