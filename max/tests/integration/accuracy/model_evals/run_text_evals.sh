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
# Runs all non-agentic text evals in sequence against a running server. Each
# eval writes to its own default out-dir (/tmp/<eval>-results), so DON'T pass
# --out-dir here. aime_eval runs twice: once for AIME25 (the default
# --dataset) and once for AIME26; its metric prefix and out-dir follow the
# dataset, so the AIME26 run only needs --dataset.
#
# Common flags (forwarded to all evals):
#   --base-url, --model, --sample-size, --seed, --workers
#
# Per-eval repeats (only forwarded to the evals that support --repeats):
#   --aime25-repeats N    repeats per problem for AIME25   (default: 16)
#   --aime26-repeats N    repeats per problem for AIME26   (default: 16)
#   --gpqa-repeats N      repeats per question for GPQA    (default: 5)
#   --aa-lcr-repeats N    repeats per question for AA-LCR  (default: 5)
#
# Example:
#   ./bazelw run //max/tests/integration/accuracy/model_evals:text_evals_local -- \
#     --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \
#     --sample-size 2 --aime25-repeats 1 --aime26-repeats 1 \
#     --gpqa-repeats 1 --aa-lcr-repeats 1
#
# A gated eval whose HF token lacks access skips (exit 0); a real failure is
# reported and the remaining evals still run. Exit is non-zero if any failed.
set -uo pipefail

RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
BASE="${RUNFILES}/_main/max/tests/integration/accuracy/model_evals"

# Parse per-eval repeats flags; everything else is forwarded to all evals.
AIME25_REPEATS=()
AIME26_REPEATS=()
GPQA_REPEATS=()
AA_LCR_REPEATS=()
COMMON_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aime25-repeats) AIME25_REPEATS=(--repeats "$2"); shift 2 ;;
    --aime26-repeats) AIME26_REPEATS=(--repeats "$2"); shift 2 ;;
    --gpqa-repeats)   GPQA_REPEATS=(--repeats "$2");   shift 2 ;;
    --aa-lcr-repeats) AA_LCR_REPEATS=(--repeats "$2"); shift 2 ;;
    *)                COMMON_ARGS+=("$1");              shift   ;;
  esac
done

# Run labels, not binary names: aime25_eval and aime26_eval both run the
# aime_eval binary, and the AIME26 run selects its dataset via EXTRA below.
EVALS=(aime25_eval aime26_eval gpqa_eval hle_eval aa_lcr_eval aa_omniscience_eval)

summary=()
rc_all=0
for e in "${EVALS[@]}"; do
  case "$e" in
    aime25_eval|aime26_eval) bin_name=aime_eval ;;
    *)                       bin_name=$e        ;;
  esac
  bin="${BASE}/${bin_name}"
  if [[ ! -x "$bin" ]]; then
    echo "ERROR: $e not found in runfiles at: $bin" >&2
    summary+=("$e: MISSING")
    rc_all=1
    continue
  fi
  EXTRA=()
  case "$e" in
    aime25_eval)
      EXTRA=("${AIME25_REPEATS[@]}")
      ;;
    aime26_eval)
      EXTRA=(--dataset math-ai/aime26 "${AIME26_REPEATS[@]}")
      ;;
    gpqa_eval)
      EXTRA=("${GPQA_REPEATS[@]}")
      ;;
    aa_lcr_eval)
      EXTRA=("${AA_LCR_REPEATS[@]}")
      ;;
  esac
  echo ""
  echo "==================== $e ===================="
  if "$bin" "${COMMON_ARGS[@]}" "${EXTRA[@]}"; then
    summary+=("$e: ok")
  else
    rc=$?
    echo "::warning::$e exited $rc"
    summary+=("$e: FAILED (exit $rc)")
    rc_all=1
  fi
done

echo ""
echo "==================== summary ===================="
for line in "${summary[@]}"; do
  echo "  $line"
done
exit "$rc_all"
