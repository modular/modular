#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##
#
# Script to warm cache for all models used in pipeline verification.
# Uses `./bazelw run max -- warm-cache` to compile models for target GPUs
# without requiring physical GPU hardware.
#
# Usage:
#   ./GraphCompiler/warm_cache_pipeline_models.sh [--target TARGET] [--filter FILTER] [--dry-run]
#
# Examples:
#   # Build all GPU models for CUDA (H100)
#   ./GraphCompiler/warm_cache_pipeline_models.sh --target cuda:sm_90
#
#   # Build all GPU models for AMD (MI300X)
#   ./GraphCompiler/warm_cache_pipeline_models.sh --target hip:gfx942
#
#   # Build only CPU models
#   ./GraphCompiler/warm_cache_pipeline_models.sh --filter cpu
#
#   # Build only multi-GPU models for CUDA
#   ./GraphCompiler/warm_cache_pipeline_models.sh --target cuda:sm_90 --filter multi-gpu
#
#   # Dry run to see what would be built
#   ./GraphCompiler/warm_cache_pipeline_models.sh --dry-run
#

set -euo pipefail

# Default target (CUDA Ampere architecture)
TARGET="${TARGET:-cuda:sm_80}"
FILTER=""
DRY_RUN=false
VERBOSE=false

# Change to repo root
cd "$(dirname "$0")/.."

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --target TARGET    GPU target for compilation (default: cuda:sm_80)
                     Supported targets:
                       - cuda          (default: sm_80 - Ampere)
                       - cuda:sm_80    (Ampere: A100, RTX 30xx)
                       - cuda:sm_90    (Hopper: H100)
                       - cuda:sm_100   (Blackwell: B200)
                       - hip           (default: gfx942 - MI300X)
                       - hip:gfx942    (AMD MI300X)
  --filter FILTER    Filter models to build:
                       - cpu           Only CPU-compatible models
                       - gpu           Only GPU-compatible models (single GPU)
                       - multi-gpu     Only multi-GPU models (nvidia-multi tag)
                       - small         Only small models (exclude "big" tag)
  --dry-run          Print commands without executing
  --verbose          Show verbose output
  -h, --help         Show this help message

Examples:
  $0 --target cuda:sm_90                    # H100 GPU
  $0 --target cuda:sm_100                   # B200 GPU
  $0 --target hip:gfx942                    # AMD MI300X
  $0 --target cuda:sm_90 --filter multi-gpu # Multi-GPU models for H100
  $0 --filter cpu                           # CPU models only
EOF
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  --target)
    TARGET="$2"
    shift 2
    ;;
  --filter)
    FILTER="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --verbose)
    VERBOSE=true
    shift
    ;;
  -h | --help)
    usage
    ;;
  *)
    echo "Unknown option: $1"
    usage
    ;;
  esac
done

# Validate target
case "$TARGET" in
cuda | cuda:sm_80 | cuda:sm_90 | cuda:sm_100 | hip | hip:gfx942 | hip:gfx1100) ;;
*)
  echo "Warning: Unknown target '$TARGET'. Proceeding anyway..."
  ;;
esac

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_warm_cache() {
  local model="$1"
  local devices="$2"
  local description="$3"

  local bazel_config=""
  local cmd="./bazelw run ${bazel_config}max -- warm-cache --model $model --devices $devices --target $TARGET"

  if $DRY_RUN; then
    echo "[DRY RUN] $cmd"
    echo "  # $description"
    return 0
  fi

  log "Building: $model ($description)"
  if $VERBOSE; then
    echo "  Command: $cmd"
  fi

  if ! $cmd 2>&1; then
    log "ERROR: Failed to build $model"
    return 1
  fi

  log "SUCCESS: $model"
}

# Define all pipelines from verify_pipelines.py
# Model paths are sourced from PIPELINE_ORACLES in create_pipelines.py
# Format: model_path|devices|tags|description
# Tags: big, nvidia-only, nvidia-multi, float8-support, no-h100, no-b200, 8xb200, tts

declare -a PIPELINES=(
  # ========== Robust Pipelines ==========
  # Model paths from create_pipelines.py PIPELINE_ORACLES
  "meta-llama/Meta-Llama-3-8B-Instruct|cpu,gpu|big|Llama 3 8B float32"
  "meta-llama/Llama-3.1-8B-Instruct|cpu,gpu|big|Llama 3.1 8B float32"
  "sentence-transformers/all-mpnet-base-v2|cpu,gpu||MPNet embeddings"
  "unsloth/gpt-oss-20b-BF16|gpu||GPT-OSS 20B bfloat16"
  "allenai/OLMo-1B-hf|cpu,gpu||OLMo 1B float32"

  # ========== Brittle Pipelines ==========
  # Llama family
  "meta-llama/Meta-Llama-3-8B-Instruct|gpu||Llama 3 8B bfloat16"
  "meta-llama/Llama-3.1-8B-Instruct|gpu||Llama 3.1 8B bfloat16"
  # data-parallel uses same model_path as base Llama 3.1
  "meta-llama/Llama-3.1-8B-Instruct|gpu|nvidia-multi,no-h100|Llama 3.1 8B data-parallel bfloat16"
  # FP8 static uses RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 (not the -float8-static suffix)
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8|gpu|float8-support|Llama 3.1 8B FP8 static"
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic|gpu|float8-support|Llama 3.1 8B FP8 dynamic"
  "meta-llama/Llama-3.2-1B|gpu|nvidia-only|Llama 3.2 1B bfloat16"
  "meta-llama/Llama-3.3-70B-Instruct|gpu|nvidia-multi|Llama 3.3 70B bfloat16"

  # Mistral family
  "mistralai/Mistral-Nemo-Instruct-2407|gpu|big|Mistral Nemo bfloat16"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503|gpu|big|Mistral Small 24B bfloat16"

  # Vision models
  "OpenGVLab/InternVL3-1B-Instruct|gpu|nvidia-only|InternVL3 1B bfloat16"
  "OpenGVLab/InternVL3-8B-Instruct|gpu|nvidia-multi|InternVL3 8B bfloat16"
  "OpenGVLab/InternVL3-14B-Instruct|gpu|nvidia-multi|InternVL3 14B bfloat16"
  "OpenGVLab/InternVL3-38B-Instruct|gpu|nvidia-multi|InternVL3 38B bfloat16"
  "OpenGVLab/InternVL3_5-8B-Instruct|gpu|nvidia-multi|InternVL3.5 8B bfloat16"
  "mistral-community/pixtral-12b|gpu|big|Pixtral 12B bfloat16"

  # Qwen family
  "Qwen/Qwen2.5-7B-Instruct|gpu|nvidia-only|Qwen2.5 7B bfloat16"
  "Qwen/Qwen2.5-VL-3B-Instruct|gpu||Qwen2.5 VL 3B bfloat16"
  "Qwen/Qwen2.5-VL-7B-Instruct|gpu|nvidia-multi|Qwen2.5 VL 7B bfloat16"
  "Qwen/Qwen2.5-VL-32B-Instruct|gpu|nvidia-multi|Qwen2.5 VL 32B bfloat16"
  "Qwen/Qwen3-8B|gpu|big,nvidia-only|Qwen3 8B bfloat16"

  # Other models
  "allenai/olmOCR-2-7B-1025-FP8|gpu|nvidia-only|OLMoOCR 7B FP8"
  "allenai/OLMo-2-1124-7B|cpu,gpu|big|OLMo 2 7B float32"
  "HuggingFaceM4/Idefics3-8B-Llama3|gpu|big,nvidia-only|Idefics3 8B bfloat16"
  "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct|cpu,gpu|big|EXAONE 7.8B float32"
  "microsoft/Phi-3.5-mini-instruct|gpu||Phi 3.5 mini bfloat16"
  "microsoft/phi-4|gpu|big|Phi 4 bfloat16"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4|gpu|nvidia-only|Llama 3.1 8B GPTQ"
  "kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit|gpu|nvidia-only|DeepSeek R1 Distill GPTQ"

  # DeepSeek family
  "deepseek-ai/DeepSeek-V2-Lite-Chat|gpu|big,nvidia-only|DeepSeek V2 Lite bfloat16"
  "kathywu95/deepseek-v3-small-random|gpu|nvidia-only,no-b200|DeepSeek V3 small bfloat16"
  "kathywu95/deepseek-v3-small-random-fp8|gpu|nvidia-only,no-h100|DeepSeek V3 small FP8"
  "deepseek-ai/DeepSeek-R1|gpu|nvidia-multi,8xb200|DeepSeek R1 FP8"

  # Gemma family
  "google/gemma-3-1b-it|gpu||Gemma 3 1B bfloat16"
  "google/gemma-3-12b-it|gpu|big|Gemma 3 12B bfloat16"
  "google/gemma-3-27b-it|gpu|nvidia-multi|Gemma 3 27B bfloat16"
  "RedHatAI/gemma-3-27b-it-FP8-dynamic|gpu|big,float8-support|Gemma 3 27B FP8 single GPU"
  "RedHatAI/gemma-3-27b-it-FP8-dynamic|gpu|big,float8-support,nvidia-multi|Gemma 3 27B FP8 multi GPU"

  # Misc
  "HKUSTAudio/Llasa-8B|gpu|big,tts|Llasa 8B bfloat16"
  # LoRA models use base model path from create_pipelines.py
  "HuggingFaceTB/SmolLM2-360M-Instruct|gpu|nvidia-only|SmolLM2 360M LoRA bfloat16"
  # FP8-dynamic-BF16-LoRA uses RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic as base
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic|gpu|nvidia-only,no-h100,float8-support|Llama 3.1 8B FP8 LoRA"
)

# Function to check if a pipeline matches the filter
matches_filter() {
  local devices="$1"
  local tags="$2"

  case "$FILTER" in
  "")
    return 0 # No filter, match all
    ;;
  "cpu")
    [[ "$devices" == *"cpu"* ]]
    ;;
  "gpu")
    # Single GPU (not multi-GPU)
    [[ "$devices" == *"gpu"* ]] && [[ "$tags" != *"nvidia-multi"* ]] && [[ "$tags" != *"8xb200"* ]]
    ;;
  "multi-gpu")
    [[ "$tags" == *"nvidia-multi"* ]] || [[ "$tags" == *"8xb200"* ]]
    ;;
  "small")
    [[ "$tags" != *"big"* ]]
    ;;
  *)
    echo "Unknown filter: $FILTER" >&2
    return 1
    ;;
  esac
}

# Function to determine devices for warm-cache
get_devices_for_target() {
  local devices="$1"
  local tags="$2"

  # For CPU-only models
  if [[ "$devices" == "cpu" ]]; then
    echo "cpu"
    return
  fi

  # For multi-GPU models, determine device count
  if [[ "$tags" == *"8xb200"* ]]; then
    echo "gpu:0,1,2,3,4,5,6,7"
  elif [[ "$tags" == *"nvidia-multi"* ]]; then
    # Default to 4 GPUs for multi-GPU models, or 2 for B200 multi
    if [[ "$TARGET" == *"sm_100"* ]]; then
      echo "gpu:0,1" # B200 multi uses 2 GPUs
    else
      echo "gpu:0,1,2,3" # H100 multi uses 4 GPUs
    fi
  else
    echo "gpu:0" # Single GPU
  fi
}

# Function to check if model should be skipped based on target and tags
should_skip() {
  local tags="$1"

  # Skip nvidia-only models for HIP targets
  if [[ "$TARGET" == hip* ]] && [[ "$tags" == *"nvidia-only"* ]]; then
    return 0
  fi

  # Skip non-float8 models if building for targets that don't support float8
  if [[ "$TARGET" == hip* ]] && [[ "$tags" == *"float8-support"* ]]; then
    return 0
  fi

  # Skip H100-excluded models for sm_90
  if [[ "$TARGET" == "cuda:sm_90" ]] && [[ "$tags" == *"no-h100"* ]]; then
    return 0
  fi

  # Skip B200-excluded models for sm_100
  if [[ "$TARGET" == "cuda:sm_100" ]] && [[ "$tags" == *"no-b200"* ]]; then
    return 0
  fi

  # Skip 8xb200 models unless explicitly building for B200 8x
  if [[ "$tags" == *"8xb200"* ]] && [[ "$TARGET" != "cuda:sm_100" ]]; then
    return 0
  fi

  return 1
}

# Main execution
log "Starting warm-cache for pipeline verification models"
log "Target: $TARGET"
if [[ -n "$FILTER" ]]; then
  log "Filter: $FILTER"
fi

total=0
skipped=0
success=0
failed=0

for pipeline in "${PIPELINES[@]}"; do
  IFS='|' read -r model devices tags description <<<"$pipeline"

  ((total++)) || true

  # Check filter
  if ! matches_filter "$devices" "$tags"; then
    if $VERBOSE; then
      log "SKIP (filter): $model"
    fi
    ((skipped++)) || true
    continue
  fi

  # Check if should skip based on target
  if should_skip "$tags"; then
    if $VERBOSE; then
      log "SKIP (target incompatible): $model"
    fi
    ((skipped++)) || true
    continue
  fi

  # Determine devices for this pipeline
  warm_cache_devices=$(get_devices_for_target "$devices" "$tags")

  # Skip CPU models when building for GPU target
  if [[ "$warm_cache_devices" == "cpu" ]] && [[ "$TARGET" != "cpu" ]]; then
    if $VERBOSE; then
      log "SKIP (CPU model, GPU target): $model"
    fi
    ((skipped++)) || true
    continue
  fi

  # Run warm-cache
  if run_warm_cache "$model" "$warm_cache_devices" "$description"; then
    ((success++)) || true
  else
    ((failed++)) || true
  fi
done

log "============================================"
log "Summary:"
log "  Total pipelines: $total"
log "  Skipped: $skipped"
log "  Success: $success"
log "  Failed: $failed"
log "============================================"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
