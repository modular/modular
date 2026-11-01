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
#
# VGI-Bench at OpenRouter AutoExacto parity, against any OpenAI-compatible
# endpoint — a local MAX/mach serve, or an authenticated external one.
#
# VGI-Bench is Seldon's multiple-choice video QA benchmark: 439 public
# questions over 323 long videos. OpenRouter scores provider endpoints on it
# (benchmark_type "vgi_bench") with its own TypeScript harness,
# https://github.com/OpenRouterTeam/benchmark-harness, which only talks to the
# OpenRouter Responses API. Like run_exacto_gpqa_local.sh, this runs an Inspect
# task that reproduces it instead (exacto_vgi_task.py, samples ported in
# exacto_vgi_openrouter.py, answers extracted by exacto_mcq.py; all three are
# unit-tested against the harness's own output). At parity:
#   * the dataset, Seldon-Technologies/VGIBench `default`/`train` at tag v1.0.1
#     (loaded by the commit the tag points at), all 439 questions;
#   * the video, resolved through the harness's committed media manifest to
#     https://vgi-bench-mirror.openrouter.ai/<videoId>.mp4 and sent by URL.
#     The manifest ships byte for byte, and its manifestHash is recorded on
#     every sample and in score.json;
#   * one user turn, the video_url part then the prompt, no system message;
#   * the prompt, byte for byte (options "A) ...", "Answer: $LETTER" last);
#   * temperature 0, one epoch, reasoning effort "high" (the harness CLI's
#     default, sent on every request), no output-token cap, no client timeout.
#     A Mach engine caps a request without max_tokens at its --max-new-tokens
#     (256 unless set), so serve it with that raised to the max length;
#   * the harness's MCQ extraction. It is a port of extract.ts, not openbench's
#     scorer: the two have drifted, and openbench only reads A-D where VGI-Bench
#     has up to E;
#   * failures scored as the harness scores them: a client error after the
#     request is sent (a 400 because the server could not fetch or decode the
#     video, a 413) scores 0; a 429 or 5xx that outlives 7 attempts is left out
#     of the score; a 401/403/404 stops the harness, and here it fails the
#     error budget instead. The accuracy is over every question not left out;
#   * streamed responses, as the harness always streams.
#
# Not at parity, and why:
#   * Chat completions instead of the Responses API, as for GPQA: OpenRouter
#     translates the harness's input_video part into the provider's own API,
#     which for an OpenAI-compatible provider is this video_url part.
#   * A 408 is retried by Inspect but scored 0 by the harness. A local server
#     does not send one.
#   * The server fetches every video itself, once per request, from
#     OpenRouter's mirror. That is the point (a provider fetches it too), but
#     it needs egress to vgi-bench-mirror.openrouter.ai and moves ~13 GB for a
#     full run (mean 30 MB, max 49.6 MB per video). Serving the files locally
#     changes the URL the model is asked about, and MAX caps file:// media at
#     MAX_SERVE_MAX_LOCAL_IMAGE_BYTES (20 MiB by default), which 237 of the 323
#     videos exceed, so this runner always sends the mirror URL.
#
# What MAX does with the URL (max/python/max/serve/router/_image_resolution.py):
# it streams the body in 256 KiB chunks with connect=10 s, read=30 s per chunk
# (no total cap), at most 4 redirects, through SSRF protection (the mirror is a
# public host, so nothing to allowlist), and charges every byte to the
# per-request media budget, MAX_SERVE_MAX_MEDIA_BYTES (100 MiB by default).
# The largest video, 49.6 MB, fits. A fetch failure or a video over the budget
# is a 400, which scores 0 here as it would on OpenRouter.
#
# A model that ignores the video still answers, and still gets a score: chance
# is ~0.27, and several published providers sit far below it. Check that the
# deployment under test actually consumes video before reading anything into a
# number from this runner.
#
# Usage:
#   run_exacto_vgi_local.sh [options]
#
# Options:
#   --url URL          Endpoint base URL (default: http://localhost:8000). A
#                       trailing /v1 is optional.
#   --api-key KEY      API key for an authenticated endpoint. Defaults to
#                       $EXACTO_API_KEY, then $OPENAI_API_KEY, then "dummy".
#                       Passed by environment, never written to disk or logged.
#   --model MODEL      Model name for the API `model` field. Default:
#                       auto-detect via GET {url}/v1/models.
#   --epochs N         Repeated passes over the dataset (default: 1, the
#                       harness's). A reference run for openrouter_reference.py
#                       needs 2 or more.
#   --limit N          Evaluate only the first N of the 439 questions, for a
#                       smoke run. Reported PARTIAL; never comparable.
#   --max-connections N  Concurrent requests (default: 8, the harness CLI's
#                       --concurrency). Each one makes the server fetch a video.
#   --temperature F    Override sampling temperature. Defaults to unset, which
#                       leaves the harness's 0 in place.
#   --reasoning-effort E
#                      reasoning_effort sent with every request (default: high,
#                       the harness's). "unset" sends none.
#   --reference-range LOW,HIGH
#                      Optional range published providers land in for THIS
#                       model; see internal/openrouter_reference.py
#                       --benchmark vgi_bench.
#   --reference-source S
#                      Where --reference-range came from.
#   --serve-config S   Describes the deployment under test, recorded verbatim
#                       in score.json.
#   --out-dir DIR      Results directory (default: /tmp/exacto-vgi-results).
#   --no-stream        Send non-streaming requests. Differs from the harness.
#   --skip-video-preflight
#                      Skip the one request with the smallest mirrored video
#                       (11 KB) that checks the endpoint takes a video_url.
#   --refresh          Rebuild the harness venv from scratch.
#   -h, --help         Show this help.
#
# Run the parity eval against a local server:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_vgi_local
#
# Quick smoke (2 questions) while iterating:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_vgi_local -- \
#     --limit 2
#
# Env overrides: EXACTO_CACHE, EXACTO_OPENBENCH_VERSION, EXACTO_API_KEY.

set -euo pipefail

# The same pinned venv as run_exacto_gpqa_local.sh; see the pins explained
# there. openbench itself is unused here, but it brings Inspect and datasets at
# the versions the GPQA runner is known to work with, and the venv is shared.
OPENBENCH_VERSION="${EXACTO_OPENBENCH_VERSION:-0.5.3}"
MCP_CONSTRAINT="mcp<2"
OPENAI_CONSTRAINT="openai<3"

# The harness's vgi_bench settings, restated so a run that overrides one is
# flagged as not comparable instead of passing silently.
EXPECTED_DATASET="Seldon-Technologies/VGIBench"
EXPECTED_TEMPERATURE="0.0"
EXPECTED_REASONING_EFFORT="high"
EXPECTED_MANIFEST_HASH="872691450ddc053ae0b374f554ff8eb6b19fe379aab29e22ec6a0a7ef37bb738"
# The smallest video in the manifest, for the preflight.
PREFLIGHT_VIDEO="https://vgi-bench-mirror.openrouter.ai/collision-bench-level1-scene1.mp4"
# rateLimitRetrySchedule: 6 retries, so 7 attempts.
MAX_ATTEMPTS=7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
CACHE="${EXACTO_CACHE:-$REPO_ROOT/.derived/cache}"
VENV="$CACHE/exacto-openbench-venv"

url="http://localhost:8000"
api_key="${EXACTO_API_KEY:-${OPENAI_API_KEY:-}}"
model=""
epochs=1
limit=""
max_connections=8
temperature=""
reasoning_effort="$EXPECTED_REASONING_EFFORT"
reference_range=""
reference_source=""
serve_config=""
out_dir="/tmp/exacto-vgi-results"
refresh=0
stream=1
video_preflight=1

usage() { sed -n '/^# VGI-Bench/,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p; }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --max-connections) max_connections="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --reference-range) reference_range="$2"; shift 2 ;;
    --reference-source) reference_source="$2"; shift 2 ;;
    --serve-config) serve_config="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --refresh) refresh=1; shift ;;
    --no-stream) stream=0; shift ;;
    --skip-video-preflight) video_preflight=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 1 ;;
  esac
done

url="${url%/}"; url="${url%/v1}"; url="${url%/}"

vpy() { env -u PYTHONPATH -u PYTHONHOME "$VENV/bin/python" "$@"; }

find_uv() {
  local uv
  uv="$(command -v uv || true)"
  [[ -z "$uv" && -x "$HOME/.local/bin/uv" ]] && uv="$HOME/.local/bin/uv"
  if [[ -z "$uv" ]]; then
    echo "ERROR: uv not found; install uv to provision the harness venv." >&2
    exit 1
  fi
  echo "$uv"
}

RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
find_module() {
  local candidate
  for candidate in \
    "$RUNFILES/_main/max/tests/integration/accuracy/model_evals/$1" \
    "$SCRIPT_DIR/$1"; do
    if [[ -f "$candidate" ]]; then echo "$candidate"; return; fi
  done
  echo "ERROR: $1 not found in runfiles or next to this script" >&2
  exit 1
}
REPORT="$(find_module exacto_report.py)"
TASK="$(find_module exacto_vgi_task.py)"
for sibling in exacto_vgi_openrouter.py exacto_vgi_media_manifest.json \
  exacto_mcq.py exacto_media.py exacto_streaming.py; do
  [[ -f "$(dirname "$TASK")/$sibling" ]] || {
    echo "ERROR: $sibling is not next to $TASK" >&2
    exit 1
  }
done

ensure_venv() {
  local uv; uv="$(find_uv)"
  if [[ "$refresh" == 1 ]]; then rm -rf "$VENV"; fi
  if [[ ! -x "$VENV/bin/bench" ]]; then
    echo "[exacto] provisioning openbench $OPENBENCH_VERSION in $VENV"
    mkdir -p "$CACHE"
    "$uv" venv "$VENV" >/dev/null
    "$uv" pip install --python "$VENV/bin/python" \
      "openbench==$OPENBENCH_VERSION" "$MCP_CONSTRAINT" "$OPENAI_CONSTRAINT"
  fi
  vpy - <<'PYCHECK'
import sys
from importlib.metadata import version
bad = []
if int(version("openai").split(".")[0]) >= 3:
    bad.append(f'openai {version("openai")} (need <3)')
if int(version("mcp").split(".")[0]) >= 2:
    bad.append(f'mcp {version("mcp")} (need <2)')
if bad:
    sys.exit(
        "ERROR: harness venv has incompatible pins: "
        + ", ".join(bad)
        + "\n       re-run with --refresh"
    )
PYCHECK
}

detect_model() {
  if [[ -n "$model" ]]; then return; fi
  echo "[exacto] auto-detecting served model from $url/v1/models"
  local ids count
  ids="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models" | vpy -c '
import json, sys
data = json.load(sys.stdin).get("data") or []
print("\n".join(str(m.get("id")) for m in data if m.get("id")))
' 2>/dev/null || true)"
  count="$(printf '%s' "$ids" | grep -c . || true)"
  if [[ "${count:-0}" -eq 0 ]]; then
    echo "ERROR: could not auto-detect a model from $url/v1/models; pass --model." >&2
    exit 1
  fi
  if [[ "$count" -gt 1 ]]; then
    {
      echo "ERROR: $url/v1/models lists $count models; pass --model with one of:"
      printf '%s\n' "$ids" | sed 's/^/         /'
    } >&2
    exit 1
  fi
  model="$ids"
  echo "[exacto] model: $model"
}

capture_endpoint_meta() {
  local out="$1"
  local models_json model_json
  models_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models" 2>/dev/null || echo '')"
  model_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models/$model" 2>/dev/null || echo '')"
  vpy - "$out" "$url" "$model" "${serve_config:-}" \
    "$models_json" "$model_json" <<'PYMETA'
import json, subprocess, sys

out, url, model, serve_config, models_json, model_json = sys.argv[1:7]


def maybe_json(raw):
    try:
        return json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        return None


def git_head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


meta = {
    "base_url": url,
    "served_model": model,
    "models_endpoint": maybe_json(models_json),
    "model_endpoint": maybe_json(model_json),
    "serve_config": maybe_json(serve_config) or (serve_config or None),
    "client_commit": git_head(),
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
PYMETA
}

# Text first, so a wrong model name or a dead endpoint is told apart from a
# server that rejects video.
preflight_endpoint() {
  local resp code payload label body
  for label in text video; do
    if [[ "$label" == video ]]; then
      [[ "$video_preflight" == 1 ]] || continue
      body="{\"model\": \"$model\", \"max_tokens\": 8, \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"video_url\", \"video_url\": {\"url\": \"$PREFLIGHT_VIDEO\"}}, {\"type\": \"text\", \"text\": \"Reply with OK.\"}]}]}"
    else
      body="{\"model\": \"$model\", \"max_tokens\": 8, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
    fi
    resp="$(curl -s -m 180 -w '\n%{http_code}' \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${api_key:-dummy}" \
      -d "$body" "$url/v1/chat/completions" || true)"
    code="${resp##*$'\n'}"
    payload="${resp%$'\n'*}"
    if [[ "$code" != "200" ]]; then
      cat >&2 <<MSG
ERROR: the endpoint rejected a $label chat completion (HTTP ${code:-no response})
       for model '$model'.

       Nothing would be measured, so this stops before the run. For the video
       request, the server fetches $PREFLIGHT_VIDEO
       itself: check that it can reach that host and that the model takes
       video. --skip-video-preflight skips this check.
       Response: $(printf '%s' "$payload" | tr -d '\n' | head -c 300)
MSG
      exit 1
    fi
    echo "[exacto] endpoint preflight: $label chat completion OK for '$model'"
  done
}

ensure_venv
detect_model
preflight_endpoint

# The eval runs from inside out_dir, and `bazel run` starts in the runfiles
# tree, so resolve a relative path against the caller's directory first.
[[ "$out_dir" = /* ]] || out_dir="${BUILD_WORKING_DIRECTORY:-$PWD}/$out_dir"
mkdir -p "$out_dir"
LOG_DIR="$out_dir/logs"
rm -rf "$LOG_DIR"

provider="openai-stream"
[[ "$stream" == 0 ]] && provider="openai-chat"

args=(
  eval "$TASK@vgi_bench"
  --model "$provider/local/$model"
  --epochs "$epochs"
  --max-connections "$max_connections"
  --log-format json
  --log-dir "$LOG_DIR"
  --display plain
  # No client timeout and no SDK retries, as for GPQA. Inspect retries a 429,
  # a 5xx or a dropped stream itself, up to the harness's attempt count.
  -M timeout=null
  -M max_retries=0
  --max-retries "$MAX_ATTEMPTS"
)
# A request that still fails after its retries is left out of the score, as
# the harness skips it, rather than stopping the eval. Inspect stops the eval
# once those pass the error budget exacto_report enforces afterwards, so a
# server that dies mid-run fails within the budget instead of retrying every
# remaining sample for hours.
error_budget="${EVAL_MAX_ERROR_RATE:-0.10}"
if awk -v b="$error_budget" 'BEGIN { exit !(b > 0 && b < 1) }'; then
  args+=(--fail-on-error "$error_budget")
else
  args+=(--no-fail-on-error)
fi
[[ -n "$limit" ]] && args+=(--limit "$limit")
[[ -n "$temperature" ]] && args+=(--temperature "$temperature")
[[ "$reasoning_effort" != unset ]] && args+=(--reasoning-effort "$reasoning_effort")

echo "[exacto] running vgi_bench: epochs=$epochs limit=${limit:-439(all)} temp=${temperature:-0} reasoning_effort=$reasoning_effort provider=$provider"
(
  cd "$out_dir"
  env -u PYTHONPATH -u PYTHONHOME \
    LOCAL_BASE_URL="$url/v1" \
    LOCAL_API_KEY="${api_key:-dummy}" \
    "$VENV/bin/python" -c '
import sys
sys.path.insert(0, sys.argv[1])
import exacto_streaming
from inspect_ai._cli.main import main
sys.argv = ["inspect"] + sys.argv[2:]
sys.exit(main())
' "$(dirname "$TASK")" "${args[@]}"
)

LOG_JSON="$(find "$LOG_DIR" -name '*.json' -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2-)"
if [[ -z "$LOG_JSON" || ! -f "$LOG_JSON" ]]; then
  echo "ERROR: no inspect JSON log found under $LOG_DIR" >&2
  exit 1
fi
echo "[exacto] harness log: $LOG_JSON"

ENDPOINT_META="$out_dir/endpoint.json"
capture_endpoint_meta "$ENDPOINT_META"

REPORT_ARGS=()
[[ -n "$reference_range" ]] && REPORT_ARGS+=(--reference-range "$reference_range")
[[ -n "$reference_source" ]] && REPORT_ARGS+=(--reference-source "$reference_source")

vpy "$REPORT" \
  --harness openbench \
  --log "$LOG_JSON" \
  --dataset vgi_bench \
  --out-dir "$out_dir" \
  --endpoint-meta "$ENDPOINT_META" \
  "${REPORT_ARGS[@]}" \
  --metric-prefix EXACTO_VGI

vpy - "$out_dir/score.json" "$EXPECTED_DATASET" "$EXPECTED_TEMPERATURE" \
  "$EXPECTED_REASONING_EFFORT" "$EXPECTED_MANIFEST_HASH" "$limit" <<'PYWARN'
import json, sys
score_path, want_dataset, want_temp, want_effort, want_hash, limit = sys.argv[1:7]
s = json.load(open(score_path))
if s.get("dataset_name") != want_dataset:
    print(f"::warning::dataset is {s.get('dataset_name')}, expected {want_dataset}: "
          "score is not comparable to OpenRouter's.")
if str(s.get("temperature")) != want_temp:
    print(f"::warning::temperature is {s.get('temperature')}, expected {want_temp}: "
          "score is not comparable to OpenRouter's.")
if s.get("reasoning_effort") != want_effort:
    print(f"::warning::reasoning_effort is {s.get('reasoning_effort')}, expected "
          f"{want_effort}: score is not comparable to OpenRouter's.")
if s.get("media_manifest_hash") != want_hash:
    print(f"::warning::media manifest {s.get('media_manifest_hash')} is not the "
          "harness's: the videos may differ from OpenRouter's.")
if limit:
    print(f"::warning::PARTIAL run (--limit {limit}): a subset of the 439 "
          "questions has a different denominator than the published score.")
if s.get("model_errors_scored_wrong"):
    print(f"::warning::{s['model_errors_scored_wrong']} requests were rejected "
          "by the server and scored 0, as OpenRouter scores them. Check the "
          "explanations in results.jsonl before blaming the model.")
PYWARN

echo "[exacto] results: $out_dir/score.json"
