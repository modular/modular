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
# tau2-bench airline — the agentic tool-calling half of OpenRouter's AutoExacto
# gate — against any OpenAI-compatible endpoint.
#
# This is the half of the gate we had no coverage for: the suite's other agentic
# evals (SWE-bench, GAIA, xbench) do not exercise multi-turn tool calling
# against a simulated user, which is what AutoExacto actually scores providers
# on. OpenRouter uses the AWS-AGI verified variant of the dataset, so this pins
# amazon-agi/tau2-bench-verified rather than Sierra's original.
#
# OpenRouter benchmarks with its own harness,
# https://github.com/OpenRouterTeam/benchmark-harness
# (src/benchmarks/tau-bench-airline), a TypeScript port that only talks to the
# OpenRouter API. Its 50 airline tasks and database are identical to this pin's.
# This runner matches its settings: agent temperature 0 and reasoning effort
# "high", simulator temperature 0 and reasoning effort "medium", 200-step cap,
# no request timeout, and up to 6 retries of a failed request.
# On GLM-5.3 the chat template renders a missing effort as "Max", so an agent
# run without one reasons longer than the harness asks for.
#
# The score is the mean binary reward over num_tasks x num_trials simulations,
# where each reward is the database-state check AND the communication checks,
# and a run that blows the 200-step cap scores 0.
#
# IMPORTANT — the user simulator is half the measurement, and which model plays
# it changes the score. OpenRouter pins it to gemini-2.5-flash ("We pin it to
# gemini-2.5-flash so agent scores stay comparable"), but we have no Gemini
# credential: the GOOGLE_API_KEY referenced by minimaxM3AgentClassDatasetEval
# was never provisioned as a repo secret. So this defaults to self-simulation —
# the endpoint under test plays the customer as well as the agent — which needs
# no new secret and matches how the HLE, AA-LCR and AA-Omniscience steps handle
# the same missing key. That number tracks our own runs over time but is NOT
# comparable to OpenRouter's leaderboard. For a comparable score, pass
# --user-llm openrouter/google/gemini-2.5-flash with OPENROUTER_API_KEY set
# (the route the harness itself uses), or gemini/gemini-2.5-flash with
# GEMINI_API_KEY or GOOGLE_API_KEY.
#
# Self-simulation is a weaker measurement than a pinned third-party simulator:
# one model plays both sides of the conversation, and it holds the scenario
# instructions while doing so. Treat a self-simulated score as a regression
# signal, not as our AutoExacto standing.
#
# Usage:
#   run_exacto_taubench_local.sh [options]
#
# Options:
#   --url URL          Endpoint base URL for the agent under test
#                       (default: http://localhost:8000). Trailing /v1 optional.
#   --api-key KEY      API key for an authenticated endpoint. Defaults to
#                       $EXACTO_API_KEY, then $OPENAI_API_KEY, then "dummy".
#   --model MODEL      Model name for the API `model` field. Default:
#                       auto-detect via GET {url}/v1/models.
#   --user-llm LLM     litellm name for the user simulator. Default: empty,
#                       meaning self-simulate with the endpoint under test (no
#                       extra credential). For OpenRouter parity pass
#                       openrouter/google/gemini-2.5-flash (needs
#                       OPENROUTER_API_KEY) or gemini/gemini-2.5-flash (needs
#                       GEMINI_API_KEY/GOOGLE_API_KEY).
#   --reasoning-effort E
#                      reasoning_effort sent with every agent request (default:
#                       high, the harness's). "unset" sends none, which leaves
#                       the choice to the chat template.
#   --user-reasoning-effort E
#                      reasoning_effort for the user simulator. Default: medium
#                       (the harness's) for a third-party simulator; under
#                       self-simulation, the agent's effort, since one endpoint
#                       plays both sides. "unset" sends none.
#   --num-trials N     Runs per task (default: 4, the published tau2 leaderboard
#                       convention, giving 200 graded simulations). Do not lower
#                       this to save time and then read the verdict: reward is
#                       binary, so 50 simulations carry a standard error of
#                       about 0.06, which is wider than the whole peer band.
#                       Anything below 4 is a smoke run, not a measurement.
#   --num-tasks N      Run only the first N tasks, for a smoke run. A subset has
#                       a different denominator than the published score, so it
#                       is reported PARTIAL and never "passes".
#   --task-split NAME  Task split (default: base, all 50 airline tasks).
#   --max-concurrency N  Concurrent simulations (default: 8).
#   --max-steps N      Step cap per simulation (default: 200, OpenRouter's).
#   --reference-user-llm LLM
#                      The simulator OpenRouter pins, used only as the
#                       comparability target: the run warns when --user-llm
#                       differs from it, and it is recorded in score.json so
#                       nothing downstream hardcodes the name. Default
#                       gemini/gemini-2.5-flash. This does NOT change which
#                       simulator runs; --user-llm does that.
#   --reference-range LOW,HIGH
#                      Optional range published providers land in for THIS
#                       model, shown next to the score as context. Provider
#                       spread is per-model, so there is no default; get it
#                       from the model's OpenRouter page.
#   --reference-source S
#                      Where --reference-range came from, e.g. an OpenRouter
#                       model slug.
#   --serve-config S   Describes the deployment under test (recipe name,
#                       quantization, KV dtype, spec-decode mode). Free text
#                       or JSON, recorded verbatim in score.json, because a
#                       score you cannot attribute to a config is not
#                       evidence.
#   --out-dir DIR      Results directory (default: /tmp/exacto-taubench-results).
#   --refresh          Re-clone the harness and rebuild its venv.
#   -h, --help         Show this help.
#
# Smoke run (2 tasks, 1 trial) while iterating:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_taubench_local -- \
#     --num-tasks 2 --max-concurrency 2
#
# Full parity run against a local server:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_taubench_local
#
# Env overrides: EXACTO_CACHE, EXACTO_TAU2_PIN, EXACTO_TAU2_PYTHON,
#               EXACTO_REFERENCE_USER_LLM, EXACTO_API_KEY, GEMINI_API_KEY,
#               OPENROUTER_API_KEY.

set -euo pipefail

# Pinned for reproducibility; this script is the single source of truth.
PIN="${EXACTO_TAU2_PIN:-864350a8971a8f8ee9e7b8472e2edc380a806b0c}"
HARNESS_REPO="https://github.com/amazon-agi/tau2-bench-verified"
HARNESS_PYTHON="${EXACTO_TAU2_PYTHON:-3.12}"

# The simulator OpenRouter pins. Single source of truth: it seeds the parity
# default, drives the comparability warning, and is recorded in score.json so
# nothing downstream has to hardcode the name.
DEFAULT_REFERENCE_USER_LLM="${EXACTO_REFERENCE_USER_LLM:-gemini/gemini-2.5-flash}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
CACHE="${EXACTO_CACHE:-$REPO_ROOT/.derived/cache}"
CHECKOUT="$CACHE/tau2-bench-verified"
VENV="$CHECKOUT/.venv"

url="http://localhost:8000"
api_key="${EXACTO_API_KEY:-${OPENAI_API_KEY:-}}"
model=""
user_llm=""
reasoning_effort="high"
user_reasoning_effort=""
num_trials=4
num_tasks=""
task_split="base"
max_concurrency=8
max_steps=200
reference_user_llm="$DEFAULT_REFERENCE_USER_LLM"
reference_range=""
reference_source=""
serve_config=""
out_dir="/tmp/exacto-taubench-results"
refresh=0

usage() { sed -n '15,128p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --user-llm) user_llm="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --user-reasoning-effort) user_reasoning_effort="$2"; shift 2 ;;
    --num-trials) num_trials="$2"; shift 2 ;;
    --num-tasks) num_tasks="$2"; shift 2 ;;
    --task-split) task_split="$2"; shift 2 ;;
    --max-concurrency) max_concurrency="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --reference-user-llm) reference_user_llm="$2"; shift 2 ;;
    --reference-range) reference_range="$2"; shift 2 ;;
    --reference-source) reference_source="$2"; shift 2 ;;
    --serve-config) serve_config="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --refresh) refresh=1; shift ;;
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
REPORT=""
for candidate in \
  "$RUNFILES/_main/max/tests/integration/accuracy/model_evals/exacto_report.py" \
  "$SCRIPT_DIR/exacto_report.py"; do
  if [[ -f "$candidate" ]]; then REPORT="$candidate"; break; fi
done
if [[ -z "$REPORT" ]]; then
  echo "ERROR: exacto_report.py not found in runfiles or next to this script" >&2
  exit 1
fi

# An explicitly requested third-party simulator needs its credential, and
# failing here beats dying mid-simulation after burning GPU time. Self-
# simulation (the default) needs nothing beyond the endpoint itself.
resolve_user_key() {
  case "$user_llm" in
    openrouter/*)
      if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        cat >&2 <<'MSG'
ERROR: --user-llm asks for a simulator through OpenRouter but
       OPENROUTER_API_KEY is not set.

       Drop --user-llm to self-simulate with the endpoint under test instead.
       That needs no credential, but the score is not comparable to
       OpenRouter's leaderboard.
MSG
        exit 1
      fi
      ;;
    gemini/*)
      if [[ -z "${GEMINI_API_KEY:-}" && -n "${GOOGLE_API_KEY:-}" ]]; then
        export GEMINI_API_KEY="$GOOGLE_API_KEY"
      fi
      if [[ -z "${GEMINI_API_KEY:-}" ]]; then
        cat >&2 <<'MSG'
ERROR: --user-llm asks for a Gemini simulator but neither GEMINI_API_KEY nor
       GOOGLE_API_KEY is set. GOOGLE_API_KEY is referenced by
       minimaxM3AgentClassDatasetEval.yaml but was never provisioned as a repo
       secret, so it is probably not available to you.

       Drop --user-llm to self-simulate with the endpoint under test instead.
       That needs no credential, but the score is not comparable to
       OpenRouter's leaderboard.
MSG
        exit 1
      fi
      ;;
  esac
}

# tau2 raises on an agent turn with neither text nor tool calls (a generation
# that ran to the context cap, for one), and nothing catches it per task, so a
# single such turn aborts the whole run. The OpenRouter harness hands the empty
# text to the user simulator and carries on; this patch makes tau2 do the same.
patch_empty_agent_turn() {
  vpy - "$CHECKOUT/src/tau2/orchestrator/orchestrator.py" <<'PYPATCH'
import sys
path = sys.argv[1]
anchor = "            agent_msg.validate()\n"
replacement = (
    "            if not (agent_msg.has_text_content() or agent_msg.is_tool_call()):\n"
    "                agent_msg.content = \"\"\n"
)
src = open(path).read()
if src.count(anchor) != 1:
    sys.exit(f"ERROR: tau2 changed; cannot patch the empty-agent-turn check in {path}")
open(path, "w").write(src.replace(anchor, replacement))
PYPATCH
}

ensure_checkout() {
  local uv; uv="$(find_uv)"
  if [[ "$refresh" == 1 ]]; then rm -rf "$CHECKOUT"; fi
  if [[ ! -d "$CHECKOUT/.git" ]]; then
    echo "[exacto] cloning tau2-bench-verified into $CHECKOUT"
    mkdir -p "$CACHE"
    git clone -q "$HARNESS_REPO" "$CHECKOUT"
  fi
  git -C "$CHECKOUT" fetch -q origin "$PIN" 2>/dev/null || git -C "$CHECKOUT" fetch -q origin
  # -f discards the patch below from a previous run, so it applies to a clean
  # tree even when the pin moves.
  git -C "$CHECKOUT" checkout -q -f "$PIN"
  if [[ ! -x "$VENV/bin/tau2" ]]; then
    echo "[exacto] provisioning harness venv in $VENV"
    # Python 3.12, not the system interpreter: tau2 declares
    # `litellm>=1.65.0` unbounded, and current litellm imports
    # `typing.NotRequired`, which does not exist before 3.11 — so a 3.10 venv
    # dies with `ImportError: cannot import name 'NotRequired'` the moment the
    # CLI loads. uv fetches the interpreter if the host lacks it.
    "$uv" venv --python "$HARNESS_PYTHON" "$VENV" >/dev/null
    (cd "$CHECKOUT" && "$uv" pip install --python "$VENV/bin/python" -q -e .)
  fi
  patch_empty_agent_turn
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
    cat >&2 <<MSG
ERROR: could not auto-detect a model from $url/v1/models.

       The list came back empty, which has two usual causes: the credential
       is missing or rejected (the endpoint 401s and the list reads empty),
       or this is a mach engine behind the Mammoth orchestrator, which never
       populates the list. Either way, pass --model with the served name.
MSG
    exit 1
  fi
  if [[ "$count" -gt 1 ]]; then
    # A multi-model gateway lists everything it fronts, in no useful order.
    # Guessing the first entry once picked an image-generation model and the
    # whole run 404'd, so refuse and make the caller choose.
    {
      echo "ERROR: $url/v1/models lists $count models, so there is nothing to"
      echo "       auto-detect. Pass --model with one of:"
      printf '%s\n' "$ids" | sed 's/^/         /'
    } >&2
    exit 1
  fi
  # Copy the id verbatim: MAX serve matches the OpenAI `model` field by exact
  # string equality, so a reconstructed name 400s every request.
  model="$ids"
  echo "[exacto] model: $model"
}

# tau2 hands the model the airline tool schemas on every turn. An endpoint that
# rejects the `tools` field, or an architecture with no tool-call parser wired
# up, scores near zero for reasons that have nothing to do with accuracy. Check
# that before spending 50 simulations to find out.
preflight_tools() {
  local body resp code payload
  body="$(vpy -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [
        {"role": "user", "content": "What is the weather in Paris? Call the tool."}
    ],
    "tools": [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }}],
    "max_tokens": 128,
}))
' "$model")"
  resp="$(curl -s -m 180 -w '\n%{http_code}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${api_key:-dummy}" \
    -d "$body" "$url/v1/chat/completions" || true)"
  code="${resp##*$'\n'}"
  payload="${resp%$'\n'*}"
  if [[ "$code" != "200" ]]; then
    cat >&2 <<MSG
ERROR: the endpoint rejected a tool-calling request (HTTP ${code:-no response}).

       tau2 is a tool-calling benchmark, so it would score near zero here for
       infrastructure reasons rather than accuracy ones.
       Response: $(printf '%s' "$payload" | tr -d '\n' | head -c 400)
MSG
    exit 1
  fi
  vpy - "$payload" <<'PYCHECK'
import json, sys

try:
    choices = json.loads(sys.argv[1]).get("choices") or []
except json.JSONDecodeError:
    sys.exit("ERROR: endpoint returned non-JSON for a tool-calling request")
message = (choices[0].get("message") if choices else {}) or {}
if message.get("tool_calls"):
    print("[exacto] tool-calling preflight: endpoint returned a tool_call")
else:
    print(
        "::warning::the endpoint accepted `tools` but returned no tool_call for "
        "a prompt that asks for one. If tau2 scores near zero, check that this "
        "architecture has a tool-call parser wired up before reading the score "
        "as an accuracy result."
    )
PYCHECK
}

# A score is only evidence if you can say what produced it. Quantization, KV
# cache dtype and speculative decoding all shift accuracy, and none of them
# appear in the harness log, so probe the endpoint for whatever it volunteers
# and let the caller state the rest with --serve-config.
capture_endpoint_meta() {
  local out="$1"
  local models_json model_json
  models_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models" 2>/dev/null || echo '')"
  model_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models/$model" 2>/dev/null || echo '')"
  vpy - "$out" "$url" "$model" "${serve_config:-}" \
    "$models_json" "$model_json" <<'PYMETA'
import json, os, subprocess, sys

out, url, model, serve_config, models_json, model_json = sys.argv[1:7]


def maybe_json(raw):
    try:
        return json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        return None


def git_head():
    # The client's commit, not the server's. MAX serve exposes no build id, so
    # a score against someone else's endpoint cannot record what built it.
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
    # Free-text or JSON the caller supplies: the recipe, quantization, KV dtype,
    # spec-decode mode, whatever identifies the deployment under test.
    "serve_config": maybe_json(serve_config) or (serve_config or None),
    "client_commit": git_head(),
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
PYMETA
}

resolve_user_key
ensure_checkout
detect_model
preflight_tools

mkdir -p "$out_dir"
RUN_NAME="exacto-airline-$$"
RESULTS_JSON="$CHECKOUT/data/simulations/$RUN_NAME.json"
rm -f "$RESULTS_JSON"

# litellm reaches our endpoint through the openai provider, which takes the base
# URL as an api_base kwarg rather than a flag.
# "unset" leaves reasoning_effort out, so the chat template's default applies.
#
# The OpenRouter harness sets no request timeout and retries a retryable error
# up to 6 times. Left alone, litellm gives up after 600 s and tau2 resends up to
# 3 times; a request still generating (a repetition loop runs to the context
# cap) is resent at temperature 0, loops again, and the server keeps decoding
# the abandoned copy, since it does not cancel on disconnect. litellm treats
# "no timeout" as 600 s, so a day stands in for none.
llm_args() {
  vpy -c '
import json, sys
args = {"temperature": 0.0, "timeout": 86400, "num_retries": 6}
if sys.argv[1] != "unset":
    args["reasoning_effort"] = sys.argv[1]
if sys.argv[2]:
    args["api_base"] = sys.argv[2]
print(json.dumps(args))
' "$1" "$2"
}
agent_args="$(llm_args "$reasoning_effort" "$url/v1")"

# Default: the endpoint under test plays the customer too, so the simulator
# needs the same api_base. A third-party simulator resolves through its own
# provider credential and must NOT be handed our base URL.
if [[ -z "$user_llm" ]]; then
  user_llm="openai/$model"
  user_reasoning_effort="${user_reasoning_effort:-$reasoning_effort}"
  user_args="$(llm_args "$user_reasoning_effort" "$url/v1")"
  self_simulated=1
else
  user_reasoning_effort="${user_reasoning_effort:-medium}"
  user_args="$(llm_args "$user_reasoning_effort" "")"
  self_simulated=0
fi

args=(
  run
  --domain airline
  --agent-llm "openai/$model"
  --agent-llm-args "$agent_args"
  --user-llm "$user_llm"
  --user-llm-args "$user_args"
  --num-trials "$num_trials"
  --task-split-name "$task_split"
  --max-steps "$max_steps"
  --max-concurrency "$max_concurrency"
  --save-to "$RUN_NAME"
)
[[ -n "$num_tasks" ]] && args+=(--num-tasks "$num_tasks")

echo "[exacto] running tau2 airline: trials=$num_trials tasks=${num_tasks:-50(all)} reasoning_effort=$reasoning_effort user=$user_llm$([[ "$self_simulated" == 1 ]] && echo " (self-simulated)") user_reasoning_effort=$user_reasoning_effort"
(
  cd "$CHECKOUT"
  # Pin the data directory explicitly. tau2 otherwise derives it from the
  # installed module's __file__, which is not this checkout when the venv was
  # built elsewhere — the results then land next to that other tree and this
  # script reports "produced no results" on an otherwise successful run.
  env -u PYTHONPATH -u PYTHONHOME \
    TAU2_DATA_DIR="$CHECKOUT/data" \
    OPENAI_API_KEY="${api_key:-dummy}" \
    "$VENV/bin/tau2" "${args[@]}"
)

if [[ ! -f "$RESULTS_JSON" ]]; then
  echo "ERROR: tau2 produced no results at $RESULTS_JSON" >&2
  exit 1
fi
cp "$RESULTS_JSON" "$out_dir/tau2-results.json"
echo "[exacto] harness results: $out_dir/tau2-results.json"

ENDPOINT_META="$out_dir/endpoint.json"
capture_endpoint_meta "$ENDPOINT_META"

REPORT_ARGS=(--reference-user-llm "$reference_user_llm")

# Count the chosen split from the pinned checkout rather than hardcoding the
# dataset size anywhere: that is what tells a subset run from a full one.
FULL_TASK_COUNT="$(vpy -c '
import json, sys
path, split = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        splits = json.load(f)
except (OSError, json.JSONDecodeError):
    sys.exit(0)
tasks = splits.get(split) if isinstance(splits, dict) else None
if isinstance(tasks, list):
    print(len(tasks))
' "$CHECKOUT/data/tau2/domains/airline/split_tasks.json" "$task_split" 2>/dev/null || true)"
[[ -n "$FULL_TASK_COUNT" ]] && REPORT_ARGS+=(--full-task-count "$FULL_TASK_COUNT")
[[ -n "$reference_range" ]] && REPORT_ARGS+=(--reference-range "$reference_range")
[[ -n "$reference_source" ]] && REPORT_ARGS+=(--reference-source "$reference_source")

vpy "$REPORT" \
  --harness tau2 \
  --log "$out_dir/tau2-results.json" \
  --dataset taubench_airline \
  --out-dir "$out_dir" \
  --endpoint-meta "$ENDPOINT_META" \
  "${REPORT_ARGS[@]}" \
  --metric-prefix EXACTO_TAUBENCH

# Comparability guard: report, don't fail.
vpy - "$out_dir/score.json" "$num_tasks" "$self_simulated" <<'PYWARN'
import json, sys
score_path, num_tasks, self_simulated = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
s = json.load(open(score_path))


def model_name(llm):
    # gemini/gemini-2.5-flash and openrouter/google/gemini-2.5-flash are the same
    # simulator reached through different litellm providers.
    return str(llm).rsplit("/", 1)[-1]


def effort(args):
    return (args or {}).get("reasoning_effort")


if model_name(s.get("user_llm")) != model_name(s.get("reference_user_llm")):
    print(f"::warning::user simulator is {s.get('user_llm')}, not OpenRouter's "
          f"pinned {s.get('reference_user_llm')}, so this score is a regression signal "
          "for our own runs and NOT our AutoExacto standing. Self-simulation in "
          "particular has one model on both sides of the conversation.")
if effort(s.get("agent_llm_args")) != "high":
    print(f"::warning::agent reasoning_effort is {effort(s.get('agent_llm_args'))}, "
          "not the harness's high, so this score is not comparable to OpenRouter's.")
if not self_simulated and effort(s.get("user_llm_args")) != "medium":
    print(f"::warning::simulator reasoning_effort is "
          f"{effort(s.get('user_llm_args'))}, not the harness's medium.")
if s.get("max_steps") != 200:
    print(f"::warning::max_steps is {s.get('max_steps')}, expected 200.")
graded = s.get("total") or 0
if num_tasks:
    print(f"::warning::PARTIAL run (--num-tasks {num_tasks}): {graded} graded "
          "simulations has a different denominator than the published score.")
elif graded < 45:
    print(f"::warning::only {graded} graded simulations; OpenRouter requires a "
          "minimum of 45 per model-provider pair.")
PYWARN

echo "[exacto] results: $out_dir/score.json"
