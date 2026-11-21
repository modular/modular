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
# TerminalBench agentic-cost gate against any OpenAI-compatible endpoint.
#
# Runs TerminalBench 2.1 through Harbor with the opencode agent (native tool
# calls over streaming /v1/chat/completions), re-runs trials that failed in the
# harness rather than the model, and scores the run with terminalbench_score.py.
# With --baseline it also gates the run against an earlier run's results.jsonl.
#
# The endpoint can be MAX, vLLM, SGLang or any other engine: the runner never
# starts or stops it. It does watch it, because Harbor keeps going when the
# engine dies and records every later trial as a reward-0 failure. If the
# endpoint stops answering, the runner stops Harbor, scores the trials that
# finished, writes INVALID and exits 2: the run is not a measurement.
#
# Why opencode: Terminus-2 sends no native tool calls, so it cannot exercise the
# engine's tool-call parsing; Codex needs /v1/responses streaming. opencode's
# openai provider also calls /v1/responses, so the runner declares an
# @ai-sdk/openai-compatible provider instead.
#
# Usage:
#   run_terminalbench_local.sh [options]
#
# Options:
#   --url URL            Endpoint base URL (default: http://localhost:8000).
#                         A localhost URL is rewritten to the host's address so
#                         the task containers can reach it.
#   --model MODEL        Model name for the API. Default: GET {url}/v1/models.
#   -k N                 Valid trials per task (default: 2). 2 is the release
#                         candidate setting; use 3 or more for a baseline.
#   -n N                 Concurrent trials (default: 24).
#   --output-cap N       Sent as max_tokens on every request (default: 32768).
#                         A response that reaches it counts as capped.
#   --context N          Context limit given to opencode (default: 1048576).
#   --reasoning-effort E Sent as reasoning_effort on every request (default:
#                         unset, the chat template's default).
#   --temperature T      Sent as temperature on every request (default: unset,
#                         the endpoint's default).
#   --top-p P            Sent as top_p on every request (default: unset).
#   --thinking on|off    Sent as chat_template_kwargs.thinking on every
#                         request (default: unset, the chat template's
#                         default).
#   --exclude TASK       Exclude a task (repeatable). Default: qemu-alpine-ssh
#                         and qemu-startup, which need nested virtualization
#                         and never produce a valid trial on our hosts. Passing
#                         --exclude replaces the default list.
#   --task TASK          Run only this task (repeatable), for a smoke run. A
#                         subset is not a measurement; never gate on one.
#   --runtime R          podman (default) or docker.
#   --max-fill-passes N  Re-run passes for harness failures (default: 8).
#   --agent-timeout-multiplier X
#                         Scales each task's agent timeout (Harbor option). A
#                         trial that times out is still valid.
#   --agent-setup-timeout-multiplier X
#                         Scales each task's agent setup timeout, the time to
#                         install opencode in the task container (default: 3).
#                         Concurrent trials slow the installs, and a timeout
#                         is a harness failure.
#   --build-timeout-multiplier X
#                         Scales each task's environment build timeout
#                         (default: 3). Harbor serializes identical builds, so
#                         a trial's wait for the build lock counts against its
#                         build timeout; a timeout is a harness failure.
#   --endpoint-grace S   Seconds the endpoint may fail health checks before the
#                         runner stops Harbor (default: 60). A busy engine can
#                         answer /v1/models slowly under long-context load.
#   --serve-config S     Describes the deployment (image SHA, recipe, flags).
#                         Recorded verbatim in score.json.
#   --baseline FILE      results.jsonl of the run to gate against. The exit
#                         status is then 1 when the gate fails.
#   --out-dir DIR        Results directory (default:
#                         /tmp/terminalbench-results/<UTC timestamp>).
#   -h, --help           Show this help.
#
# Env overrides: TB_CACHE (venv + dataset, default ~/.cache/terminalbench-gate),
#   TB_DATASET_REPO, TB_DATASET_COMMIT, TB_HARBOR_VERSION,
#   TB_PODMAN_COMPOSE_VERSION, TB_OPENCODE_VERSION, TB_API_KEY (default
#   "dummy").
#
# Exit status: 0 scored (and gate passed), 1 gate failed, 2 run invalid.
#
# Example, a release candidate against the last approved release:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:terminalbench_local -- \
#     --serve-config "max a12e237 prod-1 recipe" \
#     --baseline /path/to/approved/results.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '15,/^set -euo/{/^set -euo/!p}' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

url="http://localhost:8000"
model=""
k=2
n=24
output_cap=32768
context=1048576
reasoning_effort=""
temperature=""
top_p=""
thinking=""
excludes=()
includes=()
runtime="podman"
max_fill=8
agent_timeout_mult=""
build_timeout_mult=3
agent_setup_timeout_mult=3
endpoint_grace=60
serve_config=""
baseline=""
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    -k) k="$2"; shift 2 ;;
    -n) n="$2"; shift 2 ;;
    --output-cap) output_cap="$2"; shift 2 ;;
    --context) context="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --top-p) top_p="$2"; shift 2 ;;
    --thinking) thinking="$2"; shift 2 ;;
    --exclude) excludes+=("$2"); shift 2 ;;
    --task) includes+=("$2"); shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    --max-fill-passes) max_fill="$2"; shift 2 ;;
    --agent-timeout-multiplier) agent_timeout_mult="$2"; shift 2 ;;
    --build-timeout-multiplier) build_timeout_mult="$2"; shift 2 ;;
    --agent-setup-timeout-multiplier) agent_setup_timeout_mult="$2"; shift 2 ;;
    --endpoint-grace) endpoint_grace="$2"; shift 2 ;;
    --serve-config) serve_config="$2"; shift 2 ;;
    --baseline) baseline="$(realpath "$2")"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ ${#excludes[@]} -eq 0 ]] && excludes=(qemu-alpine-ssh qemu-startup)
[[ -z "$thinking" || "$thinking" == on || "$thinking" == off ]] \
  || { echo "ERROR: --thinking takes on or off, not $thinking" >&2; exit 2; }
# Fail before the hours of trials, not after them when compare cannot read it.
[[ -z "$baseline" || -f "$baseline" ]] \
  || { echo "ERROR: --baseline file not found: $baseline" >&2; exit 2; }

CACHE="${TB_CACHE:-$HOME/.cache/terminalbench-gate}"
DATASET_REPO="${TB_DATASET_REPO:-https://github.com/harbor-framework/terminal-bench-2-1}"
DATASET_COMMIT="${TB_DATASET_COMMIT:-7131e4375048a0e408a8fb404b5f499d726b695b}"
HARBOR_VERSION="${TB_HARBOR_VERSION:-0.25.0}"
PODMAN_COMPOSE_VERSION="${TB_PODMAN_COMPOSE_VERSION:-1.6.0}"
OPENCODE_VERSION="${TB_OPENCODE_VERSION:-1.19.5}"
API_KEY="${TB_API_KEY:-dummy}"
out_dir="${out_dir:-/tmp/terminalbench-results/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out_dir" "$CACHE"
out_dir="$(realpath "$out_dir")"
log() { echo "[terminalbench] $*" >&2; }

RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
SCORER=""
for cand in \
  "$RUNFILES/_main/max/tests/integration/accuracy/model_evals/terminalbench_score.py" \
  "$SCRIPT_DIR/terminalbench_score.py"; do
  [[ -f "$cand" ]] && { SCORER="$cand"; break; }
done
[[ -n "$SCORER" ]] || { echo "ERROR: terminalbench_score.py not found" >&2; exit 2; }

# Concurrent runs share $CACHE; the lock serializes venv and dataset setup.
exec 9> "$CACHE/.setup.lock"
flock 9

# Harness venv, pinned. podman-compose lives in the same venv because Harbor
# shells out to it for multi-container tasks.
VENV="$CACHE/harbor-$HARBOR_VERSION-pc$PODMAN_COMPOSE_VERSION"
if [[ ! -f "$VENV/.tb-gate-ready" ]]; then
  log "creating harness venv $VENV"
  rm -rf "$VENV"
  uv venv -q --python 3.12 "$VENV"
  uv pip install -q --python "$VENV/bin/python" "harbor==$HARBOR_VERSION" \
    "podman-compose==$PODMAN_COMPOSE_VERSION"
  touch "$VENV/.tb-gate-ready"
fi

DATASET="$CACHE/terminal-bench-$DATASET_COMMIT"
if [[ ! -d "$DATASET/tasks" ]]; then
  log "fetching dataset $DATASET_REPO@$DATASET_COMMIT"
  rm -rf "$DATASET.tmp"
  git init -q "$DATASET.tmp"
  git -C "$DATASET.tmp" fetch -q --depth 1 "$DATASET_REPO" "$DATASET_COMMIT"
  git -C "$DATASET.tmp" checkout -q FETCH_HEAD
  mv "$DATASET.tmp" "$DATASET"
fi
exec 9>&-
TASKS="$DATASET/tasks"

# Harbor's task images use short names (org/image). podman refuses those
# without a TTY unless a registries.conf allows docker.io; the shim sets one
# for Harbor's podman calls only, leaving the system config alone.
# Rootful podman (via sudo) also writes `podman cp` output as root, mode 0600
# for files a task keeps private, and Harbor then cannot read them to copy
# them into the verifier container: the trial ends without a reward. The shim
# hands copied files back to the invoking user.
SHIM="$CACHE/shim"
if [[ "$runtime" == "podman" ]]; then
  real_podman="$(command -v podman.real || true)"
  [[ -z "$real_podman" && -x /usr/bin/podman.real ]] && real_podman=/usr/bin/podman.real
  real_podman="${real_podman:-$(command -v podman)}"
  mkdir -p "$SHIM"
  # Temp file plus mv: a concurrent run may be executing the shim right now.
  tmp="$(mktemp "$SHIM/.registries.XXXXXX")"
  printf 'unqualified-search-registries = ["docker.io"]\nshort-name-mode = "permissive"\n' > "$tmp"
  mv "$tmp" "$SHIM/registries.conf"
  sudo_prefix=""
  [[ "$real_podman" == *podman.real ]] && sudo_prefix="sudo "
  tmp="$(mktemp "$SHIM/.podman.XXXXXX")"
  cat > "$tmp" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" != cp || -z "$sudo_prefix" ]]; then
  exec ${sudo_prefix}env CONTAINERS_REGISTRIES_CONF=$SHIM/registries.conf $real_podman "\$@"
fi
${sudo_prefix}env CONTAINERS_REGISTRIES_CONF=$SHIM/registries.conf $real_podman "\$@"
rc=\$?
dest="\${@: -1}"
if [[ \$rc -eq 0 && "\$dest" != *:* && -e "\$dest" ]]; then
  sudo chown -R "\$(id -u):\$(id -g)" "\$dest" && chmod -R u+rwX "\$dest"
fi
exit \$rc
EOF
  chmod +x "$tmp"
  mv "$tmp" "$SHIM/podman"
  ln -s podman "$SHIM/.docker.$$"
  mv -T "$SHIM/.docker.$$" "$SHIM/docker"
  export PATH="$SHIM:$PATH"
fi
export PATH="$VENV/bin:$PATH"

cleanup() {
  # Harbor killed mid-trial leaves its compose networks behind, and podman's
  # default pool holds only ~250 subnets. `network rm` refuses networks that
  # still have containers, but compose creates a network before it attaches
  # them, so only networks older than any live trial of a concurrent run go.
  if [[ "$runtime" == "podman" ]]; then
    local cutoff net created
    cutoff=$(date -d '-3 hours' +%s)
    for net in $(podman network ls --format '{{.Name}}' 2>/dev/null | grep -E '__(env|verifier)'); do
      # A non-numeric age means unknown, so keep the network.
      created=$(podman network inspect "$net" --format '{{.Created.Unix}}' 2>/dev/null || true)
      [[ "$created" =~ ^[0-9]+$ ]] || continue
      [[ "$created" -lt "$cutoff" ]] || continue
      # Fails while a concurrent run still uses the network; that is expected.
      podman network rm "$net" >/dev/null 2>&1 || true
    done
  fi
  return 0
}
# Keep the script's own exit status; cleanup must never override it.
# Harbor runs in its own session (see watched), so a signal to this script
# does not reach it; without this, a cancelled run keeps its trials, task
# containers and endpoint requests going.
harbor_pid=""
stop_harbor() {
  [[ -n "$harbor_pid" ]] && kill -0 "$harbor_pid" 2>/dev/null || return 0
  log "stopping Harbor (pid $harbor_pid)"
  kill -- "-$harbor_pid" 2>/dev/null || true
  wait "$harbor_pid" 2>/dev/null || true
  harbor_pid=""
}
on_exit() {
  local rc=$?
  stop_harbor
  cleanup
  exit "$rc"
}
trap on_exit EXIT
# Exiting is what runs on_exit.
trap 'exit 143' TERM INT HUP
cleanup

base="${url%/}"; base="${base%/v1}"
health() { curl -sf -m 10 -H "Authorization: Bearer $API_KEY" "$base/v1/models" -o /dev/null; }
health || { echo "ERROR: $base/v1/models does not answer" >&2; exit 2; }
if [[ -z "$model" ]]; then
  model="$(curl -sf -H "Authorization: Bearer $API_KEY" "$base/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
fi

# Task containers cannot reach the host's loopback. Use the address of the
# host's default route instead; derive it per run because it changes when a
# workspace restarts.
agent_base="$base"
if [[ "$base" =~ ^(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?$ ]]; then
  host_ip="$(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
  agent_base="${BASH_REMATCH[1]}$host_ip${BASH_REMATCH[3]}"
fi

# The config names the key by environment variable, so neither Harbor's
# command line nor the saved config holds it; --ae below has Harbor copy the
# variable into the task container.
export TB_ENDPOINT_API_KEY="$API_KEY"
# opencode 1.19.5 sends max_tokens 32000 whatever limit.output says, and only
# sends temperature for models it knows support it. The provider passes
# unknown model options into the request body as given, so every sampling
# setting goes there.
OC_CONFIG="$(python3 - "$agent_base/v1" "$model" "$context" "$output_cap" \
  "$reasoning_effort" "$temperature" "$top_p" "$thinking" <<'PY'
import json, sys
url, model, context, output, effort, temperature, top_p, thinking = sys.argv[1:]
options = {"max_tokens": int(output)}
if effort:
    options["reasoningEffort"] = effort
if temperature:
    options["temperature"] = float(temperature)
if top_p:
    options["top_p"] = float(top_p)
if thinking:
    options["chat_template_kwargs"] = {"thinking": thinking == "on"}
entry = {"name": model, "tool_call": True,
         "limit": {"context": int(context), "output": int(output)},
         "options": options}
print(json.dumps({"provider": {"gate": {
    "npm": "@ai-sdk/openai-compatible", "name": "gate",
    "options": {"baseURL": url, "apiKey": "{env:TB_ENDPOINT_API_KEY}"},
    "models": {model: entry}}}}))
PY
)"
echo "$OC_CONFIG" > "$out_dir/opencode_config.json"

JOBS="$out_dir/jobs"
mkdir -p "$JOBS"
exclude_args=()
for t in "${excludes[@]}"; do exclude_args+=(-x "$t"); done
task_args=()
for t in "${includes[@]}"; do task_args+=(-i "$t"); done
# An array, not a function: setsid can only execute a program. The --ae value
# is a template that Harbor resolves from this environment, so it records only
# the variable's name.
# shellcheck disable=SC2016 # Harbor expands the template, not the shell.
harbor_cmd=(harbor run -p "$TASKS" -e "$runtime" -a opencode -m "gate/$model"
  --ak "opencode_config=$OC_CONFIG" --ak "version=$OPENCODE_VERSION"
  --ae 'TB_ENDPOINT_API_KEY=${TB_ENDPOINT_API_KEY}'
  -o "$JOBS" --yes)
[[ -n "$agent_timeout_mult" ]] && harbor_cmd+=(--agent-timeout-multiplier "$agent_timeout_mult")
harbor_cmd+=(--environment-build-timeout-multiplier "$build_timeout_mult")
harbor_cmd+=(--agent-setup-timeout-multiplier "$agent_setup_timeout_mult")

# Runs Harbor in its own process group and stops it once the endpoint has
# failed health checks for --endpoint-grace seconds; shorter outages are tolerated. Returns 2 when the endpoint went down.
watched() {
  local log_file="$1"; shift
  setsid "$@" > "$log_file" 2>&1 &
  local pid=$! misses=0
  harbor_pid=$pid
  while kill -0 "$pid" 2>/dev/null; do
    if health; then misses=0; else misses=$((misses + 1)); fi
    if [[ $((misses * 15)) -ge $endpoint_grace ]]; then
      log "endpoint down; stopping Harbor (pid $pid)"
      kill -- "-$pid" 2>/dev/null || true
      date -u +%FT%TZ > "$out_dir/ENDPOINT_DOWN"
      wait "$pid" 2>/dev/null || true
      harbor_pid=""
      return 2
    fi
    # In the background, so a signal interrupts the wait instead of waiting
    # out the sleep.
    sleep 15 & wait $!
  done
  wait "$pid" || log "harbor exited non-zero (see $log_file); scoring decides validity"
  harbor_pid=""
  return 0
}

# Writes meta.json and scores every trial that finished, as results.jsonl and
# score.json.
finalize() {
  python3 - "$out_dir/meta.json" "$base" "$model" "$serve_config" \
    "$DATASET_REPO@$DATASET_COMMIT" "$HARBOR_VERSION" "opencode@$OPENCODE_VERSION" \
    "$k" "$n" "$output_cap" "$context" "$reasoning_effort" "$temperature" "$top_p" \
    "$thinking" "$(IFS=,; echo "${includes[*]}")" "$pass" "$TASKS" "${excludes[@]}" <<'PY' || return 1
import json, sys
from pathlib import Path
(out, endpoint, model, serve_config, dataset, harbor, agent, k, n, cap, context,
 effort, temperature, top_p, thinking, subset, fill_passes, tasks_dir,
 *excluded) = sys.argv[1:]
# The tasks the run set out to score, so a report can count one that never
# produced a valid trial. Every task in the dataset is recorded too, so an
# uploaded row's task index is the same in full and subset runs.
dataset_tasks = sorted(p.parent.name for p in Path(tasks_dir).glob("*/task.toml"))
planned = set(dataset_tasks)
planned -= set(excluded)
if subset:
    planned &= set(subset.split(","))
json.dump({
    "endpoint": endpoint, "model": model, "serve_config": serve_config,
    "dataset": dataset, "harbor": harbor, "agent": agent, "k": int(k),
    "n_concurrent": int(n), "output_cap": int(cap), "context": int(context),
    "reasoning_effort": effort or None,
    "temperature": float(temperature) if temperature else None,
    "top_p": float(top_p) if top_p else None,
    "thinking": thinking or None, "excluded_tasks": excluded,
    "task_subset": subset.split(",") if subset else None,
    "planned_tasks": sorted(planned),
    "dataset_tasks": dataset_tasks,
    "fill_passes": int(fill_passes),
}, open(out, "w"), indent=2)
PY
  python3 "$SCORER" score --jobs-dir "$JOBS" --out-dir "$out_dir" -k "$k" \
    --output-cap "$output_cap" --meta "$out_dir/meta.json" || return 1
  log "results: $out_dir/score.json"
}

# The trials that finished before the endpoint went down are still scored, so
# they reach the report; INVALID marks the run as not a measurement.
endpoint_lost() {
  finalize || log "scoring the finished trials failed"
  echo "reason=endpoint_down" > "$out_dir/INVALID"
  exit 2
}

log "model=$model endpoint=$agent_base k=$k n=$n out=$out_dir"
pass=0
watched "$out_dir/harbor_main.log" "${harbor_cmd[@]}" -k "$k" -n "$n" "${exclude_args[@]}" "${task_args[@]}" --job-name main || endpoint_lost

while [[ $pass -lt $max_fill ]]; do
  short="$out_dir/missing_$((pass + 1)).txt"
  # Excluded tasks are never run, so drop them from the shortfall.
  python3 "$SCORER" missing --jobs-dir "$JOBS" --tasks-dir "$TASKS" -k "$k" \
    | awk 'NR == FNR {ex[$1] = 1; next} !($1 in ex)' <(printf '%s\n' "${excludes[@]}") - \
    > "$short"
  if [[ ${#includes[@]} -gt 0 ]]; then
    awk 'NR == FNR {in_set[$1] = 1; next} $1 in in_set' <(printf '%s\n' "${includes[@]}") \
      "$short" > "$short.tmp" && mv "$short.tmp" "$short"
  fi
  [[ -s "$short" ]] || break
  pass=$((pass + 1))
  log "fill pass $pass: $(wc -l < "$short") tasks short of k"
  # One Harbor job per distinct shortfall, so each task runs exactly what it needs.
  mapfile -t needs < <(awk '{print $2}' "$short" | sort -un)
  for need in "${needs[@]}"; do
    mapfile -t names < <(awk -v m="$need" '$2 == m {print $1}' "$short")
    include_args=()
    for t in "${names[@]}"; do include_args+=(-i "$t"); done
    watched "$out_dir/harbor_fill${pass}_k$need.log" \
      "${harbor_cmd[@]}" -k "$need" -n "$n" "${include_args[@]}" --job-name "fill${pass}-k$need" || endpoint_lost
  done
done

finalize

# A run with no valid trial, or with tasks still short of k, is not a measurement.
read -r scored _ < <(python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
print(s["total"], s["tasks_short_of_k"])' "$out_dir/score.json")
remaining=$(python3 "$SCORER" missing --jobs-dir "$JOBS" --tasks-dir "$TASKS" -k "$k" \
  | awk 'NR == FNR {ex[$1] = 1; next} !($1 in ex)' <(printf '%s\n' "${excludes[@]}") - \
  | awk 'NR == FNR {in_set[$1] = 1; next} n == 0 || $1 in in_set' n="${#includes[@]}" \
    <(printf '%s\n' "${includes[@]}") - | wc -l)
if [[ "$scored" -eq 0 || "$remaining" -gt 0 ]]; then
  log "INVALID: $scored valid trials; $remaining tasks still short of k=$k after $pass fill passes"
  echo "valid_trials=$scored tasks_short_of_k=$remaining" > "$out_dir/INVALID"
  exit 2
fi

if [[ -n "$baseline" ]]; then
  # A task this run excluded may still be in the baseline; compare gives no
  # verdict for a task in one arm only unless it is named.
  compare_excludes=()
  for t in "${excludes[@]}"; do compare_excludes+=(--exclude "$t"); done
  python3 "$SCORER" compare --baseline "$baseline" --candidate "$out_dir/results.jsonl" \
    "${compare_excludes[@]}" --out "$out_dir/verdict.json" | tee "$out_dir/verdict.md"
  exit "${PIPESTATUS[0]}"
fi
