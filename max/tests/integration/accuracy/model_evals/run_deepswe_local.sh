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
# Run DeepSWE (113 long-horizon coding tasks; mini-swe-agent driven by Pier,
# task containers under podman/docker) against an OpenAI-compatible MAX Serve
# endpoint, then score it the way the public leaderboard does
# (deepswe_score.py). Provisions Pier and the tasks first through
# deepswe_setup.sh, which is idempotent. Extra arguments go to `pier run`,
# e.g. `--n-tasks 5 --sample-seed 0`.
#   MODEL=MiniMaxAI/MiniMax-M3.1-Flash PORT=8000 DEEPSWE_HOME=~/deepswe \
#     ./bazelw run //max/tests/integration/accuracy/model_evals:deepswe_eval_local
# See DEEPSWE.md next to this script.
#
# Env:
#   MODEL                served model name (required), sent as openai/<MODEL>
#   MAX_SERVE_URL        server URL reachable from inside containers; empty means a
#                        local server on PORT, reached as host.containers.internal
#                        (Podman only; with Docker, set this to the host's IP)
#   PORT                 local server port (default 8000)
#   OPENAI_API_KEY       bearer token Pier requires for openai/ models (default dummy);
#                        scrubbed from everything under OUT when the script exits
#   DEEPSWE_HOME         install root, see deepswe_setup.sh (default /tmp/deepswe)
#   DEEPSWE_SKIP_SETUP   set to 1 to skip deepswe_setup.sh (it is cheap once done)
#   DEEPSWE_ATTEMPTS     attempts per task, pier -k (default 1; the leaderboard
#                        averages pass@1 over 4)
#   DEEPSWE_CONCURRENCY  tasks at once, pier -n (default 4; each task holds a 2-CPU,
#                        8 GB agent container, then a verifier of the same size,
#                        for up to 3 hours)
#   SAMPLE_SIZE          run N sampled tasks (--n-tasks N --sample-seed 0)
#   OUT                  results dir (default $DEEPSWE_HOME/results/<timestamp>);
#                        trials land in $OUT/deepswe, the score in $OUT/score.json
#   DEEPSWE_MODEL_PORTS  ports the egress proxy admits (must match setup.sh)
#   DOCKER_HOST          engine socket (default unix:///tmp/podman.sock, which is
#                        started with `podman system service` if dead; any other
#                        socket must already be answering)
#   DOCKER_BUILDKIT      default 0: compose's BuildKit path boots a buildkit
#                        container, which podman cannot create with cgroups
#                        disabled; the classic builder uses the engine's /build
set -euo pipefail

# Under `bazel run` the siblings sit in runfiles next to this script; run
# directly from the source tree they are next to it as well.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
if [ -d "$RUNFILES/_main/max/tests/integration/accuracy/model_evals" ]; then
  here="$RUNFILES/_main/max/tests/integration/accuracy/model_evals"
fi
: "${MODEL:?set MODEL to the served model name}"
DEEPSWE_HOME=${DEEPSWE_HOME:-/tmp/deepswe}
export DEEPSWE_HOME
if [ "${DEEPSWE_SKIP_SETUP:-0}" != 1 ]; then
  "$here/deepswe_setup.sh"
fi
PORT=${PORT:-8000}
OUT=${OUT:-$DEEPSWE_HOME/results/$(date +%Y%m%d-%H%M%S)}
JOB_NAME=${JOB_NAME:-deepswe}
DEEPSWE_MODEL_PORTS=${DEEPSWE_MODEL_PORTS:-"8000 9000"}
export DEEPSWE_ATTEMPTS=${DEEPSWE_ATTEMPTS:-1}
DEEPSWE_CONCURRENCY=${DEEPSWE_CONCURRENCY:-4}
DEFAULT_PODMAN_SOCK=/tmp/podman.sock
export DOCKER_HOST=${DOCKER_HOST:-unix://$DEFAULT_PODMAN_SOCK}
export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}
export COMPOSE_DOCKER_CLI_BUILD=${COMPOSE_DOCKER_CLI_BUILD:-0}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export PATH="$DEEPSWE_HOME/bin:$PATH"
pier=$DEEPSWE_HOME/pier-venv/bin/pier
[ -x "$pier" ] || { echo "Pier not installed under $DEEPSWE_HOME; deepswe_setup.sh did not run" >&2; exit 1; }

# The agent runs inside a task container, so a server on this host is reached
# through the engine's host alias, never through localhost. Pier's squid proxy
# only forwards the ports setup.sh admitted.
if [ -n "${MAX_SERVE_URL:-}" ]; then
  OPENAI_API_BASE="${MAX_SERVE_URL%/}/v1"
  probe="$OPENAI_API_BASE/models"
else
  OPENAI_API_BASE="http://host.containers.internal:${PORT}/v1"
  probe="http://127.0.0.1:${PORT}/v1/models"
fi
export OPENAI_API_BASE
port=$(python3 -c 'import sys, urllib.parse as u; p = u.urlparse(sys.argv[1]); print(p.port or (443 if p.scheme == "https" else 80))' "$OPENAI_API_BASE")
case " 80 443 $DEEPSWE_MODEL_PORTS " in
  *" $port "*) ;;
  *)
    echo "The egress proxy only forwards ports 80, 443 and $DEEPSWE_MODEL_PORTS; $OPENAI_API_BASE uses $port." >&2
    echo "Rerun with DEEPSWE_MODEL_PORTS including $port so deepswe_setup.sh admits it." >&2
    exit 1
    ;;
esac
probe_status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -H "Authorization: Bearer $OPENAI_API_KEY" "$probe" || true)
if [ "$probe_status" != 200 ]; then
  echo "No model server answering at $probe (HTTP ${probe_status:-000}; 000 means no connection, 401 means the bearer token was rejected)" >&2
  exit 1
fi
echo "DeepSWE model endpoint: $OPENAI_API_BASE (model $MODEL)"

# Pier isolates the agent on an `internal` compose network; netavark creates
# one by writing net.ipv4.conf.<bridge>.forwarding=0, which needs a writable
# /proc/sys. Non-privileged Kubernetes pods mount it read-only. The remount is
# a security-relevant change, so it is left to the operator.
if grep -qE ' /proc/sys proc ro[ ,]' /proc/mounts; then
  echo "/proc/sys is read-only: internal container networks cannot be created here." >&2
  echo "Run once per pod lifetime, then retry:  sudo mount -o remount,bind,rw /proc/sys" >&2
  exit 1
fi

# Engine socket. CI starts it in an earlier step; locally start a rootful
# podman service. The podman wrapper on these hosts runs as root, so the socket
# is root-owned until chmod'ed. Only the script's own default socket is ever
# replaced: a probe also fails on a live daemon the user cannot read, such as
# /var/run/docker.sock outside the docker group, and that one must survive.
sock=${DOCKER_HOST#unix://}
if [ "$sock" != "$DOCKER_HOST" ] && ! curl -sf --unix-socket "$sock" http://localhost/version >/dev/null 2>&1; then
  if [ "$sock" != "$DEFAULT_PODMAN_SOCK" ]; then
    echo "No container engine answers at $DOCKER_HOST, or this user cannot reach it." >&2
    echo "Start that engine or fix its permissions, or unset DOCKER_HOST to let this" >&2
    echo "script run its own podman service at $DEFAULT_PODMAN_SOCK." >&2
    exit 1
  fi
  sudo rm -f "$sock" 2>/dev/null || rm -f "$sock"
  setsid -f podman system service --time=0 "unix://$sock" >/dev/null 2>&1
  for _ in $(seq 60); do
    if sudo test -S "$sock" 2>/dev/null || test -S "$sock"; then break; fi
    sleep 0.5
  done
  sudo chmod 0777 "$sock" 2>/dev/null || true
fi
docker info >/dev/null

# host.containers.internal is Podman's alias for the host; Docker on Linux
# does not define it, so the egress proxy could never reach a local server.
engine_version=$(curl -sf --unix-socket "$sock" http://localhost/version 2>/dev/null || true)
if [ -z "${MAX_SERVE_URL:-}" ] && [ "$sock" != "$DOCKER_HOST" ] \
  && [[ "$engine_version" != *'"Podman Engine"'* ]]; then
  echo "The engine at $DOCKER_HOST is not Podman, so containers cannot reach a local server" >&2
  echo "as host.containers.internal. Set MAX_SERVE_URL to an address they can reach," >&2
  echo "e.g. MAX_SERVE_URL=http://<this host's IP>:$PORT, or use Podman." >&2
  exit 1
fi

mkdir -p "$OUT"

# The key sits in the agent's sandbox, where the model runs arbitrary shell
# commands, so an `env` writes it into trajectories and logs. Scrub it from
# everything under $OUT on any exit, before anyone uploads or shares them, and
# fail the run when any file could not be checked. Files the rootful engine
# left root-owned get a second pass under sudo.
pump_pid=
redact_results() {
  local rc=$?
  if [ -n "$pump_pid" ]; then
    # A TERM sent before the pump has run setsid can be lost, and the wait
    # would then block forever, so signal only once it leads its session.
    for _ in $(seq 50); do
      [ "$(ps -o sid= -p "$pump_pid" 2>/dev/null | tr -d ' ')" != "$pump_pid" ] || break
      sleep 0.1
    done
    kill -- "-$pump_pid" 2>/dev/null || true
    wait "$pump_pid" 2>/dev/null || true
  fi
  if ! python3 -I "$here/deepswe_redact.py" "$OUT"; then
    # As root, a file that still cannot be scrubbed is deleted and listed in
    # $OUT/REMOVED_UNSCRUBBABLE_FILES.txt rather than failing the whole run.
    echo "Retrying the scrub as root for the files above" >&2
    if ! sudo -n --preserve-env=OPENAI_API_KEY python3 -I "$here/deepswe_redact.py" \
      --remove-unscrubbable "$OUT"; then
      echo "ERROR: could not scrub or delete every file under $OUT that may hold the API key; do not share it" >&2
      [ "$rc" -ne 0 ] || rc=1
    fi
  fi
  exit "$rc"
}
trap redact_results EXIT

# Podman schedules container healthchecks through systemd timers. Without
# systemd, Pier's squid proxy would stay "starting" and the agent container,
# which waits on service_healthy, would never start; run the checks ourselves.
# The pump leads its own process group so the scrub can stop it, and the
# `podman logs` it may still be writing, before rewriting its log.
if [ ! -d /run/systemd/system ]; then
  setsid "$here/deepswe_healthcheck_pump.sh" $$ >>"$OUT/healthcheck-pump.log" 2>&1 &
  pump_pid=$!
fi

SAMPLE_ARGS=()
if [ -n "${SAMPLE_SIZE:-}" ]; then
  SAMPLE_ARGS=(--n-tasks "$SAMPLE_SIZE" --sample-seed 0)
fi

# openai/ models default to mini-swe-agent's Responses API client;
# model_class=litellm keeps it on /v1/chat/completions. mini-swe-agent sends a
# bash tool definition, and litellm's tool-call path imports its MCP handler,
# so the agent image also needs the `mcp` package. Pier exits non-zero when
# any trial errors; score.py accounts for that.
"$pier" run \
  -p "$DEEPSWE_HOME/deep-swe/tasks" \
  --agent mini-swe-agent \
  --ak model_class=litellm \
  --ak 'extra_python_packages=["mcp"]' \
  --model "openai/$MODEL" \
  --env docker \
  -k "$DEEPSWE_ATTEMPTS" \
  -n "$DEEPSWE_CONCURRENCY" \
  -o "$OUT" \
  --job-name "$JOB_NAME" \
  -y \
  "${SAMPLE_ARGS[@]}" "$@" || echo "pier exited $? (some trials may have failed)"

# deepswe_score is the py_binary in runfiles under bazel; straight from the
# source tree only the module is there.
if [ -x "$here/deepswe_score" ]; then
  "$here/deepswe_score" "$OUT/$JOB_NAME"
else
  python3 -I "$here/deepswe_score.py" "$OUT/$JOB_NAME"
fi
