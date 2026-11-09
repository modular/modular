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
# Install what the DeepSWE eval needs: the pinned deep-swe tasks, Pier in its
# own venv, a `docker compose` for Pier, and two Pier patches. Idempotent;
# everything lives under DEEPSWE_HOME. run_deepswe_local.sh runs it before
# every run; it also works on its own.
#
# Env (all optional):
#   DEEPSWE_HOME         install root (default /tmp/deepswe)
#   DEEPSWE_REF          deep-swe commit (default: v1.1, 113 tasks, needs Pier > 0.3.0)
#   PIER_VERSION         datacurve-pier version (default 0.3.1)
#   COMPOSE_VERSION      docker compose v2 release installed when `docker compose` is missing
#   DEEPSWE_MODEL_PORTS  ports Pier's egress proxy admits for the model endpoint besides
#                        80 and 443 (default "8000 9000"; run_deepswe_local.sh checks against the same list)
#   DOCKER_HOST          container engine socket (default unix:///tmp/podman.sock)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEEPSWE_HOME=${DEEPSWE_HOME:-/tmp/deepswe}
DEEPSWE_REF=${DEEPSWE_REF:-0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea}
PIER_VERSION=${PIER_VERSION:-0.3.1}
COMPOSE_VERSION=${COMPOSE_VERSION:-v2.29.7}
DEEPSWE_MODEL_PORTS=${DEEPSWE_MODEL_PORTS:-"8000 9000"}
export DOCKER_HOST=${DOCKER_HOST:-unix:///tmp/podman.sock}

mkdir -p "$DEEPSWE_HOME/bin"
cd "$DEEPSWE_HOME"

# uv needs a writable cache; fall back to a local one when ~/.cache/uv is not ours.
uv_cache=${UV_CACHE_DIR:-$HOME/.cache/uv}
if [ -e "$uv_cache" ] && [ ! -w "$uv_cache" ]; then
  export UV_CACHE_DIR="$DEEPSWE_HOME/.uv-cache"
fi

# 1. Tasks.
[ -d deep-swe/.git ] || git clone --quiet https://github.com/datacurve-ai/deep-swe.git deep-swe
git -C deep-swe checkout --quiet "$DEEPSWE_REF"

# 2. Pier, in its own venv.
if ! [ -x pier-venv/bin/pier ] || [ "$(pier-venv/bin/python -c 'import importlib.metadata as m; print(m.version("datacurve-pier"))')" != "$PIER_VERSION" ]; then
  uv venv --python 3.12 pier-venv
  uv pip install --python pier-venv/bin/python "datacurve-pier==$PIER_VERSION"
fi

# 3. `docker compose`. Pier shells out to it. A podman-backed `docker` (the CI
# runner image and dev boxes) may lack the `compose` subcommand, so install
# the standalone compose v2 binary and a `docker` shim that routes `compose`
# to it and everything else to the original docker. Compose talks to the
# engine through DOCKER_HOST.
if ! docker compose version >/dev/null 2>&1; then
  if ! [ -x bin/docker-compose ]; then
    curl -fsSL -o bin/docker-compose \
      "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)"
    chmod +x bin/docker-compose
  fi
  orig_docker=$(command -v docker)
  cat > bin/docker.tmp <<SHIM
#!/usr/bin/env bash
if [ "\${1:-}" = compose ]; then shift; exec "$DEEPSWE_HOME/bin/docker-compose" "\$@"; fi
exec "$orig_docker" "\$@"
SHIM
  chmod +x bin/docker.tmp && mv -f bin/docker.tmp bin/docker
fi
PATH="$DEEPSWE_HOME/bin:$PATH" docker compose version

# 4. Pier patches (see deepswe_patch_pier.py): admit the model port in the squid
# egress proxy, and keep containers startable on hosts whose oom_score_adj
# cannot be lowered.
read -r -a ports <<<"$DEEPSWE_MODEL_PORTS"
pier-venv/bin/python -I "$here/deepswe_patch_pier.py" --ports "${ports[@]}"

echo "DeepSWE setup done in $DEEPSWE_HOME"
