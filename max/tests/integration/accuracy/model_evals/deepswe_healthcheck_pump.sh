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
# Run podman container healthchecks by hand while a parent process lives.
# Podman schedules HEALTHCHECKs through systemd timers; without systemd a
# container stays "starting" forever and compose never satisfies
# `depends_on: condition: service_healthy`. Containers without a healthcheck
# make `podman healthcheck run` fail harmlessly.
# usage: healthcheck_pump.sh <pid to watch>
set -u
parent=${1:?pid to watch}
export DOCKER_HOST=${DOCKER_HOST:-unix:///tmp/podman.sock}
# Every line this script prints lands in healthcheck-pump.log, which is the
# only record of what the checks saw once Pier has torn the containers down.
# The first check of each container is logged in full; a container that goes
# unhealthy gets its state and recent logs recorded once.
declare -A seen dumped
while kill -0 "$parent" 2>/dev/null; do
  ps_out=$(podman ps --format '{{.ID}} {{.Status}}' 2>&1) || echo "$(date -u +%FT%TZ) podman ps failed: $ps_out"
  for c in $(printf '%s\n' "$ps_out" | grep -E '\((starting|unhealthy)\)' | cut -d' ' -f1); do
    out=$(podman healthcheck run "$c" 2>&1); rc=$?
    if [ -z "${seen[$c]:-}" ]; then
      seen[$c]=1
      echo "$(date -u +%FT%TZ) first healthcheck $c rc=$rc: $(printf '%s' "$out" | head -c 300)"
    fi
    if [ "$rc" -ne 0 ] && printf '%s\n' "$ps_out" | grep -q "^$c .*(unhealthy)" && [ -z "${dumped[$c]:-}" ]; then
      dumped[$c]=1
      echo "$(date -u +%FT%TZ) unhealthy $c: $(podman inspect "$c" --format '{{.Name}} status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} health={{.State.Health.Status}} streak={{.State.Health.FailingStreak}} last={{with index .State.Health.Log 0}}{{.ExitCode}} {{.Output}}{{end}}' 2>&1 | head -c 600)"
      podman logs --tail 30 "$c" 2>&1 | sed "s/^/    $c| /" | head -40
    fi
  done
  sleep 2
done
