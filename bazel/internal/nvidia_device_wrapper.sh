#!/bin/bash
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

# Gives a single-GPU test the NVIDIA GPU with the least memory in use.
#
# BuildBuddy schedules single-GPU tests on executors with more than one GPU,
# and CUDA puts each of them on device 0 unless told otherwise. Several tests
# share one GPU, so this balances them instead of locking a device.
#
# Usage:
#   bazel test --run_under="//bazel/internal:nvidia_device_wrapper --tests-only" ...
#
# A leading --tests-only runs anything but a Bazel test as is, for a
# `--run_under` that `bazel run` also applies.
#
# The wrapper fails open: when it cannot read the GPUs, the test runs as it
# would without it.

set -uo pipefail

if [ "${1:-}" = "--tests-only" ]; then
    shift
    if [ -z "${TEST_TMPDIR:-}" ]; then
        exec "$@"
    fi
fi

# The test rules set MODULAR_TEST_GPU_COUNT for GPU tests. A test without it
# may need every GPU, so it keeps them all. A set CUDA_VISIBLE_DEVICES, even
# an empty one, is the test's own choice.
if [ "${MODULAR_TEST_GPU_COUNT:-}" != "1" ] || [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    exec "$@"
fi

query=(nvidia-smi "--query-gpu=uuid,memory.used" "--format=csv,noheader,nounits")
if command -v timeout >/dev/null 2>&1; then
    query=(timeout 10 "${query[@]}")
fi
gpus=$("${query[@]}" 2>/dev/null) || exec "$@"

# Each running test holds a CUDA context and its memory manager pool, which is
# sized to its gpu-memory request, so memory in use is the load from tests
# weighted by their requests. GPUs near the minimum count as tied and one is
# picked at random, so tests that start together spread out.
choice=$(awk -F',' \
    -v tie="${NVIDIA_DEVICE_WRAPPER_TIE_MIB:-512}" \
    -v seed="$RANDOM" '
    BEGIN { n = 0; m = 0 }
    {
        gsub(/[ \t\r]/, "")
        if ($1 ~ /^GPU-/ && $2 ~ /^[0-9]+$/) {
            uuid[n] = $1
            used[n] = $2 + 0
            n++
        }
    }
    END {
        if (n < 2) exit
        min = used[0]
        for (i = 1; i < n; i++) if (used[i] < min) min = used[i]
        for (i = 0; i < n; i++) if (used[i] <= min + tie) tied[m++] = uuid[i]
        print tied[seed % m]
    }' <<<"$gpus")

if [ -z "$choice" ]; then
    exec "$@"
fi

echo "nvidia_device_wrapper: CUDA_VISIBLE_DEVICES=$choice (memory.used MiB: $(tr '\n' ';' <<<"$gpus"))" >&2
export CUDA_VISIBLE_DEVICES="$choice"
exec "$@"
