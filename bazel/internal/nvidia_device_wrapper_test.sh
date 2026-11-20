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

# Runs nvidia_device_wrapper against a fake nvidia-smi and checks the
# CUDA_VISIBLE_DEVICES each case hands the test.

set -euo pipefail

WRAPPER="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

FAKE_BIN="$(mktemp -d "${TEST_TMPDIR:-/tmp}/fake_bin.XXXXXX")"
cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/bin/bash
[ "$*" = "--query-gpu=uuid,memory.used --format=csv,noheader,nounits" ] || exit 9
printf '%b' "${FAKE_NVIDIA_SMI_OUTPUT:-}"
exit "${FAKE_NVIDIA_SMI_EXIT:-0}"
EOF

# The command the wrapper runs: prints the CUDA_VISIBLE_DEVICES it sees.
cat >"$FAKE_BIN/show-devices" <<'EOF'
#!/bin/bash
echo "${CUDA_VISIBLE_DEVICES-<unset>}"
EOF

cat >"$FAKE_BIN/echo-args-exit-3" <<'EOF'
#!/bin/bash
echo -n "$*|"
exit 3
EOF
chmod +x "$FAKE_BIN"/*

failures=0

# Prints the CUDA_VISIBLE_DEVICES the wrapped command sees, or <unset>.
# Arguments are `env` assignments that override the defaults.
run_wrapper() {
    env -u CUDA_VISIBLE_DEVICES -u MODULAR_TEST_GPU_COUNT \
        PATH="$FAKE_BIN:$PATH" TEST_TMPDIR="${TEST_TMPDIR:-/tmp}" \
        MODULAR_TEST_GPU_COUNT=1 "$@" \
        "$WRAPPER" --tests-only "$FAKE_BIN/show-devices" 2>/dev/null
}

expect() {
    local name="$1" want="$2" got="$3"
    if [ "$got" = "$want" ]; then
        echo "PASS: $name"
    else
        echo "FAIL: $name: want '$want', got '$got'"
        failures=$((failures + 1))
    fi
}

TWO_GPUS='GPU-aaa, 4000\nGPU-bbb, 1000\n'

expect "picks the GPU with the least memory in use" "GPU-bbb" \
    "$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "picks among more than two GPUs" "GPU-ccc" \
    "$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT='GPU-aaa, 9000\nGPU-bbb, 3000\nGPU-ccc, 100\nGPU-ddd, 5000\n')"

outside_tie=""
for _ in $(seq 20); do
    got="$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT='GPU-aaa, 1200\nGPU-bbb, 1000\nGPU-ccc, 9000\n')"
    case "$got" in
    GPU-aaa | GPU-bbb) ;;
    *) outside_tie="$got" ;;
    esac
done
expect "picks only among GPUs within the tie margin" "" "$outside_tie"

expect "honors a smaller tie margin" "GPU-bbb" \
    "$(run_wrapper NVIDIA_DEVICE_WRAPPER_TIE_MIB=100 \
        FAKE_NVIDIA_SMI_OUTPUT='GPU-aaa, 1200\nGPU-bbb, 1000\n')"

expect "skips lines it cannot parse" "GPU-ccc" \
    "$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT='GPU-aaa, [N/A]\nGPU-bbb, 5000\nGPU-ccc, 10\n')"

expect "passes through with one GPU" "<unset>" \
    "$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT='GPU-aaa, 10\n')"

expect "passes through a multi-GPU test" "<unset>" \
    "$(run_wrapper MODULAR_TEST_GPU_COUNT=2 FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "passes through a test without a GPU count" "<unset>" \
    "$(run_wrapper MODULAR_TEST_GPU_COUNT= FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "keeps a set CUDA_VISIBLE_DEVICES" "1" \
    "$(run_wrapper CUDA_VISIBLE_DEVICES=1 FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "keeps an empty CUDA_VISIBLE_DEVICES" "" \
    "$(run_wrapper CUDA_VISIBLE_DEVICES= FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "fails open when nvidia-smi fails" "<unset>" \
    "$(run_wrapper FAKE_NVIDIA_SMI_EXIT=1 FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "fails open on output it cannot parse" "<unset>" \
    "$(run_wrapper FAKE_NVIDIA_SMI_OUTPUT='No devices were found\n')"

expect "fails open without nvidia-smi" "<unset>" \
    "$(env -u CUDA_VISIBLE_DEVICES PATH=/usr/bin:/bin TEST_TMPDIR=/tmp \
        MODULAR_TEST_GPU_COUNT=1 \
        "$WRAPPER" --tests-only "$FAKE_BIN/show-devices" 2>/dev/null)"

expect "runs a non-test command as is" "<unset>" \
    "$(run_wrapper TEST_TMPDIR= FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS")"

expect "passes the arguments and exit status through" "a b|3" \
    "$(env -u CUDA_VISIBLE_DEVICES PATH="$FAKE_BIN:$PATH" TEST_TMPDIR=/tmp \
        MODULAR_TEST_GPU_COUNT=1 FAKE_NVIDIA_SMI_OUTPUT="$TWO_GPUS" \
        "$WRAPPER" --tests-only "$FAKE_BIN/echo-args-exit-3" a b 2>/dev/null; echo "$?")"

exit "$failures"
