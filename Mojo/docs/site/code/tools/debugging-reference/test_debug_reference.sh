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
# Debugging reference: manual test suite. See README.md.
#
# Tests the tool behavior that tools/debugging-reference.mdx describes. Uses
# histogram.mojo from tools/debugging.mdx, as the reference page does.
#
# Requires `mojo` on PATH. Run on macOS and on Linux.
#
# Covered by code/tools/debugging/test_debugging.sh, not repeated here:
#   - crash traces and llvm-symbolizer, error-trace forms and platforms,
#     breakpoint() with and without a debugger, value after last use,
#     subscripts and expressions, conditions and ignore counts, break on
#     raise, hidden arguments, finish, pending breakpoints, built programs,
#     attach, scalar type names
# Not tested:
#   - VS Code (needs a person at the editor; see README.md)
#   - Linux platform notes (need a broken environment)

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GUIDE="$SCRIPT_DIR/../../../tools/debugging.mdx"
MOJO=$(command -v mojo) || { echo "mojo not on PATH"; exit 2; }
MOJO_BIN=$(dirname "$MOJO")

WORKDIR=$(mktemp -d /tmp/mojo-debug-ref-test.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR" || exit 2

# The compilation cache can serve an earlier build of the same code, with
# that build's file paths and line numbers. Start every run with an empty
# cache so location checks see this run's files.
export MODULAR_CACHE_DIR="$WORKDIR/cache"

pass=0
fail=0

awk '$0 == "```mojo title=\"histogram.mojo\"" { on = 1; next }
     on && /^```/ { exit }
     on { print }' "$GUIDE" > histogram.mojo
[[ -s histogram.mojo ]] || { echo "could not extract histogram.mojo"; exit 2; }

cat > crash.mojo <<'EOF'
from std.os import abort

def checked_divide(total: Int, count: Int) -> Int:
    if count == 0:
        abort("average of an empty frame")
    return total // count

def main():
    print(checked_divide(784, 0))
EOF

cat > segfault.mojo <<'EOF'
def read_pixel(address: Int) -> UInt8:
    var ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    return ptr[]

def main():
    print(read_pixel(0))
EOF

cat > header.mojo <<'EOF'
def main() raises:
    raise Error("not a PPM header")
EOF

cat > guarded.mojo <<'EOF'
from std.sys import is_defined

def bin_index(value: UInt8, bin_count: Int) -> Int:
    var index = Int(value) * bin_count // 256
    comptime if is_defined["BREAK_ON_BAD_BIN"]():
        if index >= bin_count:
            breakpoint()
    return index

def main():
    print(bin_index(255, 4))
EOF

cat > display.mojo <<'EOF'
from std.utils import Variant

def main():
    var gains: Dict[String, Float64] = {"red": 1.25, "green": 1.0}
    var exposure: Optional[Int] = 3
    var sample: Variant[Int, String] = String("raw")
    var lanes = SIMD[DType.float32, 4](0.5, 1.5, 2.5, 3.5)
    var calibrated = True
    var scale = Float64(0.75)
    var pair = (640, "rgb")
    print(len(gains), exposure.value(), sample.isa[String](), lanes, calibrated, scale, pair[1])
EOF

# --- Helpers ---

report() {  # report <num> <desc> <0|1>
    if [[ $3 == 0 ]]; then
        echo "PASS  $1  $2"; ((pass++))
    else
        echo "FAIL  $1  $2"; ((fail++))
    fi
}

has() { grep -qE -- "$1" <<< "$OUT"; }

DBG_PRE=()
if [[ -f /.dockerenv ]]; then
    DBG_PRE=(--X -O --X "settings set target.disable-aslr false")
fi

dbg() {  # dbg [build options] <target> -- <lldb commands...>, sets OUT
    # Options go before the target; anything after it goes to the program.
    local pre=()
    while [[ $1 != -- ]]; do pre+=("$1"); shift; done; shift
    local target=${pre[${#pre[@]}-1]}
    unset 'pre[${#pre[@]}-1]'
    local args=()
    for c in "$@" "process kill" "quit"; do args+=(--X -o --X "$c"); done
    OUT=$("$MOJO" debug "${DBG_PRE[@]}" "${pre[@]}" "${args[@]}" "$target" 2>&1 < /dev/null)
}

# --- The mojo debug command ---

OUT=$(echo "version" | "$MOJO" debug 2>&1)
has '^lldb version'
report 1 "mojo debug with no program starts the LLDB prompt" $?

OUT=$("$MOJO" debug "${DBG_PRE[@]}" --X -b --X -o --X "b histogram.mojo:25" --X -o --X run \
    --X -o --X "v pixels" --X -o --X kill histogram.mojo 2>&1)
has '\(List\) pixels = \(size 8\)'
report 2 "the --X -b -o example runs its commands and exits" $?

# --- Debug information and optimization levels ---

PATH="$MOJO_BIN:$PATH"
"$MOJO" build -g0 crash.mojo -o crash_g0 > /dev/null 2>&1
"$MOJO" build -g1 crash.mojo -o crash_g1 > /dev/null 2>&1
"$MOJO" build crash.mojo -o crash_default > /dev/null 2>&1

OUT=$(./crash_g0 2>&1)
has ' main' && ! has 'crash\.mojo:9'
report 3 "-g0: function names in crash traces, no line numbers" $?

OUT=$(./crash_g1 2>&1)
has 'checked_divide .*crash\.mojo:5' && has ' main .*crash\.mojo:9'
report 4 "-g1: file and line numbers in crash traces" $?

OUT=$(./crash_default 2>&1)
! has ' main .*crash\.mojo:9'
report 5 "default debug level is none" $?

dbg -O3 histogram.mojo -- "b bin_index" "run" "bt 1"
! has 'stop reason = breakpoint'
report 6 "-O3: a breakpoint on an inlined function never stops" $?

dbg -O0 histogram.mojo -- "b bin_index" "run" "bt 1"
has 'stop reason = breakpoint' && has 'bin_index\(value=12'
report 7 "-O0: the same breakpoint stops" $?

# --- Programmatic breakpoints ---

OUT=$("$MOJO" run guarded.mojo 2>&1)
has '^3$'
report 8 "guarded breakpoint() doesn't run without the define" $?

# --- Stack traces ---

"$MOJO" build -O0 -g1 segfault.mojo -o segfault > /dev/null 2>&1
OUT=$(./segfault 2>&1)
has 'main\(\) .*segfault\.mojo:6'
report 9 "SIGSEGV prints a stack trace by default" $?

OUT=$("$MOJO" run header.mojo 2>&1); status=$?
has 'Unhandled exception caught during execution: not a PPM header'
report 10 "unhandled error prints its message" $?
"$MOJO" build header.mojo -o header > /dev/null 2>&1
./header > /dev/null 2>&1; status=$?
report 11 "unhandled error exits with status 1" "$(( status == 1 ? 0 : 1 ))"

# --- MODULAR_DEBUG ---

OUT=$(MODULAR_DEBUG=stack-trace-on-error,stack-trace-on-crash "$MOJO" run header.mojo 2>&1)
! has 'unknown option' && ! has 'stack trace was not collected'
report 12 "a comma-separated list sets both entries" $?

OUT=$(MODULAR_DEBUG=stack_trace_on_error "$MOJO" run header.mojo 2>&1)
has "unknown option 'stack_trace_on_error'; ignoring" && has 'stack trace was not collected'
report 13 "underscore spelling warns and is ignored" $?

OUT=$(MODULAR_DEBUG=stack-trace-on-crash=false ./crash_g1 2>&1)
has "unknown option 'stack-trace-on-crash'" && has 'checked_divide'
report 14 "no form turns an entry off" $?

# --- LLDB with Mojo ---

dbg display.mojo -- "b display.mojo:11" "run" "v"
has '\(Dict\) gains = \(size 2\)' && has 'key = "red"' && has 'value = 1\.25'
report 15 "Dict: size, then each key and value" $?
has 'lanes = \(\[0\] = 0\.5, \[1\] = 1\.5, \[2\] = 2\.5, \[3\] = 3\.5\)'
report 16 "SIMD: each element" $?
has 'calibrated = True' && has 'scale = 0\.75'
report 17 "scalars: the value" $?
has '\(Optional\) exposure = \{' && has '_value = \{'
report 18 "Optional: internal storage fields" $?
has '\(Variant\) sample = \{' && has '_storage = \{'
report 19 "Variant: internal storage fields" $?
has '\(Tuple\) pair = \{' && ! has '\[1\] = "rgb"'
report 20 "string inside a Tuple: not shown as text" $?

dbg histogram.mojo -- "b bin_index" "run" "bt 2"
has 'JIT\(0x[0-9a-f]+\)`histogram::bin_index'
report 21 "frames carry a JIT(...) prefix" $?
[[ $(grep -cE '^ +\*? *frame #' <<< "$OUT") -le 3 ]] && ! has 'frame #2:'
report 22 "bt N shows N frames" $?

# --- Summary ---

echo ""
echo "==========================="
echo "$(uname -s): $pass passed, $fail failed"
[[ $fail -eq 0 ]]
