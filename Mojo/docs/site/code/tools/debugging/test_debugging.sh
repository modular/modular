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
# Debugging page: manual test suite. See README.md.
#
# Tests the tool behavior that debugging.mdx describes: crash traces,
# breakpoints, and LLDB sessions. Extracts histogram.mojo, crash.mojo, and
# the stderr sample from the page itself, so the code and line numbers match
# the page.
#
# Requires `mojo` on PATH. Run on macOS and on Linux; some claims differ by
# platform, and the script checks the platform's expected result.
#
# Not tested here:
#   - Debug in VS Code (needs a person at the editor; see README.md)
#   - Debug GPU code (link only)
#   - Troubleshooting (needs a broken environment)

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAGE="$SCRIPT_DIR/../../../tools/debugging.mdx"
MOJO=$(command -v mojo) || { echo "mojo not on PATH"; exit 2; }
MOJO_BIN=$(dirname "$MOJO")
OS=$(uname -s)

WORKDIR=$(mktemp -d /tmp/mojo-debugging-test.XXXXXX)
trap 'rm -rf "$WORKDIR"; [[ -n "${W:-}" ]] && kill "$W" 2>/dev/null' EXIT
cd "$WORKDIR" || exit 2

# The compilation cache can serve an earlier build of the same code, with
# that build's file paths and line numbers. Start every run with an empty
# cache so location checks see this run's files.
export MODULAR_CACHE_DIR="$WORKDIR/cache"

pass=0
fail=0
skip=0

extract() {  # extract <title> <out-file>: copy a titled mojo block from the page
    awk -v t="\`\`\`mojo title=\"$1\"" '
        $0 == t { on = 1; next }
        on && /^```/ { exit }
        on { print }' "$PAGE" > "$2"
    [[ -s "$2" ]] || { echo "could not extract $1 from $PAGE"; exit 2; }
}
extract histogram.mojo histogram.mojo
extract crash.mojo crash.mojo

# The stderr sample has no title: take the first mojo block in its section.
awk '/^## Print values as the program runs/ { s = 1 }
     s && $0 == "```mojo" { on = 1; next }
     on && /^```/ { exit }
     on { print }' "$PAGE" > stderr_sample.mojo
[[ -s stderr_sample.mojo ]] || { echo "could not extract the stderr sample"; exit 2; }
grep -v '^from std.sys import stderr$' stderr_sample.mojo > stderr_noimport.mojo

cat > paused.mojo <<'EOF'
def checksum(pixels: List[UInt8]) -> Int:
    var total = 0
    for p in pixels:
        total += Int(p)
    return total

def main():
    var pixels: List[UInt8] = [255, 128, 64]
    var label = String("frame-07")
    var total = checksum(pixels)
    breakpoint()
    print(label, total)
EOF

cat > header.mojo <<'EOF'
from std.sys import argv

def parse_header(line: String, form: String) raises -> Int:
    if not line.startswith("P6"):
        if form == "literal":
            raise Error("not a PPM header")
        elif form == "args":
            raise Error("not a PPM header: ", line)
        else:
            raise Error("not a PPM header: " + line)
    return 6

def main() raises:
    var args = argv()
    print(parse_header("GIF89a", String(args[1])))
EOF

cat > watcher.mojo <<'EOF'
from std.time import sleep

def main():
    var tick = 0
    while True:
        tick += 1
        sleep(0.01)
EOF

# --- Helpers ---

report() {  # report <num> <desc> <0|1> [detail]
    if [[ $3 == 0 ]]; then
        echo "PASS  $1  $2"; ((pass++))
    else
        echo "FAIL  $1  $2${4:+ ($4)}"; ((fail++))
    fi
}

skipped() { echo "SKIP  $1  $2 ($3)"; ((skip++)); }

has() { grep -qE -- "$1" <<< "$OUT"; }

DBG_PRE=()
if [[ -f /.dockerenv ]]; then
    # Docker's default seccomp profile blocks LLDB's ASLR change.
    DBG_PRE=(--X -O --X "settings set target.disable-aslr false")
fi

dbg() {  # dbg <target> <lldb commands...>: scripted LLDB session, sets OUT
    # No -b: batch mode stops at the first failed command or trap stop, and
    # several claims are about commands that fail.
    local target=$1; shift
    local args=()
    for c in "$@" "process kill" "quit"; do args+=(--X -o --X "$c"); done
    OUT=$("$MOJO" debug "${DBG_PRE[@]}" "${args[@]}" "$target" 2>&1 < /dev/null)
}

# --- Print values as the program runs ---

OUT=$("$MOJO" run stderr_sample.mojo 2>&1 >/dev/null)
has '^pixels read: 4$'
report 1a "stderr sample: file=stderr writes to standard error" $?

OUT=$("$MOJO" run stderr_sample.mojo 2>/dev/null)
has '^200$' && ! has 'pixels read'
report 1b "stderr sample: normal output stays on standard output" $?

OUT=$("$MOJO" run stderr_noimport.mojo 2>&1)
has "unknown declaration 'stderr'"
report 1c "stderr needs the import from std.sys" $?

OUT=$("$MOJO" run -D TRACE_BINS histogram.mojo 2>&1 >/dev/null)
has '^bin_index: 12 -> 0$'
report 1 "-D TRACE_BINS writes the trace to stderr" $?

OUT=$("$MOJO" run -D TRACE_BINS histogram.mojo 2>/dev/null)
! has 'bin_index:'
report 2 "trace output stays out of stdout" $?

OUT=$("$MOJO" run histogram.mojo 2>&1)
! has 'bin_index:' && has 'frame-07.ppm bin 3 -> 3'
report 3 "without -D TRACE_BINS, no trace" $?

# --- Read a crash report ---

"$MOJO" build -g1 crash.mojo -o crash_g1 > /dev/null 2>&1
"$MOJO" build crash.mojo -o crash_g0 > /dev/null 2>&1

OUT=$(PATH="$MOJO_BIN:/usr/bin:/bin" ./crash_g1 2>&1)
has 'crash\.mojo:6:14: average of an empty frame'
report 4 "abort() prints its message with file and line" $?

has 'checked_divide .*crash\.mojo:6' && has ' main .*crash\.mojo:11'
report 5 "-g1 + llvm-symbolizer: checked_divide and main with lines" $?

OUT=$(LLVM_SYMBOLIZER_PATH="$MOJO_BIN/llvm-symbolizer" PATH=/usr/bin:/bin ./crash_g1 2>&1)
has 'checked_divide .*crash\.mojo:6'
report 6 "LLVM_SYMBOLIZER_PATH works in place of PATH" $?

OUT=$(PATH="$MOJO_BIN:/usr/bin:/bin" ./crash_g0 2>&1)
! has 'checked_divide .*crash\.mojo:6' && ! has ' main .*crash\.mojo:11'
report 7 "without debug info: no line numbers in frames" $?

OUT=$(PATH=/usr/bin:/bin ./crash_g1 2>&1)
has 'Stack dump without symbol names' && ! has 'crash\.mojo:11'
report 8 "without llvm-symbolizer: no line numbers" $?

OUT=$("$MOJO" run header.mojo string 2>&1)
has 'Unhandled exception caught during execution: not a PPM header' \
    && ! has 'parse_header'
report 9 "unhandled error: message, no stack trace" $?

# --- Stop at a breakpoint ---

OUT=$("$MOJO" run paused.mojo 2>&1)
has 'execution crashed'
report 10 "breakpoint() without a debugger crashes the program" $?

dbg paused.mojo "run" "bt 1"
has 'stop reason = EXC_BREAKPOINT|stop reason = signal SIGTRAP' && has 'paused\.mojo:12'
report 11 "breakpoint() under mojo debug stops; shows the next line" $?

# --- Debug from the command line ---

dbg histogram.mojo "b histogram.mojo:25" "run" "v pixels" "s" "finish" "up" "v source"
has "Unable to resolve breakpoint" && has 'location added to breakpoint'
report 12 "breakpoint set before run warns, then resolves" $?
has 'histogram\.mojo:25'
report 13 "b histogram.mojo:25 stops on line 25" $?
has '\(List\) pixels = \(size 8\)\[12, 200, 37, 255, 128, 90, 64, 241\]'
report 14 "v pixels shows the list's size and elements" $?
has 'stop reason = step in' && has 'bin_index\(value=12'
report 15 "s steps into bin_index()" $?
has 'stop reason = step out' && ! has 'Return value'
report 16 "finish returns to the loop, without a return value" $?
has '\(String\) source = "frame-07\.ppm"'
report 17 "up, then v source shows the string" $?

dbg histogram.mojo "b build_histogram" "run" "bt 3"
has 'frame #0: .*build_histogram' && has 'frame #1: .*main'
report 18 "b build_histogram stops there; bt shows the call stack" $?
has '__error__='
report 19 "frames list compiler-generated arguments" $?

dbg histogram.mojo "breakpoint set -f histogram.mojo -l 25 -i 4" "run" "v p"
has 'p = 128'
report 20 "-i 4 skips four hits (stops on the fifth pixel)" $?
has '!kgen\.scalar<ui8>'
report 21 "scalars show internal type names" $?

dbg histogram.mojo "breakpoint set -f histogram.mojo -l 25 -c 'p > 200'" "run" "v p"
has 'error evaluating condition' && has 'p = 12$'
report 22 "a breakpoint condition fails and stops on the first hit" $?

# A failed command ends LLDB's command stream, so each failing command
# gets its own session and runs last.
dbg histogram.mojo "b histogram.mojo:33" "run" "v hist" "v hist.bins" "p hist.samples"
has 'bins = \(size 4\)\[2, 2, 1, 3\]' && has 'samples = 8'
report 23 "v shows a struct field by field" $?
has '\(List\) hist\.bins = \(size 4\)'
report 24 "v accepts field paths" $?
has '\) 8$'
report 25a "p reads a field" $?

dbg histogram.mojo "b histogram.mojo:33" "run" "p hist.samples + 1"
! has '\) 9$'
report 25b "p doesn't evaluate arithmetic" $?

dbg histogram.mojo "b histogram.mojo:33" "run" "v hist.bins[1]"
has 'error: integer literal is too large'
report 26 "v rejects subscripts" $?

dbg paused.mojo "run" "v pixels" "v label"
has 'pixels = <variable not available>' && has 'label = "frame-07"'
report 27 "a value after its last use shows as not available" $?

dbg histogram.mojo "breakpoint set -E mojo" "run" "bt 1"
has 'range\.mojo.*__next__|__next__.*range\.mojo'
report 28 "break on raise stops inside a for loop's iterator" $?

# --- Debug a built program or a running process ---

"$MOJO" build -O0 -g histogram.mojo -o histogram > /dev/null 2>&1
dbg ./histogram "b histogram.mojo:25" "run" "v p"
if [[ $OS == Darwin ]]; then
    ! has 'p = 12'
    report 29 "macOS: built program shows no source-level values" $?
else
    has 'p = 12'
    report 29 "Linux: built program stops at a source line" $?
fi

"$MOJO" build -O0 -g watcher.mojo -o watcher > /dev/null 2>&1
./watcher > /dev/null 2>&1 & W=$!
sleep 1
OUT=$("$MOJO" debug --pid "$W" --X -b --X -o --X "bt 6" --X -o --X detach 2>&1)
if has 'Not allowed to attach|ptrace_scope|Operation not permitted'; then
    skipped 30 "attach by PID" "this environment denies ptrace"
elif [[ $OS == Darwin ]]; then
    has 'frame #'
    report 30 "macOS: attach by PID connects" $?
else
    has 'watcher\.mojo'
    report 30 "Linux: attach by PID shows Mojo source lines" $?
fi
kill "$W" 2>/dev/null; W=

# --- Limitations: raised errors ---

for form in literal args; do
    OUT=$(MODULAR_DEBUG=stack-trace-on-error PATH="$MOJO_BIN:$PATH" "$MOJO" run header.mojo $form 2>&1)
    ! has 'stack trace was not collected'
    report "31$form" "stack-trace-on-error collects a trace ($form form)" $?
done
OUT=$(MODULAR_DEBUG=stack-trace-on-error "$MOJO" run header.mojo string 2>&1)
has 'stack trace was not collected'
report 32 "single String form: no trace" $?

"$MOJO" build -O0 -g header.mojo -o header > /dev/null 2>&1
OUT=$(MODULAR_DEBUG=stack-trace-on-error PATH="$MOJO_BIN:$PATH" ./header literal 2>&1)
if [[ $OS == Darwin ]]; then
    ! has 'header\.mojo:6'
    report 33 "macOS: error trace stops before the raising line" $?
else
    has 'parse_header.*header\.mojo:6'
    report 33 "Linux: error trace reaches the raising line" $?
fi

# --- Troubleshooting: file paths or line numbers are wrong ---

mkdir -p cache_a/first cache_a/second shared_cache empty_cache
printf 'from std.os import abort\n\ndef fail_here(count: Int) -> Int:\n    if count == 0:\n        abort("empty frame")\n    return count\n\ndef main():\n    print(fail_here(0))\n' > cache_a/first/probe.mojo
{ printf '\n\n'; cat cache_a/first/probe.mojo; } > cache_a/second/probe.mojo
for d in first second; do
    (cd "cache_a/$d" && MODULAR_CACHE_DIR="$WORKDIR/shared_cache" "$MOJO" build -g1 probe.mojo -o probe > /dev/null 2>&1)
done
OUT=$(./cache_a/second/probe 2>&1)
has 'first/probe\.mojo:5:14'
report 34 "a cached build reports the earlier build's file and line" $?
(cd cache_a/second && MODULAR_CACHE_DIR="$WORKDIR/empty_cache" "$MOJO" build -g1 probe.mojo -o probe > /dev/null 2>&1)
OUT=$(./cache_a/second/probe 2>&1)
has 'second/probe\.mojo:7:14'
report 35 "an empty MODULAR_CACHE_DIR gives the correct file and line" $?

# --- Summary ---

echo ""
echo "==========================="
echo "$OS: $pass passed, $fail failed, $skip skipped"
[[ $fail -eq 0 ]]
