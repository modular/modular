# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

# Test --elaboration-stats. It takes the format and the destination of the
# timing reports: text on stderr by itself, JSON with `--timing-json`, and a
# file with `--timing-file`.

# A warm cache skips elaboration and reports nothing, so every run below needs
# its own empty MODULAR_CACHE_DIR.

# Text on stderr: a report for the host pipeline with the loop in `main`.
# RUN: rm -rf %t.text && env MODULAR_CACHE_DIR=%t.text %mojo-build --elaboration-stats %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_TEXT

# JSON to a file, beside the MLIR timing: both reports are members of one
# object, and stderr stays clean.
# RUN: rm -rf %t.json %t.json.out && env MODULAR_CACHE_DIR=%t.json %mojo-build --elaboration-stats --mlir-timing --timing-json --timing-file=%t.json.out %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_QUIET --allow-empty
# RUN: FileCheck %s --check-prefix=CHECK_JSON --input-file=%t.json.out

# CHECK_TEXT: Elaboration statistics (--elaboration-stats): host
# CHECK_TEXT: comptime for: {{[0-9,]+}} loops, {{[0-9,]+}} instantiations, {{[0-9,]+}} ops produced
# CHECK_TEXT: ops produced instances iterations body ops loop
# CHECK_TEXT: {{[0-9]+}} 1 4 {{[0-9]+}} {{.*}}elaboration_stats.mojo:[[@LINE+44]]:{{[0-9]+}} @{{.*}}main
# CHECK_TEXT: interpreter: {{[0-9,]+}} calls, {{[0-9,]+}} cache hits, {{[0-9,]+}} interpreted in {{[0-9.]+}} s CPU
# CHECK_TEXT: By callee, each function's calls summed over all its call sites:
# CHECK_TEXT-NEXT: CPU s interpreted cache hits max s call sites callee
# CHECK_TEXT: By call site, one operation's calls to one function:
# CHECK_TEXT-NEXT: CPU s interpreted cache hits max s call site callee

# CHECK_QUIET-NOT: Elaboration statistics
# CHECK_QUIET-NOT: Execution time report

# CHECK_JSON: "mlir": [
# CHECK_JSON: "elaboration": [
# CHECK_JSON: "pipeline": "host
# CHECK_JSON: "comptime_for": {
# CHECK_JSON: "loop_count":
# CHECK_JSON: "loops": [
# CHECK_JSON: "file": "{{.*}}elaboration_stats.mojo"
# CHECK_JSON-NEXT: "line": [[@LINE+27]],
# CHECK_JSON: "generator": "{{.*}}main"
# CHECK_JSON: "iterations": {
# CHECK_JSON-NEXT: "min": 4,
# CHECK_JSON-NEXT: "max": 4
# CHECK_JSON: "interpreter": {
# CHECK_JSON-NEXT: "call_count":
# CHECK_JSON-NEXT: "cache_hit_count":
# CHECK_JSON-NEXT: "interpreted_count":
# CHECK_JSON-NEXT: "cpu_seconds":
# CHECK_JSON-NEXT: "callees": [
# CHECK_JSON: "callee":
# CHECK_JSON-NEXT: "call_sites":
# CHECK_JSON-NEXT: "cache_hits":
# CHECK_JSON-NEXT: "interpreted":
# CHECK_JSON-NEXT: "cpu_seconds":
# CHECK_JSON-NEXT: "max_cpu_seconds":
# CHECK_JSON: "calls": [
# CHECK_JSON: "callee":
# CHECK_JSON-NEXT: "cache_hits":
# CHECK_JSON-NEXT: "interpreted":
# CHECK_JSON-NEXT: "cpu_seconds":
# CHECK_JSON-NEXT: "max_cpu_seconds":


def main():
    var total = 0
    comptime for i in range(4):
        total += i
    print(total)
