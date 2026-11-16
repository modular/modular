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

# Test --parser-stats. It takes the format and the destination of the timing
# reports: text on stderr by itself, JSON with `--timing-json`, and a file with
# `--timing-file`.

# RUN: %mojo-build --parser-stats %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_TEXT

# RUN: rm -f %t.json && %mojo-build --parser-stats --timing-json --timing-file=%t.json %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_QUIET --allow-empty
# RUN: FileCheck %s --check-prefix=CHECK_JSON --input-file=%t.json

# CHECK_TEXT: Parser statistics (--parser-stats): {{.*}}parser_stats.mojo
# CHECK_TEXT: parse: {{[0-9.]+}} s wall, {{[0-9,]+}} declarations
# CHECK_TEXT: Self time by activity:
# CHECK_TEXT: signature
# CHECK_TEXT-NEXT: body
# CHECK_TEXT-NEXT: synthetic_signature
# CHECK_TEXT-NEXT: synthetic_body
# CHECK_TEXT-NEXT: bytecode
# CHECK_TEXT-NEXT: unattributed
# CHECK_TEXT-NEXT: profiler_overhead
# CHECK_TEXT: Declarations by self time:
# CHECK_TEXT-NEXT: self s signature s body s other s frames declaration

# CHECK_QUIET-NOT: Parser statistics

# CHECK_JSON: "parser": [
# CHECK_JSON: "module": "{{.*}}parser_stats.mojo"
# CHECK_JSON-NEXT: "total_seconds":
# CHECK_JSON-NEXT: "unattributed_seconds":
# CHECK_JSON-NEXT: "profiler_overhead_seconds":
# CHECK_JSON-NEXT: "activities": {
# CHECK_JSON-NEXT: "signature":
# CHECK_JSON-NEXT: "body":
# CHECK_JSON-NEXT: "synthetic_signature":
# CHECK_JSON-NEXT: "synthetic_body":
# CHECK_JSON-NEXT: "bytecode":
# CHECK_JSON: "declarations": [

# A closure resolves its body while it resolves its signature. The body still
# counts as the closure's body, not as its signature.
# CHECK_JSON: "name": "parser_stats.main.inner"
# CHECK_JSON-NEXT: "kind": "def"
# CHECK_JSON-NEXT: "location": "{{.*}}parser_stats.mojo:[[@LINE+10]]:{{[0-9]+}}"
# CHECK_JSON-NEXT: "self_seconds":
# CHECK_JSON-NEXT: "frames": 2
# CHECK_JSON-NEXT: "activities": {
# CHECK_JSON-NEXT: "signature":
# CHECK_JSON-NEXT: "body":
# CHECK_JSON-NEXT: }


def main():
    def inner() -> Int:
        return 2

    print(inner())
