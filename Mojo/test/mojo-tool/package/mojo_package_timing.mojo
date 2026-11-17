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

# Test the timing reports of `mojo precompile`: --mlir-timing, --parser-stats,
# and their format and destination with --timing-json and --timing-file.

# The MLIR timing covers the parse and the check passes.
# RUN: mojo precompile --mlir-timing %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_MLIR

# The timing stays off if the option is not present.
# RUN: mojo precompile %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_OFF --allow-empty

# RUN: not mojo precompile --mlir-timing-display=bogus %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_BAD_MODE

# RUN: mojo precompile --parser-stats %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_PARSER

# JSON to a file. Both reports become members of one object, and stderr stays
# clean.
# RUN: rm -f %t.json && mojo precompile --parser-stats --mlir-timing --timing-json --timing-file=%t.json %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_QUIET --allow-empty
# RUN: FileCheck %s --check-prefix=CHECK_JSON --input-file=%t.json

# The backend timing option stays rejected, because precompile runs no LLVM.
# RUN: not mojo precompile --llvm-timing %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_NO_LLVM

# CHECK_MLIR: MLIR pass timing (--mlir-timing)
# CHECK_MLIR: Execution time report
# CHECK_MLIR-DAG: Import Mojo
# CHECK_MLIR-DAG: LowerSemanticCF
# CHECK_MLIR-DAG: CheckLifetimes

# CHECK_OFF-NOT: Execution time report
# CHECK_OFF-NOT: Parser statistics

# CHECK_BAD_MODE: error: invalid mlir-timing-display 'bogus'

# CHECK_PARSER: Parser statistics (--parser-stats): {{.*}}test_package
# CHECK_PARSER: parse: {{[0-9.]+}} s wall, {{[0-9,]+}} declarations
# CHECK_PARSER: Declarations by self time:

# CHECK_QUIET-NOT: Execution time report
# CHECK_QUIET-NOT: Parser statistics

# CHECK_JSON-DAG: "parser": [
# CHECK_JSON-DAG: "module": "{{.*}}test_package"
# CHECK_JSON-DAG: "mlir": [
# CHECK_JSON-DAG: "name": "Import Mojo"
# CHECK_JSON-DAG: "name": "CheckLifetimes"

# CHECK_NO_LLVM: error: unrecognized argument '--llvm-timing'
