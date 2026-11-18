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

# Test --run-parameter-verifier for `mojo precompile`.
# The parameter verifier is the VerifyParameters pass of the check pipeline,
# so the MLIR timing report shows whether it ran. Each run names the option,
# because the default depends on whether the compiler is a production build.

# RUN: mojo precompile --run-parameter-verifier=true --mlir-timing %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_ON
# RUN: mojo precompile --run-parameter-verifier=false --mlir-timing %S/test_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=CHECK_OFF

# CHECK_ON: Execution time report
# CHECK_ON-DAG: LowerSemanticCF
# CHECK_ON-DAG: VerifyParameters
# CHECK_ON-DAG: CheckLifetimes

# CHECK_OFF: Execution time report
# CHECK_OFF-NOT: VerifyParameters
# CHECK_OFF: LowerSemanticCF
# CHECK_OFF-NOT: VerifyParameters
# CHECK_OFF: CheckLifetimes
# CHECK_OFF-NOT: VerifyParameters

# CHECK_BAD: invalid value 'maybe' for '--run-parameter-verifier=', expected one of: `true`, `false`
