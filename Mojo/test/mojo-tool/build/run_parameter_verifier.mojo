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

# Test --run-parameter-verifier for `mojo build`. The
# option decides whether the pipeline runs EnsureNoParameters, which the MLIR
# timing report shows. Each run names the option, because the default depends
# on whether the compiler is a production build. An empty MODULAR_CACHE_DIR
# keeps a cached pipeline from hiding the passes.

# RUN: rm -rf %t.on && env MODULAR_CACHE_DIR=%t.on %mojo-build --run-parameter-verifier=true --mlir-timing %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_ON
# RUN: rm -rf %t.off && env MODULAR_CACHE_DIR=%t.off %mojo-build --run-parameter-verifier=false --mlir-timing %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_OFF

# CHECK_ON: Execution time report
# CHECK_ON: EnsureNoParameters

# CHECK_OFF: Execution time report
# CHECK_OFF-NOT: EnsureNoParameters
# CHECK_OFF: LowerKGENToLLVM


def main():
    pass
