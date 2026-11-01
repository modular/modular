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

# Test that -D defines reach the module environment.

# RUN: %parse-mojo-isolated -D FOO=1 -D BAR %s | FileCheck %s
# RUN: not %parse-mojo-isolated -D FOO=1 -D FOO=2 %s 2>&1 | FileCheck %s --check-prefix=DUPLICATE

# CHECK: module attributes {kgen.env = #kgen.env<{BAR, FOO = 1 : index,

# DUPLICATE: error: 'FOO=2' was defined more than once


def main():
    pass
