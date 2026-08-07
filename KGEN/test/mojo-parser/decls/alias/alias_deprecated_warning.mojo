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

# RUN: %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Test that the 'alias' keyword issues a deprecation warning

# CHECK: warning: 'alias' is deprecated; use 'comptime'
alias MY_CONSTANT = 42


struct MyStruct(Movable where False):
    # CHECK: warning: 'alias' is deprecated; use 'comptime'
    alias SIZE = Int


# Test that 'comptime' does NOT issue a warning about being deprecated
# CHECK-NOT: comptime NEW_CONSTANT
comptime NEW_CONSTANT = 99
