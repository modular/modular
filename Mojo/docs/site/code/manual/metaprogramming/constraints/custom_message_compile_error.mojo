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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: error: invalid call to 'pow2': violated constraint
# CHECK: note: constraint declared here evaluated to False, expected '(n >= Int(0))': n must be nonnegative


# start-custom-message
def pow2[n: Int]() -> Int where n >= 0 else "n must be nonnegative":
    return 2**n


def main():
    print(pow2[-1]())
    # end-custom-message
