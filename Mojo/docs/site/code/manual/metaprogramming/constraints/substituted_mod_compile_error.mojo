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

# CHECK: error: invalid call to 'needs_even_size': lacking evidence to prove correctness
# CHECK: note: constraint declared here needs evidence for '((Int(4) % Int(2)) == Int(0))'
# CHECK: note: cannot evaluate call to non-builtin function declared here


# start-substituted-mod
def needs_even_size[n: Int]() where n % 2 == 0:
    pass


def main():
    # Error: constraint declared here needs evidence for
    # '((Int(4) % Int(2)) == Int(0))'
    needs_even_size[4]()
    # end-substituted-mod
