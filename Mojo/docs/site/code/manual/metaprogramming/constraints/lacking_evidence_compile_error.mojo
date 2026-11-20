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

# CHECK: error: invalid call to 'print_first': lacking evidence to prove correctness
# CHECK: note: constraint declared here needs evidence for '(size >= Int(1))'


# start-lacking-evidence
from std.collections.array import Array


# A function that wants a non-empty list.
def print_first[size: Int](l: Array[Int, size]) where size >= 1:
    print(l[0])


# A wrapper that wants a size >= 2 list.
def print_first_two[size: Int](l: Array[Int, size]) where size >= 2:
    # Error: invalid call to 'print_first': lacking evidence to prove
    # correctness
    print_first[size](l)
    print(l[1])
    # end-lacking-evidence


def main():
    var l: Array[Int, 3] = [1, 2, 3]
    print_first_two[3](l)
