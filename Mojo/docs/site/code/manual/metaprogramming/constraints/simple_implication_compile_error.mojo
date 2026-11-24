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

# CHECK: error: invalid call to 'create_even_list': violated constraint
# CHECK: note: constraint declared here evaluated to False


def create_list[T: Copyable, size: Int]() -> List[T] where size >= 0:
    return List[T](capacity=size)


def create_even_list[
    T: Copyable, size: Int
]() -> List[T] where size >= 0 and (size / 2) * 2 == size:
    return create_list[T, size=size]()


def main():
    # start-simple-implication-odd
    # Won't compile: 5 isn't even.
    var l2 = create_even_list[Int, 5]()
    # end-simple-implication-odd
