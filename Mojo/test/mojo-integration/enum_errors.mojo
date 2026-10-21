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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

# The defaults of `Equatable.__eq__` and `Hashable.__hash__` compare and hash
# every field, and an enum's only field is its hidden storage, so an enum that
# relies on them gets an error that asks for its own method and does not name
# the storage.

from std.sys import get_defined_int


# TODO: `ImplicitlyCopyable` only so that `Compared.a` can be used as a runtime
# value; a `comptime` value of another type does not materialize.
__enum Compared(Equatable, Hashable, ImplicitlyCopyable):
    case a
    case b(Int)


def main():
    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: constraint failed: deriving `__eq__` is not yet supported for enums; write `__eq__`
        # CHECK_1-NOT: storage
        print(Compared.a == Compared.b(1))
    elif get_defined_int["test"]() == 2:
        # CHECK_2: constraint failed: deriving `__hash__` is not yet supported for enums; write `__hash__`
        # CHECK_2-NOT: storage
        print(hash(Compared.a))
