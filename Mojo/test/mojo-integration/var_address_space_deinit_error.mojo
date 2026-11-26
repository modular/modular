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

# A variable outside the default address space must have a type that the
# stdlib's `IsTriviallyDeinitable` accepts. Generic code establishes it with a
# `where` clause.

from std.memory import MaybeUninit
from std.traits import IsTriviallyDeinitable


def concrete_types():
    @__address_space(.SHARED)
    var trivial: Int32

    @__address_space(.SHARED)
    var uninit_trivial: MaybeUninit[Int32]

    # CHECK: [[@LINE+2]]:9: error: variable 'string' in address space '3' must have a trivially deinitable type, not 'String'
    @__address_space(.SHARED)
    var string: String

    # CHECK: [[@LINE+2]]:9: error: variable 'linear' in address space '3' must have a trivially deinitable type, not 'MaybeUninit[String]'
    @__address_space(.SHARED)
    var linear: MaybeUninit[String]


def generic_type[T: Deinitable]():
    # CHECK: [[@LINE+2]]:9: error: variable 'value' in address space '3' must have a trivially deinitable type; add 'where IsTriviallyDeinitable[T]'
    @__address_space(.SHARED)
    var value: T


def generic_type_with_where_clause[
    T: Deinitable
]() where IsTriviallyDeinitable[T]:
    @__address_space(.SHARED)
    var value: T


# CHECK-NOT: :{{[0-9]+}}:{{[0-9]+}}: error:


def main():
    concrete_types()
    generic_type[Int]()
    generic_type_with_where_clause[Int]()
