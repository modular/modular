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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# The shape of the t-string lowering; basic.mojo covers the format template it
# builds. Each value is wrapped in a `_FormatArgument` in the frame that owns
# it, an `Array` list literal collects the wrappers, and `__make_tstring`
# borrows the resulting array.


# CHECK-LABEL: lit.fn @"no_interpolations()"
def no_interpolations():
    # No `_FormatArgument` to build, and the bound origin is the empty one.
    # CHECK-NOT: @_FormatArgument::@"__init__
    # CHECK: lit.call @{{.*}}__list_literal__{{.*}}@_FormatArgument<:origin<false> {},{{.*}}:!Int {:scalar<index> 0},
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hello"
    var s = t"Hello"


# CHECK-LABEL: lit.fn @"one_interpolation()"
def one_interpolation():
    var name = "Alice"
    # The wrapper binds a reference to `name`'s own storage.
    # CHECK: lit.call @{{.*}}@_FormatArgument::@"__init__{{.*}}"writable": !lit.ref<!String, muttoimm *"name`"> ref
    # CHECK: lit.call @{{.*}}__list_literal__{{.*}}:!Int {:scalar<index> 1},
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hi {}!"
    var s = t"Hi {name}!"


# CHECK-LABEL: lit.fn @"two_interpolations()"
def two_interpolations():
    var x = 10
    var y = 20
    # One wrapper per interpolation, both carrying the union of both origins.
    # CHECK: lit.call @{{.*}}@_FormatArgument::@"__init__{{.*}}"writable": !lit.ref<!Int, imm {(mutcast mut *"x`"), (mutcast mut *"y`1")}> ref
    # CHECK: lit.call @{{.*}}@_FormatArgument::@"__init__{{.*}}"writable": !lit.ref<!Int, imm {(mutcast mut *"x`"), (mutcast mut *"y`1")}> ref
    # CHECK: lit.call @{{.*}}__list_literal__{{.*}}:!Int {:scalar<index> 2},
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{} and {}"
    var s = t"{x} and {y}"


# CHECK-LABEL: lit.fn @"interpolated_literal()"
def interpolated_literal():
    # The wrapper is built over the materialized `Int`, not an `IntLiteral`.
    # CHECK: lit.call @{{.*}}@_FormatArgument::@"__init__{{.*}}:!AnyType !Int>
    # CHECK: lit.call @{{.*}}__list_literal__{{.*}}:!Int {:scalar<index> 1},
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}"
    var s = t"{42}"
