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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

from std.builtin.stubs import _get_kgen_string

# ===----------------------------------------------------------------------=== #
# @__llvm_metadata
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"kernel{{.*}}"<x:
# CHECK-SAME: fnAttrs = ["nvvm.maxntid", {{.*}}#pop.array<x> : !pop.array<
@__llvm_metadata(
    `nvvm.maxntid`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def kernel[x: Int]():
    pass


# CHECK-LABEL: lit.fn @"kernel_1{{.*}}"<x:
# CHECK-SAME: fnAttrs = [#lit.struct<{}> : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
comptime mname = "nvvm.maxntid"


@__llvm_metadata(
    mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def kernel_1[x: Int]():
    pass


# CHECK-LABEL: lit.fn @"kernel_2{{.*}}"<x:
# CHECK-SAME: fnAttrs = [#lit.struct<{}> : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
@__llvm_metadata(
    `mname`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def kernel_2[x: Int]():
    pass


# TODO: Figure out how to get the value of the alias.
# CHECK-LABEL: lit.fn @"kernel_3{{.*}}"<x:
# CHECK-SAME: fnAttrs = [
# CHECK-SAME: data_to_str({{.*}}alias_parametric_fn
# CHECK-SAME: #lit.struct<{}> : !lit.struct<#IntLiteral <:!pop.int_literal 128>>
def alias_parametric_fn() -> StaticString:
    comptime if True:
        return "nvvm.maxntid"
    else:
        return "rocdl.flat_work_group_size"


comptime mname1 = _get_kgen_string[alias_parametric_fn()]()


@__llvm_metadata(mname1=128)
def kernel_3[x: Int]():
    pass


# ===----------------------------------------------------------------------=== #
# @__llvm_arg_metadata
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"llvm_arg_meta
# CHECK-SAME{LITERAL}: fnArgAttrs = [[], ["nvvm.grid_constant", unit, "myMeta", unit], [], [#lit.struct<{}> : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, #pop.array<x> : !pop.array<1, !Int>], []]
@__llvm_arg_metadata(b, `nvvm.grid_constant`, `myMeta`)
@__llvm_arg_metadata(c)
@__llvm_arg_metadata(
    d, mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def llvm_arg_meta[x: Int](a: Int, b: Int, c: Int, d: Int, e: Int):
    pass
