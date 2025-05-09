# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: mojo %s | FileCheck %s

import _mlir
from max.graph import _c


fn test_empty_graph() raises:
    with _mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        var unknown = _mlir.Location.unknown(ctx)
        var m = _mlir.Module(unknown)

        var signature = _mlir.builtin_types.FunctionType(ctx)

        _ = _c.graph_new(m, unknown, "empty_graph", signature)

        # CHECK: builtin.module
        # CHECK: mo.graph
        # CHECK: name = "empty_graph"
        print(m)


fn test_identity_graph() raises:
    with _mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        var unknown = _mlir.Location.unknown(ctx)
        var m = _mlir.Module(unknown)

        var intScalar = _mlir.Type.parse(ctx, "!mo.scalar<si32>")

        var ins = List[_mlir.Type]()
        ins.append(intScalar)
        var outs = List[_mlir.Type]()
        outs.append(intScalar)

        var signature = _mlir.builtin_types.FunctionType(ins, outs)
        var g = _c.graph_new(m, unknown, "identity_graph", signature)

        var block = g.region(0).first_block()
        var arg0 = block.argument(0)
        var operands = List[_mlir.Value]()
        operands.append(arg0)

        block.append(_mlir.Operation("mo.output", unknown, operands=operands))

        # CHECK: mo.graph @identity_graph
        # CHECK: mo.output
        print(m)


def main():
    test_empty_graph()
    test_identity_graph()
