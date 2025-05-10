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
# RUN: mojo %s

import _mlir
import testing
from max.graph import Dim, ListType, TensorType
from max.tensor import TensorSpec


fn context() -> _mlir.Context:
    var ctx = _mlir.Context()
    ctx.load_modular_dialects()
    ctx.load_all_available_dialects()
    return ctx


fn test_tensor_type() raises:
    var t = TensorType(DType.uint8, 1, 2, 3)

    var s = String(t.to_mlir(context()))
    testing.assert_equal(s, "!mo.tensor<[1, 2, 3], ui8>")


fn test_tensor_unranked() raises:
    var unranked_tensor_mlir = _mlir.Type.parse(context(), "!mo.tensor<?, f32>")
    with testing.assert_raises():
        _ = TensorType.from_mlir(unranked_tensor_mlir)


fn test_tensor_dynamic_dims() raises:
    var t = TensorType(DType.float32, 1, Dim.dynamic(), 4)

    var s = String(t.to_mlir(context()))
    testing.assert_equal(s, "!mo.tensor<[1, ?, 4], f32>")


fn test_tensor_dynamic_dims_roundtrip() raises:
    var t = TensorType(DType.float32, Dim.dynamic(), 4)
    var t2 = TensorType.from_mlir(t.to_mlir(context()))
    testing.assert_true(t == t2, "roundtrip dynamic dim")


fn test_tensor_symbolic_dims() raises:
    var t = TensorType(DType.float32, Dim.symbolic("batch"), 4)

    var s = String(t.to_mlir(context()))
    testing.assert_equal(s, "!mo.tensor<[batch, 4], f32>")


fn test_tensor_symbolic_dims_roundtrip() raises:
    var t = TensorType(DType.float32, Dim.symbolic("batch"), 4)
    var t2 = TensorType.from_mlir(t.to_mlir(context()))
    print(t.to_mlir(context()))
    print(t2.to_mlir(context()))
    testing.assert_true(t == t2, "roundtrip symbolic dim")


fn test_tensor_scalar() raises:
    var t = TensorType(DType.float32)

    var s = String(t.to_mlir(context()))
    testing.assert_equal(s, "!mo.tensor<[], f32>")


fn test_dtype_autocast() raises:
    testing.assert_equal(
        String(TensorType(DType.float32, 1, 2, 3).to_mlir(context())),
        "!mo.tensor<[1, 2, 3], f32>",
    )


fn test_int_autocast() raises:
    # This just verifes that the constructor doesn't go to the wrong overload,
    # like the List one.
    testing.assert_equal(
        String(TensorType(DType.float32, 1).to_mlir(context())),
        "!mo.tensor<[1], f32>",
    )


fn test_tensor_spec_cast() raises:
    testing.assert_equal(
        String(TensorType(TensorSpec(DType.float32)).to_mlir(context())),
        "!mo.tensor<[], f32>",
    )
    testing.assert_equal(
        String(TensorType(TensorSpec(DType.float64, 1)).to_mlir(context())),
        "!mo.tensor<[1], f64>",
    )
    testing.assert_equal(
        String(TensorType(TensorSpec(DType.int32, 1, 2)).to_mlir(context())),
        "!mo.tensor<[1, 2], si32>",
    )


fn test_list_type() raises:
    var l = ListType(TensorType(DType.uint8, 1, Dim.dynamic(), 3))

    var s = String(l.to_mlir(context()))
    testing.assert_equal(s, "!mo.list<!mo.tensor<[1, ?, 3], ui8>>")


fn main() raises:
    test_tensor_type()
    test_tensor_unranked()
    test_tensor_dynamic_dims()
    test_tensor_dynamic_dims_roundtrip()
    test_tensor_symbolic_dims()
    test_tensor_symbolic_dims_roundtrip()
    test_tensor_scalar()
    test_dtype_autocast()
    test_int_autocast()
    test_tensor_spec_cast()
    test_list_type()
