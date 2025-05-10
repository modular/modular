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
# RUN: %mojo -debug-level full %s

from collections.optional import Optional

from max.engine import InferenceSession
from max.tensor import TensorSpec
from testing import assert_equal, assert_false, assert_true


fn test_tensor_spec_basic() raises:
    var session = InferenceSession()
    var spec = TensorSpec(DType.float32, 1, 2, 3)
    var engine_spec = session.get_as_engine_tensor_spec("tensor", spec)

    assert_equal(engine_spec[0].value(), 1)

    assert_equal(engine_spec.get_name(), "tensor")

    assert_equal(String(engine_spec), "{name=tensor, spec=1x2x3xfloat32}")

    assert_equal(engine_spec.get_as_tensor_spec(), spec)

    var dynamic_dim_shape = List[Optional[Int64]]()
    dynamic_dim_shape.append(None)
    dynamic_dim_shape.append(Int64(1))
    dynamic_dim_shape.append(Int64(2))
    var dynamic_dim_spec = session.get_as_engine_tensor_spec(
        "tensor", dynamic_dim_shape, DType.float32
    )

    assert_false(dynamic_dim_spec[0])

    assert_equal(dynamic_dim_spec.rank().value(), 3)

    assert_true(dynamic_dim_spec.has_rank())

    assert_equal(String(dynamic_dim_spec), "{name=tensor, spec=-1x1x2xfloat32}")

    var dynamic_rank_spec = session.get_as_engine_tensor_spec(
        "tensor", None, DType.float32
    )

    assert_false(dynamic_rank_spec.rank())

    assert_false(dynamic_rank_spec.has_rank())

    assert_equal(
        String(dynamic_rank_spec), "{name=tensor, spec=None x float32}"
    )


fn test_engine_tensor_spec_static_dim_copy() raises:
    var session = InferenceSession()
    var spec = TensorSpec(DType.float32, 1, 2, 3)
    var static_engine_spec = session.get_as_engine_tensor_spec("tensor", spec)

    var static_engine_spec_copy = static_engine_spec

    assert_equal(
        static_engine_spec_copy[0].value(),
        static_engine_spec[0].value(),
    )

    assert_equal(
        static_engine_spec_copy.get_name(), static_engine_spec.get_name()
    )

    assert_equal(
        String(static_engine_spec_copy), "{name=tensor, spec=1x2x3xfloat32}"
    )

    assert_equal(static_engine_spec_copy.get_as_tensor_spec(), spec)


fn test_engine_tensor_spec_dynamic_dim_copy() raises:
    var session = InferenceSession()
    var dynamic_dim_shape = List[Optional[Int64]]()
    dynamic_dim_shape.append(None)
    dynamic_dim_shape.append(Int64(1))
    dynamic_dim_shape.append(Int64(2))
    var dynamic_dim_spec = session.get_as_engine_tensor_spec(
        "tensor", dynamic_dim_shape, DType.float32
    )

    var dynamic_dim_shape_spec_copy = dynamic_dim_spec

    assert_equal(
        dynamic_dim_shape_spec_copy[0].__bool__(),
        dynamic_dim_spec[0].__bool__(),
    )

    assert_equal(
        dynamic_dim_shape_spec_copy.rank().value(),
        dynamic_dim_spec.rank().value(),
    )

    assert_equal(
        dynamic_dim_shape_spec_copy.has_rank(), dynamic_dim_spec.has_rank()
    )

    assert_equal(
        String(dynamic_dim_shape_spec_copy),
        "{name=tensor, spec=-1x1x2xfloat32}",
    )

    var dynamic_rank_spec = session.get_as_engine_tensor_spec(
        "tensor", None, DType.float32
    )
    var dynamic_rank_spec_copy = dynamic_rank_spec

    assert_false(dynamic_rank_spec_copy.rank())

    assert_false(dynamic_rank_spec_copy.has_rank())

    assert_equal(
        String(dynamic_rank_spec_copy), "{name=tensor, spec=None x float32}"
    )


fn main() raises:
    test_tensor_spec_basic()
    test_engine_tensor_spec_static_dim_copy()
    test_engine_tensor_spec_dynamic_dim_copy()
