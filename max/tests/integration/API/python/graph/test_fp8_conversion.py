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
"""Test FP8 E4M3FN to E4M3FNUZ conversion operations."""

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Weight, ops
from max.nn.kernels import (
    needs_fp8_fnuz_conversion,
    normalize_e4m3fn_to_e4m3fnuz,
)


@pytest.mark.skipif(
    not needs_fp8_fnuz_conversion(),
    reason="Only for AMD GPUs with FP8 FNUZ dtype",
)
def test_normalize_e4m3fn_to_e4m3fnuz(session: InferenceSession) -> None:
    """Test FP8 E4M3FN to E4M3FNUZ conversion doubles the scale factor."""

    negative_zero = np.float32(-0.0)
    assert np.signbit(negative_zero) and negative_zero == 0.0
    weight_data = np.array(
        [[1.0, 2.0, -1.0, 0.0, negative_zero]], dtype=np.float32
    )
    scale_data = np.array([1.0], dtype=np.float32)

    with Graph("fp8_conversion", input_types=[]) as graph:
        weight_w = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[1, 5],
            device=DeviceRef.GPU(),
        )

        weight_f32 = graph.add_weight(weight_w)
        weight_e4m3fn = weight_f32.cast(DType.float8_e4m3fn)

        scale = ops.constant(
            scale_data, dtype=DType.float32, device=DeviceRef.CPU()
        )

        normalized_weight, adjusted_scale = normalize_e4m3fn_to_e4m3fnuz(
            weight_e4m3fn, scale
        )
        graph.output(
            ops.cast(normalized_weight, DType.float32),
            ops.cast(adjusted_scale, DType.float32),
        )

    weights_registry = {
        "test_weight": weight_data,
    }

    model = session.load(graph, weights_registry=weights_registry)
    result = model.execute()

    weight_out = result[0]
    scale_out = result[1]
    assert isinstance(weight_out, Tensor)
    assert isinstance(scale_out, Tensor)
    weight_out_np = weight_out.to_numpy()

    np.testing.assert_allclose(
        weight_out_np[:-1] * 2, weight_data[:-1], rtol=1e-6
    )
    zeroed_value = weight_out_np[0, -1].item()
    assert zeroed_value == 0.0 and not np.signbit(zeroed_value)

    np.testing.assert_allclose(
        scale_out.to_numpy(), scale_data * 2.0, rtol=1e-6
    )
