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
"""Correctness tests for the static-scaled FP8 quantize fusion consumer.

``mo.quantize_static_scaled_float8`` takes a ``FusedInputTensor`` so a
single-use elementwise producer (relu, add, cast) fuses INTO the quantize
load lambda instead of materializing a separate HBM tensor + kernel launch.
The quantized math must stay bit-identical to the unfused path: cast to
float32, then ``fp8_quantize(v, 1.0 / scale)`` (saturating cast). These tests
feed a producer op through the quantize and check the round-tripped values
against a NumPy reference, using values that are exactly representable in
``float8_e4m3fn`` so rounding is unambiguous.
"""

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import quantize_static_scaled_float8
from test_common.graph_utils import is_b100_b200, is_h100_h200


@pytest.mark.skipif(
    not (is_h100_h200() or is_b100_b200()),
    reason="static-scaled float8 quantize requires an H100/H200/B100/B200 GPU",
)
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_relu_producer_fuses_into_static_fp8_quant(
    session: InferenceSession, scale: float
) -> None:
    """A relu producer fuses into the quantize; result matches the reference.

    The op always divides by ``scale`` (it ignores ``scale_is_inverted``), so
    the expected value is ``relu(x) / scale`` with all inputs chosen so that
    both ``relu(x)`` and ``relu(x) / scale`` are exact in ``float8_e4m3fn``.
    """
    # All values are exact in bfloat16 and stay exact after relu and /scale, so
    # the FP8 round-trip is unambiguous.
    input_torch = torch.tensor(
        [[1.0, -2.0, 4.0, -0.5], [0.5, 8.0, -1.0, 2.0]], dtype=torch.bfloat16
    )
    input_type = TensorType(
        DType.bfloat16, tuple(input_torch.shape), device=DeviceRef.GPU()
    )
    scale_np = np.array(scale, dtype=np.float32)

    with Graph("static_fp8_quant_fusion", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        # Single-use elementwise producer that must fuse into the quant load.
        produced = ops.relu(x)
        scale_const = ops.constant(
            scale_np, dtype=DType.float32, device=DeviceRef.CPU()
        )
        quantized = quantize_static_scaled_float8(produced, scale_const)
        graph.output(ops.cast(quantized, DType.float32))

    model = session.load(graph)
    result = model(Buffer.from_dlpack(input_torch).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)

    input_np = input_torch.to(torch.float32).cpu().numpy()
    expected = np.maximum(input_np, 0.0) / scale
    np.testing.assert_array_equal(result.to_numpy(), expected)
