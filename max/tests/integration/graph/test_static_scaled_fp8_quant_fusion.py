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
"""Correctness tests for the FP8 quantize fusion consumers.

``quantization.quantize_static_scaled_float8`` takes a ``FusedInputTensor`` so a
single-use elementwise producer (relu, add, cast) fuses INTO the quantize
load lambda instead of materializing a separate HBM tensor + kernel launch.
The quantized math must stay bit-identical to the unfused path: cast to
float32, then ``fp8_quantize(v, 1.0 / scale)`` (saturating cast). These tests
feed a producer op through the quantize and check the round-tripped values
against a NumPy reference, using values that are exactly representable in
``float8_e4m3fn`` so rounding is unambiguous.

For the dynamic-scaled quantize, a single-use ``mo.rms_norm`` producer
fuses into ``mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8``,
which compiles to one rowwise kernel that writes the quantized row and its
per-token scale.
"""

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
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


def test_rms_norm_producer_fuses_into_dynamic_fp8_quant(
    session: InferenceSession,
) -> None:
    """Checks the quantized rows and per-token scales of the fused rms_norm +
    quantize, which runs as one kernel.

    Row 0 is all zeros, so its scale is 0 and its quantized values must be 0,
    not nan. Row 1 has one spike, so its max is over ``scale_ub``: its scale
    is clamped and the spike saturates at the FP8 max.
    """
    rows, hidden, eps, scale_ub, fp8_max = 6, 1024, 1e-6, 10.0, 448.0
    generator = torch.Generator().manual_seed(0)
    x_torch = torch.randn(rows, hidden, generator=generator)
    x_torch[0] = 0.0
    x_torch[1] = 0.01
    x_torch[1, 7] = 100.0
    x_torch = x_torch.bfloat16()
    w_torch = (torch.randn(hidden, generator=generator) * 0.1).bfloat16()

    x_type = TensorType(DType.bfloat16, [rows, hidden], device=DeviceRef.GPU())
    w_type = TensorType(DType.bfloat16, [hidden], device=DeviceRef.GPU())
    with Graph("dynamic_fp8_quant_fusion", input_types=[x_type, w_type]) as g:
        x, w = (v.tensor for v in g.inputs)
        # A single-use rms_norm fuses with the quantize into the composite.
        normed = ops.rms_norm(
            x, w, eps, weight_offset=1.0, multiply_before_cast=True
        )
        quantized, scales = quantize_dynamic_scaled_float8(
            normed,
            InputScaleSpec(
                granularity=ScaleGranularity.ROWWISE,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.bfloat16,
            ),
            WeightScaleSpec(
                granularity=ScaleGranularity.ROWWISE, dtype=DType.bfloat16
            ),
            scale_ub=scale_ub,
        )
        g.output(
            ops.cast(quantized, DType.float32), ops.cast(scales, DType.float32)
        )

    model = session.load(g)
    rowwise = [s for s in model.kernel_summaries if "Rowwise(" in s]
    assert len(rowwise) == 1, model.kernel_summaries
    device = model.input_devices[0]
    q_out, s_out = model(
        Buffer.from_dlpack(x_torch).to(device),
        Buffer.from_dlpack(w_torch).to(device),
    )
    assert isinstance(q_out, Buffer) and isinstance(s_out, Buffer)

    # The normalized row stays in float32; the scale is computed in the
    # scales' bfloat16, as the kernel does.
    normed_ref = torch.nn.functional.rms_norm(
        x_torch.float(), (hidden,), w_torch.float() + 1.0, eps
    )
    amax = normed_ref.abs().amax(-1, keepdim=True).bfloat16()
    scale = torch.minimum(amax, torch.tensor(scale_ub).bfloat16()) / fp8_max
    inv = torch.nan_to_num(1.0 / scale.float(), posinf=0.0)
    q_ref = (normed_ref * inv).clamp(-fp8_max, fp8_max)
    q_ref = q_ref.to(torch.float8_e4m3fn).float()

    # The scales are laid out [1, tokens], transposed from the activations.
    np.testing.assert_allclose(
        s_out.to_numpy(), scale.float().reshape(1, rows).numpy(), rtol=1e-2
    )
    assert s_out.to_numpy()[0, 0] == 0.0
    assert s_out.to_numpy()[0, 1] == pytest.approx(scale_ub / fp8_max, 1e-2)
    # A float32 rounding difference in the normalized value can move a
    # quantized value by one FP8 step (2**-3 relative).
    got = q_out.to_numpy()
    np.testing.assert_allclose(got, q_ref.numpy(), rtol=2**-3, atol=2**-9)
    assert not np.isnan(got).any()
    assert (got[0] == 0.0).all()
    assert got[1, 7] == fp8_max
