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
"""``mo.rms_norm_quantize_dynamic_scaled_float8`` against the unfused pair.

The fused op must produce the same FP8 activation and the same
``[K // group, M_padded]`` group scales as ``ops.rms_norm`` followed by
``quantize_dynamic_scaled_float8`` with a blockwise input spec -- the two
ops GLM-5.3 / DeepSeek-V3.2 ran before the q_a norm was fused. Both paths
run in one graph on the same inputs; they may differ only where the
sum-of-squares reduction order flips a bf16 rounding (at most a 0.1% FP8
mismatch rate, one FP8 ulp).
"""

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    quantize_dynamic_scaled_float8,
    rms_norm_quantize_dynamic_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_GROUP = 128
_EPS = 1e-6
_INPUT_SPEC = InputScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    origin=ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, _GROUP),
)
_WEIGHT_SPEC = WeightScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(_GROUP, _GROUP),
)


@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("hidden", [2048, 6144])
@pytest.mark.parametrize("multiply_before_cast", [False, True])
def test_fused_matches_rms_norm_then_grouped_quantize(
    session: InferenceSession,
    rows: int,
    hidden: int,
    multiply_before_cast: bool,
) -> None:
    generator = torch.Generator().manual_seed(0)
    # Per-row gain and a per-group outlier so every group scale is distinct.
    gain = torch.arange(1, rows + 1).reshape(rows, 1).float()
    x_torch = torch.randn(rows, hidden, generator=generator) * gain
    x_torch[:, ::_GROUP] *= 9.0
    x_torch = x_torch.bfloat16()
    w_torch = (torch.randn(hidden, generator=generator) * 0.5 + 1.0).bfloat16()

    x_type = TensorType(DType.bfloat16, [rows, hidden], device=DeviceRef.GPU())
    w_type = TensorType(DType.bfloat16, [hidden], device=DeviceRef.GPU())
    with Graph(
        "rms_norm_quantize_fp8_grouped", input_types=[x_type, w_type]
    ) as g:
        x, w = (v.tensor for v in g.inputs)
        normed = ops.rms_norm(
            x, w, _EPS, multiply_before_cast=multiply_before_cast
        )
        q_sep, s_sep = quantize_dynamic_scaled_float8(
            normed,
            _INPUT_SPEC,
            _WEIGHT_SPEC,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float32,
        )
        q_fus, s_fus = rms_norm_quantize_dynamic_scaled_float8(
            x,
            w,
            _EPS,
            _INPUT_SPEC,
            _WEIGHT_SPEC,
            multiply_before_cast=multiply_before_cast,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float32,
        )
        assert q_fus.type == q_sep.type
        assert s_fus.type == s_sep.type
        g.output(
            ops.cast(q_sep, DType.float32),
            s_sep,
            ops.cast(q_fus, DType.float32),
            s_fus,
        )

    model = session.load(g)
    device = model.input_devices[0]
    outputs = model(
        Buffer.from_dlpack(x_torch).to(device),
        Buffer.from_dlpack(w_torch).to(device),
    )
    q_sep_np, s_sep_np, q_fus_np, s_fus_np = (
        o.to_numpy() for o in outputs if isinstance(o, Buffer)
    )

    # Same scale layout, and the padded scale columns never matter: compare
    # only the real rows.
    assert (
        s_fus_np.shape
        == s_sep_np.shape
        == (hidden // _GROUP, -(-rows // 4) * 4)
    )
    np.testing.assert_allclose(
        s_fus_np[:, :rows], s_sep_np[:, :rows], rtol=1e-2, atol=0.0
    )

    mismatch = q_fus_np != q_sep_np
    assert mismatch.mean() <= 1e-3, f"{mismatch.sum()} FP8 mismatches"

    # Both dequantized activations track the f32 reference of the norm.
    normed_ref = torch.nn.functional.rms_norm(
        x_torch.float(), (hidden,), w_torch.float(), _EPS
    ).numpy()
    for q_np, s_np in ((q_sep_np, s_sep_np), (q_fus_np, s_fus_np)):
        scales = np.repeat(s_np[:, :rows].T, _GROUP, axis=1)
        deq = q_np * scales
        cos = (deq * normed_ref).sum() / np.sqrt(
            (deq * deq).sum() * (normed_ref * normed_ref).sum()
        )
        assert cos >= 0.999, f"cosine {cos} vs the f32 rms_norm reference"
