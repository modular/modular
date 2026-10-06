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
"""Validates ``max.nn.HyperConnection`` against the PyTorch reference.

Covers the whole ModuleV2 layer: the unweighted RMSNorm, the stream
projection, the fused gate op, and the stream collapse.
"""

from __future__ import annotations

import math

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn import HyperConnection
from test_common.hyper_connection_reference import hyper_connection_reference
from torch.utils.dlpack import from_dlpack

_HC_MULT = 4
_SINKHORN_ITERS = 20
_HIDDEN_SIZE = 128
_HC_EPS = 1e-6
_RMS_NORM_EPS = 1e-6
_MIX_WIDTH = 2 * _HC_MULT + _HC_MULT**2

# Row counts that straddle the kernel's four-warp block boundary.
_ROWS = [1, 3, 4, 7, 128, 4096]

# Keep the reference projection at full float32 so the comparison is against
# the exact layer, not against torch's own reduced-precision matmul.
torch.set_float32_matmul_precision("highest")

# MAX runs the float32 stream projection at TF32 precision, which lands ~2e-3
# of absolute error in `hc_proj` and carries it through the sigmoids and the
# Sinkhorn projection, so the projection -- not the gate math -- sets what this
# test can resolve. These bounds are measured with ~2x headroom at the largest
# row count.
_GATE_TOL = {"rtol": 5e-3, "atol": 2e-3}
_COLLAPSED_TOL = {"rtol": 5e-3, "atol": 5e-3}


@pytest.fixture(scope="module")
def hc_weights() -> dict[str, torch.Tensor]:
    """The layer's three weights, on CPU for the weights registry."""
    torch.manual_seed(0)
    fan_in = _HC_MULT * _HIDDEN_SIZE
    return {
        # Scaled by 1/sqrt(fan_in) so the projection of an RMS-normalized
        # stream lands in the sigmoid's and softmax's responsive range.
        "hc_fn": torch.randn(_MIX_WIDTH, fan_in, dtype=torch.float32)
        / math.sqrt(fan_in),
        "hc_base": torch.randn(_MIX_WIDTH, dtype=torch.float32),
        "hc_scale": 2 * torch.rand(3, dtype=torch.float32),
    }


@pytest.fixture(scope="module")
def hc_model(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> Model:
    """One compiled graph for the whole module; the row count stays symbolic."""
    device_ref = DeviceRef.GPU()
    with Graph(
        "hyper_connection",
        input_types=(
            TensorType(
                DType.float32,
                ["total_seq_len", _HC_MULT, _HIDDEN_SIZE],
                device=device_ref,
            ),
        ),
    ) as graph:
        layer = HyperConnection(
            hidden_size=_HIDDEN_SIZE,
            hc_mult=_HC_MULT,
            device=device_ref,
            hc_sinkhorn_iters=_SINKHORN_ITERS,
            hc_eps=_HC_EPS,
            rms_norm_eps=_RMS_NORM_EPS,
        )
        graph.output(*layer(graph.inputs[0].tensor))

    return gpu_session.load(graph, weights_registry=hc_weights)


@pytest.mark.parametrize("rows", _ROWS)
def test_gates_match_torch(
    hc_model: Model, hc_weights: dict[str, torch.Tensor], rows: int
) -> None:
    torch.manual_seed(rows)
    streams = torch.randn(
        rows, _HC_MULT, _HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )

    post, comb, collapsed = (from_dlpack(t) for t in hc_model.execute(streams))

    ref_post, ref_comb, ref_collapsed = hyper_connection_reference(
        streams,
        hc_weights["hc_fn"].cuda(),
        hc_weights["hc_base"].cuda(),
        hc_weights["hc_scale"].cuda(),
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_SINKHORN_ITERS,
        hc_eps=_HC_EPS,
        rms_norm_eps=_RMS_NORM_EPS,
    )

    assert tuple(post.shape) == (rows, _HC_MULT)
    assert tuple(comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(collapsed.shape) == (rows, _HIDDEN_SIZE)

    torch.testing.assert_close(post, ref_post, **_GATE_TOL)
    torch.testing.assert_close(comb, ref_comb, **_GATE_TOL)
    torch.testing.assert_close(collapsed, ref_collapsed, **_COLLAPSED_TOL)
