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
projection, the fused gate op, and the stream collapse under each ``pre_mix``
state.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
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

# The `pre_mix` collapse is elementwise over rows, so one row count that is
# not a multiple of the gate kernel's block suffices.
_PRE_MIX_ROWS = 7

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


def _load_hc(
    session: InferenceSession,
    weights: dict[str, torch.Tensor],
    pre_mix: Literal["omitted", "carried", "one_hot"],
) -> Model:
    device_ref = DeviceRef.GPU()
    input_types = [
        TensorType(
            DType.float32,
            ["total_seq_len", _HC_MULT, _HIDDEN_SIZE],
            device=device_ref,
        )
    ]
    if pre_mix == "carried":
        input_types.append(
            TensorType(
                DType.float32, ["total_seq_len", _HC_MULT], device=device_ref
            )
        )
    with Graph(f"hyper_connection_{pre_mix}", input_types=input_types) as graph:
        layer = HyperConnection(
            hidden_size=_HIDDEN_SIZE,
            hc_mult=_HC_MULT,
            device=device_ref,
            hc_sinkhorn_iters=_SINKHORN_ITERS,
            hc_eps=_HC_EPS,
            rms_norm_eps=_RMS_NORM_EPS,
        )
        streams = graph.inputs[0].tensor
        if pre_mix == "omitted":
            outputs = layer(streams)
        elif pre_mix == "carried":
            outputs = layer(streams, graph.inputs[1].tensor)
        else:
            one_hot = np.eye(_HC_MULT, dtype=np.float32)[0]
            outputs = layer(
                streams,
                ops.constant(one_hot, DType.float32, device=device_ref),
            )
        graph.output(*outputs)

    return session.load(graph, weights_registry=weights)


@pytest.fixture(scope="module")
def hc_model(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> Model:
    """One compiled graph for the whole module; the row count stays symbolic."""
    return _load_hc(gpu_session, hc_weights, "omitted")


def _reference(
    streams: torch.Tensor, hc_weights: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return hyper_connection_reference(
        streams,
        hc_weights["hc_fn"].cuda(),
        hc_weights["hc_base"].cuda(),
        hc_weights["hc_scale"].cuda(),
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_SINKHORN_ITERS,
        hc_eps=_HC_EPS,
        rms_norm_eps=_RMS_NORM_EPS,
    )


def _random_streams(rows: int) -> torch.Tensor:
    torch.manual_seed(rows)
    return torch.randn(
        rows, _HC_MULT, _HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )


@pytest.mark.parametrize("rows", _ROWS)
def test_gates_match_torch(
    hc_model: Model, hc_weights: dict[str, torch.Tensor], rows: int
) -> None:
    streams = _random_streams(rows)

    post, comb, collapsed, pre = (
        from_dlpack(t) for t in hc_model.execute(streams)
    )
    ref_post, ref_comb, ref_collapsed, ref_pre = _reference(streams, hc_weights)

    assert tuple(post.shape) == (rows, _HC_MULT)
    assert tuple(comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(collapsed.shape) == (rows, _HIDDEN_SIZE)
    assert tuple(pre.shape) == (rows, _HC_MULT)

    torch.testing.assert_close(post, ref_post, **_GATE_TOL)
    torch.testing.assert_close(comb, ref_comb, **_GATE_TOL)
    torch.testing.assert_close(pre, ref_pre, **_GATE_TOL)
    torch.testing.assert_close(collapsed, ref_collapsed, **_COLLAPSED_TOL)


def test_carried_pre_mix_weights_collapse(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> None:
    model = _load_hc(gpu_session, hc_weights, "carried")
    streams = _random_streams(_PRE_MIX_ROWS)
    pre_mix = torch.rand(
        _PRE_MIX_ROWS, _HC_MULT, dtype=torch.float32, device="cuda"
    )

    _, _, collapsed, pre = (
        from_dlpack(t) for t in model.execute(streams, pre_mix)
    )
    *_, ref_pre = _reference(streams, hc_weights)

    torch.testing.assert_close(pre, ref_pre, **_GATE_TOL)
    torch.testing.assert_close(
        collapsed, (pre_mix.unsqueeze(-1) * streams).sum(dim=1)
    )


def test_one_hot_pre_mix_selects_first_copy(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> None:
    model = _load_hc(gpu_session, hc_weights, "one_hot")
    streams = _random_streams(_PRE_MIX_ROWS)

    _, _, collapsed, _ = (from_dlpack(t) for t in model.execute(streams))

    torch.testing.assert_close(collapsed, streams[:, 0], rtol=0, atol=0)
