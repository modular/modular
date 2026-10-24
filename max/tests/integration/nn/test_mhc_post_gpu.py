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
"""Validates the mHC stream write-back op against a float64 reference.

`mo.mhc.post` computes ``post * y + comb^T @ streams`` per token in float32
and rounds once, so it is held to the float64 result rounded once plus the
float32 accumulation slack.
"""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import mhc_post
from torch.utils.dlpack import from_dlpack

_HC_MULT = 4
_HIDDEN = 4096
# 256-thread blocks do not split this one; the kernel takes 128.
_HIDDEN_128 = 5120


def _input_types(
    streams_shape: list[str | int], hidden: int = _HIDDEN
) -> tuple[TensorType, ...]:
    device_ref = DeviceRef.GPU()
    return (
        TensorType(DType.bfloat16, streams_shape, device=device_ref),
        TensorType(DType.bfloat16, ["tokens", hidden], device=device_ref),
        TensorType(DType.float32, ["tokens", _HC_MULT], device=device_ref),
        TensorType(
            DType.float32, ["tokens", _HC_MULT, _HC_MULT], device=device_ref
        ),
    )


@pytest.fixture(scope="module")
def compile_post(gpu_session: InferenceSession) -> Model:
    """Compiles the op once; the symbolic token dim serves every count."""
    with Graph(
        "mhc_post",
        input_types=_input_types(["tokens", _HC_MULT, _HIDDEN]),
    ) as graph:
        graph.output(mhc_post(*(v.tensor for v in graph.inputs)))
    return gpu_session.load(graph)


@pytest.fixture(scope="module")
def compile_post_128(gpu_session: InferenceSession) -> Model:
    with Graph(
        "mhc_post_128",
        input_types=_input_types(
            ["tokens", _HC_MULT, _HIDDEN_128], _HIDDEN_128
        ),
    ) as graph:
        graph.output(mhc_post(*(v.tensor for v in graph.inputs)))
    return gpu_session.load(graph)


@pytest.fixture(scope="module")
def compile_post_broadcast(gpu_session: InferenceSession) -> Model:
    """Feeds the op the model-entry broadcast of one residual to all streams."""
    with Graph(
        "mhc_post_broadcast", input_types=_input_types(["tokens", _HIDDEN])
    ) as graph:
        x, y, post, comb = (v.tensor for v in graph.inputs)
        streams = ops.broadcast_to(
            ops.unsqueeze(x, -2), (x.shape[0], _HC_MULT, _HIDDEN)
        )
        graph.output(mhc_post(streams, y, post, comb))
    return gpu_session.load(graph)


def _inputs(tokens: int, hidden: int = _HIDDEN) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(tokens)
    streams = torch.randn(
        tokens, _HC_MULT, hidden, dtype=torch.bfloat16, device="cuda"
    )
    y = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    post = 2 * torch.rand(tokens, _HC_MULT, device="cuda")
    comb = torch.softmax(
        torch.randn(tokens, _HC_MULT, _HC_MULT, device="cuda"), dim=-2
    )
    return streams, y, post, comb


def _assert_rounded_once(
    actual: torch.Tensor,
    streams: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> None:
    """Within half a bfloat16 ulp of the float64 result, plus fp32 slack."""
    actual, streams, y, post, comb = (
        t.cpu().double() for t in (actual, streams, y, post, comb)
    )
    exact = post.unsqueeze(-1) * y.unsqueeze(-2)
    magnitude = exact.abs()
    for j in range(_HC_MULT):
        term = comb[:, j, :, None] * streams[:, j, None, :]
        exact += term
        magnitude += term.abs()
    tol = 2.0**-8 * exact.abs() + 2.0**-20 * magnitude
    worst = ((actual - exact).abs() / tol).max().item()
    assert worst <= 1.0, f"error reaches {worst:.3f}x the tolerance"


@pytest.mark.parametrize("tokens", [1, 17, 255, 256, 2049])
def test_post_matches_float64(compile_post: Model, tokens: int) -> None:
    # Accesses turn non-temporal at 256 tokens on a 256-CU AMD part (1 per
    # CU) and at 1185 on a 148-SM NVIDIA part (past 8 per SM).
    streams, y, post, comb = _inputs(tokens)
    (out,) = compile_post.execute(streams, y, post, comb)
    actual = from_dlpack(out)

    assert actual.shape == (tokens, _HC_MULT, _HIDDEN)
    assert actual.dtype == torch.bfloat16
    _assert_rounded_once(actual, streams, y, post, comb)


@pytest.mark.parametrize("tokens", [1, 17, 1185, 2049])
def test_post_128_thread_blocks(compile_post_128: Model, tokens: int) -> None:
    streams, y, post, comb = _inputs(tokens, _HIDDEN_128)
    (out,) = compile_post_128.execute(streams, y, post, comb)
    actual = from_dlpack(out)

    assert actual.shape == (tokens, _HC_MULT, _HIDDEN_128)
    _assert_rounded_once(actual, streams, y, post, comb)


def test_post_contracts_over_the_first_stream_axis(
    compile_post: Model,
) -> None:
    """Negative control: the transposed mixer must not also pass."""
    streams, y, post, comb = _inputs(17)
    (out,) = compile_post.execute(streams, y, post, comb)
    with pytest.raises(AssertionError):
        _assert_rounded_once(
            from_dlpack(out),
            streams,
            y,
            post,
            comb.transpose(-1, -2).contiguous(),
        )


@pytest.mark.parametrize("tokens", [1, 17])
def test_post_accepts_broadcast_streams(
    compile_post_broadcast: Model, tokens: int
) -> None:
    """Layer 0's streams are a broadcast of the embeddings."""
    _, y, post, comb = _inputs(tokens)
    x = torch.randn(tokens, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    (out,) = compile_post_broadcast.execute(x, y, post, comb)
    streams = x.unsqueeze(1).expand(tokens, _HC_MULT, _HIDDEN)
    _assert_rounded_once(from_dlpack(out), streams, y, post, comb)


def test_post_rejects_unaligned_hidden_size() -> None:
    device_ref = DeviceRef.GPU()
    bad_hidden = 1536
    with Graph(
        "mhc_post_bad_hidden",
        input_types=(
            TensorType(
                DType.bfloat16, ["t", _HC_MULT, bad_hidden], device=device_ref
            ),
            TensorType(DType.bfloat16, ["t", bad_hidden], device=device_ref),
            TensorType(DType.float32, ["t", _HC_MULT], device=device_ref),
            TensorType(
                DType.float32, ["t", _HC_MULT, _HC_MULT], device=device_ref
            ),
        ),
    ) as graph:
        with pytest.raises(ValueError, match="multiple of 1024"):
            mhc_post(*(v.tensor for v in graph.inputs))


def test_post_rejects_non_power_of_two_hc_mult() -> None:
    device_ref = DeviceRef.GPU()
    hc_mult = 3
    with Graph(
        "mhc_post_bad_hc_mult",
        input_types=(
            TensorType(
                DType.bfloat16, ["t", hc_mult, _HIDDEN], device=device_ref
            ),
            TensorType(DType.bfloat16, ["t", _HIDDEN], device=device_ref),
            TensorType(DType.float32, ["t", hc_mult], device=device_ref),
            TensorType(
                DType.float32, ["t", hc_mult, hc_mult], device=device_ref
            ),
        ),
    ) as graph:
        with pytest.raises(ValueError, match="power of two"):
            mhc_post(*(v.tensor for v in graph.inputs))
