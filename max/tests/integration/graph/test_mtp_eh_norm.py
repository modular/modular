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
"""Graph-level tests for the MTP draft layer's fused input projection.

The fused op is compared against two ``ops.rms_norm`` calls and a
concatenation, in one session and on the same inputs. There is no
hand-written reference: an oracle can agree with the kernel while both
disagree with the graph the draft layer actually builds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import mtp_eh_norm
from torch.utils.dlpack import from_dlpack

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU"
)

EPS = 1e-5


def _build(hidden_size: int, fused: bool, block_threads: int = 256) -> Graph:
    gpu = DeviceRef.GPU()
    t = TensorType(DType.bfloat16, ["tokens", hidden_size], gpu)
    w = TensorType(DType.bfloat16, [hidden_size], gpu)
    with Graph(
        "mtp_eh_norm" if fused else "mtp_eh_norm_ref", input_types=(t, t, w, w)
    ) as g:
        embed, prev, ew, hw = (
            g.inputs[0].tensor,
            g.inputs[1].tensor,
            g.inputs[2].tensor,
            g.inputs[3].tensor,
        )
        if fused:
            out = mtp_eh_norm(embed, prev, ew, hw, EPS, block_threads)
        else:
            # Llama-style rms_norm on each input, then a concatenation.
            out = ops.concat(
                [
                    ops.rms_norm(
                        embed,
                        ew,
                        EPS,
                        weight_offset=0.0,
                        multiply_before_cast=False,
                    ),
                    ops.rms_norm(
                        prev,
                        hw,
                        EPS,
                        weight_offset=0.0,
                        multiply_before_cast=False,
                    ),
                ],
                axis=-1,
            )
        g.output(out)
    return g


def _to_device(arr: np.ndarray, device: Accelerator) -> Buffer:
    # bfloat16 has no numpy dtype; carry the bits through float16.
    return (
        Buffer.from_numpy(arr.view(np.float16)).view(DType.bfloat16).to(device)
    )


def _bf16(rng: np.random.Generator, *shape: int, scale: float) -> np.ndarray:
    x = torch.from_numpy(rng.standard_normal(shape).astype(np.float32) * scale)
    return x.to(torch.bfloat16).view(torch.float16).numpy()


@pytest.mark.parametrize(
    "hidden_size,tokens,block_threads",
    [
        (4096, 5, 256),  # the draft width GLM-5.3-Flash and DeepSeek-V3.2 use
        (384, 3, 256),  # not a multiple of the block: a genuinely ragged pass
        (512, 2, 32),  # one warp: the cross-warp reduction path is skipped
        # Three warps. `block_reduce_dual_sum` reduces over a power-of-two lane
        # group, so a non-power-of-two warp count can drop a warp's sum.
        (384, 2, 96),
        (1024, 2, 160),  # five warps
    ],
)
def test_fused_matches_rms_norm_and_concat(
    hidden_size: int, tokens: int, block_threads: int
) -> None:
    """The fused op must reproduce two rms_norms plus a concat."""
    rng = np.random.default_rng(0)
    device = Accelerator()
    session = InferenceSession(devices=[device])

    fused = session.load(_build(hidden_size, True, block_threads))
    ref = session.load(_build(hidden_size, False))

    # Different scales per half, so reusing one row's sum for both shows up.
    embed = _bf16(rng, tokens, hidden_size, scale=1.0)
    prev = _bf16(rng, tokens, hidden_size, scale=3.0)
    ew = _bf16(rng, hidden_size, scale=1.0)
    hw = _bf16(rng, hidden_size, scale=1.0)

    args = [
        _to_device(embed, device),
        _to_device(prev, device),
        _to_device(ew, device),
        _to_device(hw, device),
    ]
    # Compare raw bf16 bits: the claim is exactness, and a float tolerance
    # would accept a differing low bit.
    got = from_dlpack(fused.execute(*args)[0]).cpu().view(torch.uint16).numpy()
    want = from_dlpack(ref.execute(*args)[0]).cpu().view(torch.uint16).numpy()

    assert got.shape == want.shape == (tokens, 2 * hidden_size)
    mismatches = int(np.count_nonzero(got != want))
    assert mismatches == 0, (
        f"fused output differs from rms_norm+concat in {mismatches} of"
        f" {got.size} elements (hidden_size={hidden_size})"
    )


def test_rejects_mismatched_inputs() -> None:
    """Shape, dtype and device disagreements are caught before the graph."""
    gpu = DeviceRef.GPU()
    with Graph(
        "bad",
        input_types=(
            TensorType(DType.bfloat16, ["t", 128], gpu),
            TensorType(DType.bfloat16, ["t", 64], gpu),
            TensorType(DType.bfloat16, [128], gpu),
        ),
    ) as g:
        a, b, w = g.inputs[0].tensor, g.inputs[1].tensor, g.inputs[2].tensor
        with pytest.raises(ValueError, match="same shape"):
            mtp_eh_norm(a, b, w, w, EPS)
        with pytest.raises(ValueError, match="must equal the hidden size"):
            mtp_eh_norm(a, a, w[:64], w[:64], EPS)
        with pytest.raises(ValueError, match="multiple of 32"):
            mtp_eh_norm(a, a, w, w, EPS, block_threads=100)
        with pytest.raises(ValueError, match="not exceed 1024"):
            mtp_eh_norm(a, a, w, w, EPS, block_threads=2048)
        g.output(a)
