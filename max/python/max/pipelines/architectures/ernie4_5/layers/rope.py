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

"""ERNIE-4.5 RoPE: builds correct freqs_cis for GPT-J style rotation."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def build_ernie45_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
) -> TensorValue:
    """Build ERNIE-4.5's interleaved GPT-J-style ``freqs_cis`` tensor.

    The returned tensor has layout ``[max_seq_len, head_dim]`` where each
    head dim is stored as interleaved cosine/sine pairs:
    ``[cos0,sin0,cos1,sin1,...]`` for every position.

    Args:
        head_dim: Dimensionality per head (typically 128).
        max_seq_len: Maximum sequence length to build positions for.
        rope_theta: RoPE base scaling factor.

    Returns:
        A ``TensorValue`` containing the interleaved freqs_cis tensor.
    """

    # [head_dim//2] — inverse frequencies
    indices = ops.range(
        0, head_dim, step=2, dtype=DType.float64, device=DeviceRef.CPU()
    )
    inv_freqs = ops.cast(
        1.0 / (rope_theta ** (indices / head_dim)), DType.float32
    )

    # [max_seq_len] — position ids
    t = ops.range(0, max_seq_len, dtype=DType.float32, device=DeviceRef.CPU())

    # [max_seq_len, head_dim//2]
    sinusoid = ops.outer(t, inv_freqs)

    # [max_seq_len, head_dim//2, 2] → [max_seq_len, head_dim]
    freqs_cis = ops.stack([ops.cos(sinusoid), ops.sin(sinusoid)], axis=-1)
    s0, s1, s2 = freqs_cis.shape
    return ops.reshape(freqs_cis, [s0, s1 * s2])
