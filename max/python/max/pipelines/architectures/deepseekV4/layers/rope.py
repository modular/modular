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

"""RoPE for DeepSeek-V4.

Two things differ from what ``YarnRotaryEmbedding`` gives out of the box.

**No mscale.** ``YarnRotaryEmbedding.freqs_cis_base`` multiplies cos and sin by
``_yarn_get_mscale(factor, 1.0)``, which for V4's ``factor=16`` is
``0.1*log(16)+1 == 1.2773``. The reference builds its table with
``torch.polar(torch.ones_like(freqs), freqs)`` -- unit magnitude, no mscale at
all. Inheriting the default would scale every rotated value by 28%.

**Two schedules per model, chosen per layer.** ``Attention.__init__``:

    if self.compress_ratio:
        original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
    else:
        # disable YaRN and use base rope_theta in pure sliding-window attention
        original_seq_len, rope_theta = 0, args.rope_theta

and ``precompute_freqs_cis`` skips the whole YaRN block when
``original_seq_len == 0``. So compressed layers rotate at theta 160000 with YaRN
active, and window-only layers rotate at theta 10000 with YaRN off. A layer's
Compressor and Indexer share that layer's table.

Rotation is over *interleaved* pairs (the reference does
``view_as_complex(x.unflatten(-1, (-1, 2)))``), and only over the trailing
``qk_rope_head_dim`` dims of the ``head_dim``-wide latent.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.rotary_embedding import YarnRotaryEmbedding, YarnScalingParams

from ..model_config import DeepseekV4Config


class DeepseekV4RotaryEmbedding(YarnRotaryEmbedding):
    """YaRN RoPE with the mscale factor removed.

    ``scaling_params=None`` gives the plain (YaRN-off) schedule that V4's
    window-only layers use.

    ``YarnRotaryEmbedding`` takes no device and builds its table wherever
    ``_compute_inv_freqs`` lands, which is the host. Every consumer here is on
    the accelerator, so the table is moved once on construction rather than at
    each of the four use sites.
    """

    device: DeviceRef | None = None

    def freqs_cis_base(self) -> TensorValue:
        """``(max_seq_len, head_dim // 2, 2)`` of unscaled cos/sin."""
        if self._freqs_cis is None:
            inv_freqs = (
                self._compute_inv_freqs()
                if self.scaling_params is None
                else self._compute_yarn_freqs()
            )
            t = ops.range(
                0,
                self.max_seq_len,
                1,
                out_dim=self.max_seq_len,
                device=inv_freqs.device,
                dtype=DType.float32,
            )
            freqs = ops.outer(t, inv_freqs)
            table = ops.stack([ops.cos(freqs), ops.sin(freqs)], axis=-1)
            if self.device is not None and table.device != self.device:
                table = ops.transfer_to(table, self.device)
            self._freqs_cis = table
        return TensorValue(self._freqs_cis)


def rope_for_layer(
    config: DeepseekV4Config, layer_idx: int, max_seq_len: int
) -> DeepseekV4RotaryEmbedding:
    """Build the rotary table this layer's attention, compressor and indexer share."""
    compressed = config.layer_compress_ratio(layer_idx) != 0
    scaling = config.rope_scaling or {}
    rope = DeepseekV4RotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=config.compress_rope_theta if compressed else config.rope_theta,
        max_seq_len=max_seq_len,
        head_dim=config.qk_rope_head_dim,
        interleaved=True,
        scaling_params=YarnScalingParams(
            factor=float(scaling["factor"]),
            beta_fast=float(scaling["beta_fast"]),
            beta_slow=float(scaling["beta_slow"]),
            original_max_position_embeddings=int(
                scaling["original_max_position_embeddings"]
            ),
            # Declared on the dataclass but never read by the YaRN frequency
            # path; the reference has no equivalent knob.
            truncate=False,
        )
        if compressed
        else None,
    )
    rope.device = config.devices[0] if config.devices else None
    return rope


def apply_rope_tail(
    x: TensorValue,
    freqs_cis: TensorValue,
    rope_head_dim: int,
    *,
    inverse: bool = False,
) -> TensorValue:
    """Rotate the trailing ``rope_head_dim`` dims of ``x``, leaving the rest alone.

    Args:
        x: ``[..., seq, ..., width]`` with ``width >= rope_head_dim``. The
            sequence axis must be axis 1, matching the reference's
            ``freqs_cis.view(1, x.size(1), ...)`` broadcast.
        freqs_cis: ``[seq, rope_head_dim // 2, 2]`` cos/sin for the positions
            covered by ``x``.
        rope_head_dim: Width of the rotated tail.
        inverse: Rotate by the conjugate, which is what the reference does to
            the attention *output* before the output projection.
    """
    # Positive bounds, deliberately. A negative bound on the last axis can
    # come back rotated when the sliced value is itself the result of a concat
    # -- measured in .agent/backlogs/192/ISSUES.md Issue 34, where
    # ``kv[..., :-rd]`` on the output of ``apply_rope_tail`` returned
    # ``kv[rd:]`` and the latent came out rolled by rd. Every op involved is
    # correct on its own; only the composition is wrong, so the shape of the
    # slice expression is what has to stay positive.
    width = int(x.shape[-1])
    head = x[..., : width - rope_head_dim]
    tail = x[..., width - rope_head_dim :]

    # Interleaved pairs: (even, odd) are the real and imaginary parts.
    pair_shape = list(tail.shape[:-1]) + [rope_head_dim // 2, 2]
    pairs = ops.reshape(tail, pair_shape)
    real = pairs[..., 0]
    imag = pairs[..., 1]

    # freqs_cis is [seq, rope_head_dim // 2, 2]; give it the rank of ``pairs``
    # with the sequence axis at position 1 and everything between broadcast.
    bcast: list[int | Dim] = [1, freqs_cis.shape[0]]
    bcast += [1] * (pairs.rank - 4)
    bcast.append(freqs_cis.shape[1])
    cos = ops.reshape(freqs_cis[..., 0], bcast)
    sin = ops.reshape(freqs_cis[..., 1], bcast)
    if inverse:
        sin = -sin

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    rotated = ops.reshape(ops.stack([out_real, out_imag], axis=-1), tail.shape)
    return ops.concat([head, ops.cast(rotated, tail.dtype)], axis=-1)
