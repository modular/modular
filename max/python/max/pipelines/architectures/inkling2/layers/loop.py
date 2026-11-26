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
"""The looped checkpoint's message bank and the controllers that read it.

A looped checkpoint runs its core layers several times per token, and every
pass restarts from the prelude's blocks and partial sum: nothing but this bank
crosses a pass boundary. Each controller replaces one of a layer's inputs with
a gated mix of that input and a message the bank mixes out of the previous
pass, which is how the second pass gets to see what the first one computed.

The checkpoint writes the controllers in its own order, ``attn``, ``k``, ``v``,
``mlp``, and each module's ``down_weight``, ``gain_weight`` and
``forget_weight`` load under that name.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.tensor import Tensor

from .attn_res import AttnResSources, mix_attn_res, prepare_attn_res
from .norm import unweighted_rms_norm

CONTROLLER_EPS = 1e-6
"""The controllers' own RMS normalization epsilon, not the checkpoint's."""


class Inkling2MessageController(Module[[Tensor, Tensor], Tensor]):
    """Gates one layer input against one message from the previous pass.

    The gates are ``tanh``, so the destination gate ``1 - tanh(f)`` reaches 2
    and the mix is not a convex combination.
    """

    def __init__(self, width: int, rank: int) -> None:
        self.width = width
        self.down_weight = Tensor.zeros([rank, 2 * width], dtype=DType.bfloat16)
        self.gain_weight = Tensor.zeros([width, rank], dtype=DType.bfloat16)
        self.forget_weight = Tensor.zeros([width, rank], dtype=DType.bfloat16)

    def forward(self, destination: Tensor, message: Tensor) -> Tensor:
        """Returns the mixed stream, in float32."""
        dest = destination.cast(DType.float32)
        msg = message.cast(DType.float32)
        # The two halves are normalized separately and concatenated destination
        # first: the projection is trained on that layout.
        pair = F.concat(
            [
                unweighted_rms_norm(dest, CONTROLLER_EPS).cast(DType.bfloat16),
                unweighted_rms_norm(msg, CONTROLLER_EPS).cast(DType.bfloat16),
            ],
            axis=-1,
        )
        h = (pair @ self.down_weight.T).cast(DType.bfloat16)
        # The compiler merges parallel matmuls only in their untransposed form,
        # and `MatmulTranspBPattern` turns a constant `w.T` into `transpose_b`,
        # so the two gates are stacked here rather than left to that merge.
        # `out_dtype` keeps the float32 accumulator the reference computes them
        # in.
        gate_weights = F.concat(
            [self.gain_weight, self.forget_weight], axis=0
        ).T
        gates = F.tanh(F.matmul(h, gate_weights, out_dtype=DType.float32))
        gain, forget = gates.split([self.width, self.width], axis=-1)
        return (1 - forget) * dest + gain * msg


class Inkling2MessageControllers(Module[[], None]):
    """One core layer's four controllers.

    ``attn``, ``k`` and ``v`` mix the attention input streams and ``mlp`` the
    MLP input. A local layer has a single attention stream, so all three
    attention-side controllers read it and produce three different mixed
    streams.
    """

    def __init__(self, width: int, rank: int) -> None:
        self.attn = Inkling2MessageController(width, rank)
        self.k = Inkling2MessageController(width, rank)
        self.v = Inkling2MessageController(width, rank)
        self.mlp = Inkling2MessageController(width, rank)


def prepare_bank_messages(sources: Sequence[Tensor]) -> AttnResSources:
    """Normalizes the bank once for every core layer of a pass.

    The bank is one tensor set per pass while every core layer mixes a
    different query out of it, so the normalization belongs to the pass, not
    to the layer.

    Args:
        sources: The stored blocks the core does not own, then the previous
            pass's partial sum.
    """
    return prepare_attn_res(sources, normalize_values=True)


def bank_messages(prepared: AttnResSources, queries: Tensor) -> list[Tensor]:
    """The messages one core layer reads from the previous pass.

    Args:
        prepared: The pass's bank, from :func:`prepare_bank_messages`.
        queries: ``[4, hidden]`` float32, this layer's rows of
            ``scaled_loop_queries``. They already carry ``hidden ** -0.5``, so
            the mix must not apply it a second time.

    Returns:
        One float32 mixture per controller row.
    """
    return mix_attn_res(prepared, queries, queries_are_scaled=True)


def scaled_loop_queries(
    queries: Tensor, kv_queries: Tensor, width: int
) -> Tensor:
    """Interleaves the checkpoint's two query tensors into the bank's rows.

    Both ship as ``[2 * num_core, hidden]``: one row per pass pair. The bank
    wants ``[num_core, 4, hidden]`` in controller order, so the attention query
    and the key/value queries interleave as ``q[0::2]``, ``kv[0::2]``,
    ``kv[1::2]``, ``q[1::2]`` and the whole thing is scaled once by
    ``hidden ** -0.5``. This is that one application: ``bank_messages`` mixes
    with ``queries_are_scaled=True`` so the scale does not compound.
    """
    core = int(queries.shape[0]) // 2
    q = queries.reshape([core, 2, width])
    kv = kv_queries.reshape([core, 2, width])
    stacked = F.stack(
        [q[:, 0], kv[:, 0], kv[:, 1], q[:, 1]],
        axis=1,
    )
    return stacked.cast(DType.float32) * (width**-0.5)
