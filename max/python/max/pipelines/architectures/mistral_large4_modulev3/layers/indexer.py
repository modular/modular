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
"""Mistral Large 4's DSA lightning indexer.

The reference indexer derives its query straight from the post-attention-norm
hidden state, because Mistral Large 4 has no query LoRA to tap::

    q = LayerNorm(6144, bias, eps 1e-6)(x)  ->  wq_b (FP8 block)  ->  SSMax
      -> interleaved rope on dims [0:64]  -> FP8 quantize

and its keys from the same hidden state through a BF16 ``wk`` and a
``LayerNorm(128, bias, eps 1e-6)``. The quant asymmetry is deliberate: ``wq_b``
ships FP8 block-scaled, ``wk`` and ``weights_proj`` ship BF16.

Two layout facts differ from the DeepSeek-V3.2 indexer this forks:

* The query and key lay their head out ``[rope | nope]``, where the MLA lays
  its own out ``[nope | rope]``. Splitting the 128-wide head in half, as this
  does, is therefore the rope slice first.
* ``dsa.match_nope`` makes the indexer use rope exactly on the layers that use
  rope, so a NoPE layer's indexer rotates nothing and keeps its 128-wide
  query and key whole.

Scoring is the reference's: a float32 dot scaled by ``head_dim ** -0.5``, ReLU
(no softmax), then a learned per-head weight scaled by ``heads ** -0.5``. The
FP8 index cache and its per-token float32 scales are the DeepSeek-V3.2 layout.

The query's SSMax, rope and FP8 quantize run as one launch; so does the whole key
chain — the LayerNorm, the rope, the FP8 quantize and both cache stores — which
rounds exactly where the separate ops do. The key chain reads its 128 columns out
of the merged latent-and-key projection where the GEMM wrote them, so the key's
slice is never copied out into a launch of its own.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn._graph_ops import MIXED_GEMV_EPILOGUE_MAX_M
from max.nn.common_layers import quant_ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.common_layers.quant_tensor import FP8BlockTensor
from max.nn.kernels import (
    indexer_k_quantize_store,
    indexer_q_ssmax_rope_quantize,
    mixed_gemv_epilogue,
    mla_fp8_index_top_k_physical,
    row_invariant_matmul,
)
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.quant_config import QuantConfig
from max.nn.norm import LayerNorm
from max.tensor import Tensor

from ._dispatch import fused_fp32_gemv_applies
from .ssmax import SSMaxScaling


def _head_weights(x: Tensor, proj: Tensor) -> Tensor:
    """Returns the ``[rows, heads]`` head weights, in float32.

    The reference projects them in float32 on both operands, which on SM100 is
    one launch past a widening of the whole 6144-wide activation. The fused
    form folds that widening into the projection and keeps the epilogue the
    projection carries -- the two scale multiplies are still applied where they
    are, by the launch that produces the weights -- so the site is one launch
    instead of two. Its arithmetic is today's: the same split-K reduction, over
    exactly the widened BF16 values.

    The upcast weight is a legal ``b`` because it is execution-invariant and so
    hoisted to model init: the fused kernel's PDL peel reads the weight before
    it waits on the producing grid, which a weight cast per launch could not
    satisfy. The launch count test's ``_widened(folded) == 0`` is what pins
    that hoist.


    Args:
        x: ``(seq_len, hidden)`` BF16 hidden state.
        proj: The ``(heads, hidden)`` head-weight projection.
    """
    if fused_fp32_gemv_applies(x, proj):
        return mixed_gemv_epilogue(
            x, F.cast(proj, DType.float32), max_m=MIXED_GEMV_EPILOGUE_MAX_M
        )
    return F.cast(x, DType.float32) @ F.cast(proj, DType.float32).T


class MistralLarge4Indexer(Module[..., Tensor]):
    """Selects the top-k keys that sparse MLA attends to."""

    q_input_norm: LayerNorm
    """LayerNorm over the hidden state, with a bias; the query's only input
    normalization. ``q_input_norm`` in the checkpoint."""
    wq_b: QuantizedLinear
    """``q_input_norm(x)`` to ``heads * head_dim``; FP8 block-scaled."""
    wk: QuantizedLinear
    """The hidden state to one 128-wide key; BF16."""
    k_norm: LayerNorm
    """LayerNorm over the key, with a bias."""
    weights_proj: QuantizedLinear
    """The hidden state to one weight per head; BF16."""
    ssmax_scaling: SSMaxScaling
    """Per-(token, head) query scale applied before the query's rope."""

    def __init__(
        self,
        *,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        quant_config: QuantConfig,
        norm_eps: float = 1e-6,
        rope_interleaved: bool = True,
        row_invariant_decode: bool = False,
    ) -> None:
        """Builds the indexer's projections and norms.

        Args:
            hidden_size: Width of the hidden state it reads.
            n_heads: Index heads (32).
            head_dim: Width of an index head (128).
            rope_head_dim: The indexer's rope width; 0 on a NoPE layer.
            index_topk: Keys the selection keeps (2048).
            quant_config: The FP8 block config ``wq_b`` loads with.
            norm_eps: Epsilon of both LayerNorms.
            rope_interleaved: Whether the indexer's rope is GPT-J interleaved.
            row_invariant_decode: Whether the BF16 key projection runs the
                row-invariant matmul, so a decode step's key matches the
                DSpark verify's bit for bit.

        Raises:
            ValueError: If the rope width is neither 0 nor half of ``head_dim``,
                or a rope layer's rope is not interleaved.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.rope_interleaved = rope_interleaved
        self.row_invariant_decode = row_invariant_decode
        # The rotation covers the leading half of each head, so a non-zero rope
        # width must be exactly half of the head.
        if rope_head_dim != 0 and rope_head_dim * 2 != head_dim:
            raise ValueError(
                "indexer rope width must be 0 or half of head_dim; got"
                f" rope_head_dim={rope_head_dim} with head_dim={head_dim}"
            )
        # The fused query op rotates with interleaved RoPE only, where the key
        # path below honors the flag; a non-interleaved indexer would silently
        # desync the two.
        if rope_head_dim != 0 and not rope_interleaved:
            raise ValueError(
                "the fused indexer query op rotates interleaved RoPE only, so"
                " a rope layer needs rope_interleaved=True"
            )
        self.index_topk = index_topk
        self.softmax_scale = head_dim**-0.5
        # The query projection alone is FP8; wk and weights_proj stay BF16 in
        # every checkpoint of this family, including the FP8 one.
        self.q_input_norm = LayerNorm(
            dim=hidden_size, eps=norm_eps, keep_dtype=False
        )
        self.wq_b = QuantizedLinear(
            in_dim=hidden_size,
            out_dim=n_heads * head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.wk = QuantizedLinear(
            in_dim=hidden_size, out_dim=head_dim, bias=False
        )
        self.k_norm = LayerNorm(dim=head_dim, eps=norm_eps, keep_dtype=False)
        self.weights_proj = QuantizedLinear(
            in_dim=hidden_size, out_dim=n_heads, bias=False
        )
        self.ssmax_scaling = SSMaxScaling(n_heads)

    def _project_query(self, x: Tensor) -> Tensor:
        """Projects the hidden state to the index query.

        The query norm's only reader is ``wq_b``'s activation quantize, so the
        biased float32 LayerNorm and that quantize run as one launch
        (:func:`~max.nn.common_layers.quant_ops.layer_norm_quantize_fp8_block`),
        which rounds exactly where the separate ops do. A BF16 ``wq_b`` needs no
        quantize, so the norm stays a launch of its own there.

        Args:
            x: ``[tokens, hidden]`` the block's input norm produced.

        Returns:
            ``[tokens, heads * head_dim]`` BF16 query projection.
        """
        weight = self.wq_b.weight
        gamma, beta = self.q_input_norm.weight, self.q_input_norm.bias
        if (
            isinstance(weight, FP8BlockTensor)
            and not self.q_input_norm.keep_dtype
            and gamma is not None
            and beta is not None
        ):
            return quant_ops.matmul(
                quant_ops.layer_norm_quantize_fp8_block(
                    x, weight, gamma, beta, self.q_input_norm.eps
                ),
                weight,
            )
        return quant_ops.matmul(self.q_input_norm(x), weight)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        indexer_k_collection: PagedCacheValues,
        layer_idx: Tensor,
        mla_collection: PagedCacheValues,
        quant_config: QuantConfig,
        mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
        key_projection: Tensor | None = None,
        key_col_offset: int = 0,
    ) -> Tensor:
        """Returns the physical MLA cache rows of each token's top-k keys.

        Args:
            x: ``[tokens, hidden]`` the block's input norm produced.
            positions: ``[tokens]`` absolute 0-based positions, for SSMax.
            freqs_cis: The rotary table.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            indexer_k_collection: The indexer's own FP8 key cache.
            layer_idx: This layer's slot in the indexer cache.
            mla_collection: The MLA cache the selected rows index.
            quant_config: The model's FP8 block config, for the activation
                quantization the scorer takes.
            mask_variant: The causal mask the selection applies.
            key_projection: The merged latent-and-key projection when the
                caller already launched it, with the key's columns last;
                ``None`` projects the key here.
            key_col_offset: The column ``key_projection``'s key starts at,
                which its caller knows as the width of the latent columns it
                concatenated before the key. The fused key op reads its 128
                channels there, so the projection is not copied out first. A
                packed key starts at 0, and the op refuses a projection whose
                width disagrees with this rather than absorbing it.

        Returns:
            ``[tokens, index_topk]`` physical rows, ``-1`` past a token's
            candidates.
        """
        # The projection's zero bias is folded away in the graph, negative
        # zeros included, and the fused launch applies no bias.
        q = self._project_query(x)
        q = q.reshape((-1, self.n_heads, self.head_dim))
        # SSMax goes on before the rope and before the FP8 quantize, all in
        # one launch that rounds where the separate ops would.
        q_fp8, q_scale = indexer_q_ssmax_rope_quantize(
            q,
            positions,
            self.ssmax_scaling.s,
            self.ssmax_scaling.b,
            freqs_cis,
            input_row_offsets,
            indexer_k_collection.cache_lengths,
            rope_dim=self.rope_head_dim,
        )
        q_fp8 = q_fp8.rebind_mapping(q.mapping)
        q_scale = q_scale.rebind_mapping(q.mapping).reshape(
            (-1, self.n_heads, 1)
        )

        k_weight = self.k_norm.weight
        k_bias = self.k_norm.bias
        assert k_weight is not None and k_bias is not None
        # The merged projection lays the latent's columns before the key's, so
        # the fused op is told where the key starts and reads its 128 channels
        # in place; slicing them here would materialise them in a copy launch
        # of their own. The offset comes from the caller rather than from this
        # operand's width, so that a width slip fails the op's own check
        # instead of being absorbed into the offset. A key this module
        # projects itself is packed, at offset 0.
        offset = key_col_offset if key_projection is not None else 0
        if key_projection is not None:
            k = key_projection
        elif self.row_invariant_decode:
            wk = self.wk.weight
            assert isinstance(wk, Tensor)
            k = row_invariant_matmul(x, wk)
        else:
            k = self.wk(x)

        # The key's LayerNorm, rope, FP8 quantize and both cache stores are one
        # launch that rounds where the separate ops do.
        indexer_k_quantize_store(
            indexer_k_collection,
            k,
            k_weight,
            k_bias,
            self.k_norm.eps,
            freqs_cis,
            input_row_offsets,
            indexer_k_collection.cache_lengths,
            layer_idx,
            quantization_granularity=quant_config.scales_granularity_mnk[2],
            rope_dim=self.rope_head_dim,
            key_col_offset=offset,
        )

        # The head weights are projected in float32, as the reference does.
        proj = self.weights_proj.weight
        assert isinstance(proj, Tensor)
        weights = _head_weights(x, proj) * self.n_heads**-0.5
        weights = F.unsqueeze(weights, axis=-1) * q_scale * self.softmax_scale

        return mla_fp8_index_top_k_physical(
            q_fp8,
            F.squeeze(weights, axis=-1),
            input_row_offsets,
            indexer_k_collection,
            mla_collection,
            layer_idx,
            self.index_topk,
            quant_config.scales_granularity_mnk[2],
            mask_variant,
            # The sparse MLA sums over the selection in its column order, so
            # an arrival-ordered selection makes greedy decode vary run to run.
            deterministic=True,
        ).rebind_mapping(q_fp8.mapping)
