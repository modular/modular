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
"""The Mistral Large 4 DSpark drafter (ModuleV3), in the target checkpoint.

The drafter is ``mtp_dspark.*`` in the same checkpoint as the target: three
dense MLA layers, a tap projection, and a gated recurrent head. It owns no
token embedding and no LM head -- both are the target's and arrive as call
arguments -- and its latent cache is its own: 576 columns per token per layer,
``[content 512 | rope key 64]``, in the activation dtype. The target shards both
borrowed tables over its vocabulary, so on more than one rank the drafter cannot
look a token up itself: the caller embeds the block's anchors through the
target's embedding module and enters :meth:`DSparkDrafter.block_from_anchors`,
and :meth:`DSparkDrafter.base_logits` gathers the ranks' vocabulary columns of
the head's output back into a whole logit vector before the recurrent chain
reads it.

:meth:`DSparkDrafter.forward` is one whole draft step for a batch of anchors
over one shared context: it builds the context latents, runs the block over the
window, scores it, and unrolls the recurrent chain. That is the shape the
parity gate compiles and the shape a caller holding a whole prompt's taps runs.
It returns the draft tokens and their logits; the debug intermediates only
come back when the drafter is built for them (:meth:`DSparkDrafter.__init__`'s
``emit_intermediates``), which a serving graph never does.
The block driver's phases (:meth:`project_context`, :meth:`context_latents`,
:meth:`embed_block`, :meth:`forward_block`, :meth:`base_logits`,
:meth:`draft`) are the same code split at the boundaries its ``materialize`` /
``embed_block`` / ``forward_block`` / ``head`` calls name, so the cached path
and the dense path cannot drift.

The reference is Mistral's DSpark write-up, which this module transcribes
section by section.

Under a fused speculative graph the taps never cross the ABI: the target
returns them as its ``SELECTED_LAYERS`` hidden states inside the same graph
and this module consumes them there. The drafter's cache is a one-plane bf16
leaf (576 columns, one slot per drafter layer, one KV head), the block is
``depth`` rows wide and *every* row drafts one token, so a step verifies
``depth + 1`` rows per request and proposes ``depth`` -- three, for this
checkpoint.

That leaf is what :meth:`DSparkDrafter.write_context` fills and what
:meth:`DSparkDrafter.block_hidden_paged` reads, and it is the path serving
takes: a fused block graph holds no dense context pool, only the latents
earlier steps wrote. :meth:`DSparkDrafter.forward` stays the dense reference --
the parity gate and a caller holding a whole prompt's taps run it -- and
:meth:`DSparkDrafter.forward_block_paged` is its paged mirror, one whole step
returning the same ``(logits, tokens)``. The two consume the same key set, and
``test_dspark_paged.py`` gates that they draft the same tokens from it.

A paged block costs one gather of ``window`` x ``latent_dim`` per request per
layer -- 2048 x 576 rows of bf16 latent, ~2.4 MB per request per layer, so
~7 MB per request per step for this checkpoint's three drafter layers, and that
times the batch: ~450 MB at a batch of 64. That is on top of the windowed
attention the model would pay anyway, and it is the reason the two key sets
are never concatenated: their scores are, and the gather is the only tensor
that carries the window's 576 latent columns.

The fused MLA decode kernel is the alternative, and it cannot serve this
attention as it stands: it writes the current step's latent rows and reads its
keys in one launch, which is sound only for a causal query (the block is not
causal, so a block row has to see its siblings' rows), and it redoes the query
absorption and the latent's norm and rope in bf16 in-kernel, which would move
the drafter off the arithmetic its goldens pin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig, SpeculativeMethod
from max.tensor import Tensor
from max.tensor.sharding import Sharded
from numpy.typing import NDArray

from .layers.dspark import (
    BLOCK_POSITION_OFFSET_TRTLLM,
    BLOCK_POSITION_OFFSET_VLLM,
    DSparkBackbone,
    DSparkLayer,
    DSparkMarkovHead,
    cos_sin_at,
    device_scalar,
    store_context_rows,
    window_reads,
)
from .layers.rope import yarn_inv_freq
from .model_config import (
    MTP_DRAFT_INPUT_POSITION_SPECIFIC,
    MTP_HEAD_TYPE_RNN,
    MTP_NORM_STRATEGY_EAGLE,
    MistralLarge4NormalizedConfig,
    MistralLarge4Yarn,
)

__all__ = [
    "BLOCK_POSITION_OFFSET_TRTLLM",
    "BLOCK_POSITION_OFFSET_VLLM",
    "DSPARK_BORROWED_TENSORS",
    "DSPARK_MODULE_PREFIX",
    "DSparkDrafter",
    "DSparkDrafterConfig",
    "block_mask",
    "build_dspark_kv_params",
    "convert_dspark_state_dict",
    "dspark_speculative_slots",
    "dspark_tensor_names",
    "head_from_weight",
    "window_start",
]

DSPARK_MODULE_PREFIX = "mtp_dspark."
"""The drafter's block in the target checkpoint, and its module's root name."""

DSPARK_BORROWED_TENSORS: tuple[str, ...] = (
    "tok_embeddings.weight",
    "output.weight",
)
"""The target's token embedding and LM head, which the drafter borrows.

The drafter ships no LM head of its own: it scores with the target's. Neither
name is a drafter parameter, so neither appears in
:func:`dspark_tensor_names`.
"""

_PER_LAYER_TENSORS: tuple[str, ...] = (
    "attention.wq.weight",
    "attention.wkv_a_with_mqa.weight",
    "attention.kv_a_norm.weight",
    "attention.wkv_b.weight",
    "attention.wo.weight",
    "attention_norm.weight",
    "ffn_norm.weight",
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
    "feed_forward.w2.weight",
)

_GLOBAL_TENSORS: tuple[str, ...] = (
    "backbone.target_projection.weight",
    "backbone.target_norm.weight",
    "backbone.final_norm.weight",
    "backbone.mask_embedding",
    "autoregressive_head.embedding.weight",
    "autoregressive_head.output.weight",
    "autoregressive_head.joint_proj.weight",
    "autoregressive_head.joint_proj.bias",
)


def dspark_tensor_names(num_layers: int) -> list[str]:
    """Returns the ``mtp_dspark.*`` names a complete drafter needs.

    Args:
        num_layers: Depth of the drafter's layer stack.

    Returns:
        Every name, in the checkpoint's own spelling.
    """
    names = [
        f"{DSPARK_MODULE_PREFIX}backbone.layers.{layer}.{suffix}"
        for layer in range(num_layers)
        for suffix in _PER_LAYER_TENSORS
    ]
    names += [f"{DSPARK_MODULE_PREFIX}{name}" for name in _GLOBAL_TENSORS]
    return names


_TensorT = TypeVar("_TensorT")


def convert_dspark_state_dict(
    state_dict: Mapping[str, _TensorT], num_layers: int
) -> dict[str, _TensorT]:
    """Renames the checkpoint's ``mtp_dspark.*`` block to the module's names.

    The module mirrors the checkpoint's layout, so the conversion is the prefix
    strip; :func:`dspark_tensor_names` is what makes a partial checkpoint fail
    here rather than at the first matmul.

    Args:
        state_dict: The target checkpoint's tensors, keyed by name.
        num_layers: Depth of the drafter's layer stack.

    Returns:
        The drafter's tensors, keyed as its module names them.

    Raises:
        ValueError: If a tensor the drafter needs is missing.
    """
    missing = [
        name
        for name in dspark_tensor_names(num_layers)
        if name not in state_dict
    ]
    if missing:
        raise ValueError(
            f"the checkpoint is missing {len(missing)} DSpark tensor(s):"
            f" {missing[:4]}"
        )
    return {
        name.removeprefix(DSPARK_MODULE_PREFIX): value
        for name, value in state_dict.items()
        if name.startswith(DSPARK_MODULE_PREFIX)
    }


@dataclass(frozen=True)
class DSparkDrafterConfig:
    """The drafter's geometry, read from the checkpoint's ``mtp`` block."""

    hidden_size: int
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    intermediate_size: int
    num_layers: int
    depth: int
    window: int
    target_layers: tuple[int, ...]
    markov_rank: int
    vocab_size: int
    rms_norm_eps: float
    softcap: float
    yarn: MistralLarge4Yarn
    block_position_offset: int = BLOCK_POSITION_OFFSET_TRTLLM
    """Where the block's first row sits in rotation space relative to the
    anchor's own position: :data:`BLOCK_POSITION_OFFSET_TRTLLM` (1) or
    :data:`BLOCK_POSITION_OFFSET_VLLM` (0)."""
    _rope_inv_freq: NDArray[np.floating] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validates the geometry the reference's own asserts guard.

        Raises:
            ValueError: If the tap list is empty or not ascending and unique,
                the block offset is not one of the two conventions, or
                ``depth``, ``num_layers`` or ``window`` is below one.
        """
        if not self.target_layers:
            raise ValueError("DSpark needs at least one target tap")
        if list(self.target_layers) != sorted(set(self.target_layers)):
            raise ValueError(
                "DSpark target_activation_layers must be ascending and unique"
            )
        if self.block_position_offset not in (
            BLOCK_POSITION_OFFSET_VLLM,
            BLOCK_POSITION_OFFSET_TRTLLM,
        ):
            raise ValueError(
                "block_position_offset must be 0 (vLLM) or 1 (TRT-LLM), got"
                f" {self.block_position_offset}"
            )
        if self.depth < 1 or self.num_layers < 1 or self.window < 1:
            raise ValueError(
                "DSpark needs depth, num_layers and window >= 1; got"
                f" {self.depth}, {self.num_layers}, {self.window}"
            )

    @property
    def qk_head_dim(self) -> int:
        """The full query/key head width: the nope and rope slices summed."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """Width of one stored latent key: ``[content | rope key]``."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def tap_width(self) -> int:
        """Columns the drafter reads: one hidden each, per target tap."""
        return self.hidden_size * len(self.target_layers)

    @property
    def attention_scale(self) -> float:
        """``qk_head_dim ** -0.5`` -- the target's scale, no YaRN temperature."""
        return float(self.qk_head_dim**-0.5)

    @property
    def num_drafts(self) -> int:
        """Tokens a block proposes: every row drafts one, the anchor included."""
        return self.depth

    @property
    def rope_inv_freq(self) -> NDArray[np.floating]:
        """The drafter's plain-YaRN inverse frequencies.

        LeRoPE is disabled for the drafter only (``alpha`` at zero), so this is
        the target's YaRN table without its learned multiplier.
        """
        cached = self._rope_inv_freq
        if cached is None:
            cached = yarn_inv_freq(
                theta=self.yarn.rope_theta,
                factor=self.yarn.factor,
                beta_fast=self.yarn.beta_fast,
                beta_slow=self.yarn.beta_slow,
                dim=self.qk_rope_head_dim,
                original_max_position_embeddings=(
                    self.yarn.original_max_position_embeddings
                ),
            )
            object.__setattr__(self, "_rope_inv_freq", cached)
        return cached

    @classmethod
    def from_normalized(
        cls,
        normalized: MistralLarge4NormalizedConfig,
        *,
        intermediate_size: int | None = None,
        block_position_offset: int = BLOCK_POSITION_OFFSET_TRTLLM,
    ) -> DSparkDrafterConfig:
        """Reads the drafter's geometry out of a resolved ``params.json``.

        Args:
            normalized: The checkpoint's normalized config.
            intermediate_size: The drafter FFN's inner width; the checkpoint's
                ``w1``/``w3`` are ``[4 * hidden, hidden]`` and loading asserts
                the choice against them.
            block_position_offset: The block's rotation offset.

        Returns:
            The drafter's config.

        Raises:
            ValueError: If the checkpoint declares no ``mtp`` block, its
                per-layer windows disagree with the single decode width, it
                declares no YaRN geometry, or its drafter is a head, a block
                input or a norm schedule this module does not implement.
        """
        mtp = normalized.mtp
        if mtp is None:
            raise ValueError(
                "the checkpoint declares no mtp block, so it carries no DSpark"
                " drafter"
            )
        if mtp.head_type != MTP_HEAD_TYPE_RNN:
            raise ValueError(
                f"the DSpark drafter implements mtp head_type={MTP_HEAD_TYPE_RNN}"
                f" only, but the checkpoint declares {mtp.head_type!r}"
            )
        if mtp.draft_input != MTP_DRAFT_INPUT_POSITION_SPECIFIC:
            raise ValueError(
                "the DSpark drafter implements"
                f" mtp draft_input={MTP_DRAFT_INPUT_POSITION_SPECIFIC} only, but"
                f" the checkpoint declares {mtp.draft_input!r}"
            )
        if mtp.norm_strategy != MTP_NORM_STRATEGY_EAGLE:
            raise ValueError(
                "the DSpark drafter implements"
                f" mtp norm_strategy={MTP_NORM_STRATEGY_EAGLE} only, but the"
                f" checkpoint declares {mtp.norm_strategy!r}"
            )
        windows = set(mtp.window)
        if len(windows) != 1:
            raise ValueError(
                "the DSpark drafter reads one decode window per layer, but"
                f" ragged_attention declares {mtp.window}"
            )
        if normalized.yarn is None:
            raise ValueError("the DSpark drafter needs the checkpoint's YaRN")
        return cls(
            hidden_size=normalized.hidden_size,
            num_heads=normalized.n_heads,
            qk_nope_head_dim=normalized.qk_nope_head_dim,
            qk_rope_head_dim=normalized.qk_rope_head_dim,
            v_head_dim=normalized.v_head_dim,
            kv_lora_rank=normalized.kv_lora_rank,
            intermediate_size=(
                intermediate_size
                if intermediate_size is not None
                else 4 * normalized.hidden_size
            ),
            num_layers=mtp.n_layers,
            depth=mtp.depth,
            window=windows.pop(),
            target_layers=mtp.target_layers,
            markov_rank=mtp.markov_rank,
            vocab_size=normalized.vocab_size,
            rms_norm_eps=normalized.rms_norm_eps,
            softcap=normalized.final_logit_softcapping,
            yarn=normalized.yarn,
            block_position_offset=block_position_offset,
        )


def head_from_weight(head_weight: Tensor) -> Callable[[Tensor], Tensor]:
    """The target's logit projection over a whole ``[vocab, hidden]`` weight.

    The one-device form of the target's LM head module, for a graph whose
    vocabulary is not sharded over tensor parallelism -- the dense parity
    entry's caller. A served graph hands :meth:`DSparkDrafter.base_logits` the
    target's own module instead, whose per-rank result it then all-gathers.

    Args:
        head_weight: ``[vocab, hidden]`` the target's LM head weight.

    Returns:
        ``[rows, hidden] -> [rows, vocab]``, in the weight's dtype.
    """
    return lambda hidden: F.matmul(hidden, head_weight.T)


def window_start(anchor_position: int, window: int) -> int:
    """The first context position a block at ``anchor_position`` attends.

    The pool is every position up to and including the anchor's own, and the
    block reads the last ``window`` of them.

    Args:
        anchor_position: The anchor token's 0-based position.
        window: The drafter's sliding window.

    Returns:
        The first position in the window, or 0 for a shorter context.
    """
    return max(0, anchor_position + 1 - window)


def block_mask(
    context_positions: Tensor,
    anchor_positions: Tensor,
    window: int,
    depth: int,
) -> Tensor:
    """Returns which keys each block row may attend, ``[rows, n_ctx + rows]``.

    A block row reads the context rows of its own request in
    ``[max(0, anchor + 1 - window), anchor]`` -- bounded on both ends, so a
    caller holding a whole prompt's latents cannot leak the tokens after the
    anchor into the block -- and every block row of its own request, its own
    row included. Rows are request-major, so the block part is block diagonal;
    nothing about it is causal.

    Every row reads one shared set of ``context_positions``, so this is the
    dense single-pool mask: one prompt's context, with as many anchors on it
    as the batch holds. A row's request only narrows the pool per anchor, it
    does not give a request context rows of its own. The row axis is built
    with ``F.arange`` over a static ``batch``, so a symbolic batch -- what the
    serving block driver will hand it -- needs a variant that keeps the row
    count symbolic.

    Args:
        context_positions: ``[n_ctx]`` the context rows' absolute positions.
            Padding rows must carry positions outside every window: a filler
            that sits inside one is a real key to the block that reads it.
        anchor_positions: ``[batch]`` each block's anchor position.
        window: The drafter's sliding window.
        depth: Rows per block.

    Returns:
        The boolean mask, one row per block row and one column per key.
    """
    batch = anchor_positions.shape[0]
    n_ctx = context_positions.shape[0]
    rows = batch * depth
    device = context_positions.mapping
    earliest = anchor_positions - (window - 1)
    zero = device_scalar(0, anchor_positions)
    lower = F.where(
        earliest > zero,
        earliest,
        F.broadcast_to(zero, [batch]),
    )
    # Rows are request-major, so every row's own request is its index divided
    # by the block width; that gives each row its own anchor and window.
    request = (
        F.arange(0, rows, 1, out_dim=rows, dtype=DType.int32, device=device)
        // depth
    )
    anchors = F.gather(anchor_positions, request, axis=0)
    starts = F.gather(lower, request, axis=0)
    context = F.reshape(context_positions, [1, n_ctx])
    in_window = (context >= F.reshape(starts, [rows, 1])) & (
        context <= F.reshape(anchors, [rows, 1])
    )
    # Every block row of a request sees every block row of that request -- its
    # own included -- and none of another request's.
    same_request = F.reshape(request, [rows, 1]) == F.reshape(
        request, [1, rows]
    )
    return F.concat([in_window, same_request], axis=-1)


def dspark_speculative_slots(
    speculative: SpeculativeConfig | None,
    config: DSparkDrafterConfig,
) -> tuple[SpeculativeMethod | None, int]:
    """The speculative method and draft width a drafter leaf declares.

    A block draft writes ``num_draft_tokens`` rows per draft forward, which is
    what sizes the leaf's draft dispatch metadata -- and the block always runs
    its trained ``depth`` rows, whatever a step ends up verifying. The width is
    therefore the block's, not the step's: a ``--num-speculative-tokens``
    narrower than the block would size the drafter's dispatch below its own
    query width, and the leaf would record a width the graph does not use.

    Args:
        speculative: The pipeline's speculative configuration, or ``None`` when
            speculation is off.
        config: The drafter's geometry, whose ``num_drafts`` is the block's
            trained width.

    Returns:
        ``(method, num_draft_tokens)``: the block's width for ``dspark``,
        ``(None, 0)`` without speculation.

    Raises:
        ValueError: If a ``dspark`` build requests a width other than the
            block's. The block samples its anchor slot, so the step's verify
            width is the block's plus the bonus row and cannot be narrowed.
    """
    if speculative is None:
        return None, 0
    method = speculative.speculative_method
    if method != "dspark":
        return method, speculative.num_speculative_tokens or 0
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != config.num_drafts:
        raise ValueError(
            "Mistral Large 4's DSpark block drafts exactly"
            f" {config.num_drafts} tokens per step -- the block's trained"
            " depth, since the drafter samples its anchor slot and the target"
            f" verifies one row more -- got num_speculative_tokens={requested}"
        )
    return method, config.num_drafts


def build_dspark_kv_params(
    kv_cache_config: KVCacheConfig,
    *,
    devices: list[DeviceRef],
    config: DSparkDrafterConfig,
    speculative: SpeculativeConfig | None = None,
    dtype: DType = DType.bfloat16,
    data_parallel_degree: int = 1,
) -> KVCacheParams:
    """Builds the drafter's own one-plane latent cache.

    One 576-wide row per token per layer: unlike the target's latent, the
    drafter's cache is not quantized, and there is no separate value plane
    because MLA's values come out of the same latent.

    The leaf must declare the speculative method, exactly as the target's does
    (:meth:`MistralLarge4Config.construct_kv_params`): the two slots that only a
    speculative leaf carries -- the draft dispatch metadata and its split-K
    count -- are what make a record ten fields wide instead of eight, and the
    engine's slot map pins the drafter's record at that width. A leaf built
    without them is refused at boot with an arity mismatch, so the values are
    derived here from the one place they live, ``pipeline_config.speculative``,
    rather than left to each caller.

    The draft width is the *block's*, not the step's: the block always runs its
    trained ``depth`` rows, so a narrower ``--num-speculative-tokens`` is
    refused rather than folded into the leaf (see
    :func:`dspark_speculative_slots`).

    This leaf shares a pool with the target's, so the caller hands it the
    pipeline's own ``KVCacheConfig`` and ``data_parallel_degree`` rather than
    defaults: a cache tree reads its pool configuration off the leaves that
    declare one and refuses any disagreement (``_agreed_pool``:
    ``cache_params.py``) over the page size, the data-parallel degree, the
    devices, prefix caching, the connector and the hash settings. A leaf built
    from a different ``KVCacheConfig`` -- a bare ``KVCacheConfig(page_size=...)``
    in a test, say -- is a different pool.

    Args:
        kv_cache_config: The pipeline's own page-size and budget configuration,
            which the target's leaves were built from.
        devices: Devices the cache spans.
        config: The drafter's geometry.
        speculative: The pipeline's speculative configuration
            (``pipeline_config.speculative``). A servable build passes it;
            ``None`` declares the plain eight-field record.
        dtype: The activation dtype the latents are stored in, which has to be
            the drafter's own: bfloat16 for this checkpoint, and what the
            engine's sidecar pins as ``draft_kv_cache_dtype``. A float32
            caller is a numeric test.
        data_parallel_degree: Attention replicas, from
            ``pipeline_config.model.data_parallel_degree`` -- the target's own
            value, for the same reason as the config.

    Returns:
        The drafter's cache leaf.

    Raises:
        ValueError: If the speculative method is not the one this drafter
            serves, or requests a width the block does not draft.
    """
    method, drafts = dspark_speculative_slots(speculative, config)
    if method is not None and method != "dspark":
        raise ValueError(
            "the DSpark drafter serves speculative_method='dspark' only,"
            f" not {method!r}"
        )
    leaf = kv_cache_config.to_params(
        dtype=dtype,
        # One KV head: an MLA cache holds a single latent vector per token.
        n_kv_heads=1,
        head_dim=config.latent_dim,
        num_layers=config.num_layers,
        devices=devices,
        is_mla=True,
        num_q_heads=config.num_heads,
        data_parallel_degree=data_parallel_degree,
        speculative_method=method,
        num_draft_tokens=drafts,
        # No ``window_size``: the window is a read policy (which slots a block
        # gathers), and a windowed leaf would evict pages the block window of a
        # later step still reads.
    )
    assert isinstance(leaf, KVCacheParams)
    return leaf


class DSparkDrafter(Module[..., tuple[Tensor, ...]]):
    """The drafter's module tree, named after the checkpoint's block.

    Parameters are named ``backbone.*`` and ``autoregressive_head.*``, exactly
    as the checkpoint's ``mtp_dspark.*`` block names them below the prefix, so
    :func:`convert_dspark_state_dict` is a prefix strip.
    """

    def __init__(
        self, config: DSparkDrafterConfig, *, emit_intermediates: bool = False
    ) -> None:
        """Builds the projection, the layers and the head.

        Args:
            config: The drafter's geometry.
            emit_intermediates: Whether :meth:`forward` also returns the debug
                outputs the parity gate compares (head hidden, shared-head
                logits, recurrent states, context latents). A serving graph
                leaves this off: its outputs are the draft tokens and their
                logits, and nothing wider.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.window = config.window
        self.block_position_offset = config.block_position_offset
        self.emit_intermediates = emit_intermediates
        self.softcap = config.softcap
        self.rope_inv_freq = config.rope_inv_freq
        self.backbone = DSparkBackbone(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            depth=config.depth,
            tap_width=config.tap_width,
            num_heads=config.num_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            intermediate_size=config.intermediate_size,
            scale=config.attention_scale,
            norm_eps=config.rms_norm_eps,
            rope_inv_freq=config.rope_inv_freq,
        )
        self.autoregressive_head = DSparkMarkovHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            markov_rank=config.markov_rank,
        )

    def project_context(self, taps: Tensor) -> Tensor:
        """Projects the concatenated target taps to the drafter's stream.

        Args:
            taps: ``[rows, n_taps * hidden]``, the target's post-layer
                residuals of layers ``target_layers``, concatenated in that
                order. The column order is part of the trained parameter.

        Returns:
            ``[rows, hidden]``, projected and normalized.
        """
        return self.backbone.target_norm(self.backbone.target_projection(taps))

    def context_latents(
        self, projected: Tensor, positions: Tensor
    ) -> list[Tensor]:
        """Builds every layer's latent cache rows for a run of context tokens.

        The context bypasses the layers' input norms: ``projected`` is already
        normalized, which is the eagle strategy and what the drafter trained
        with.

        Args:
            projected: ``[rows, hidden]`` from :meth:`project_context`.
            positions: ``[rows]`` each row's absolute position.

        Returns:
            One ``[rows, latent_dim]`` tensor per layer, in layer order.
        """
        cos_sin = cos_sin_at(self.rope_inv_freq, positions, projected.dtype)
        return [
            layer.attention.latent_keys(projected, cos_sin)
            for layer in self.backbone.layers
        ]

    def write_context(
        self,
        taps: Tensor,
        positions: Tensor,
        row_offsets: Tensor,
        kv: PagedCacheValues,
    ) -> None:
        """Writes the taps' rows into the drafter's paged latent cache.

        The block driver's ``materialize`` phase: every merged row of the step
        -- the prompt chunk of a prefill row, the committed tokens of a decode
        row -- is projected and written at its own position, which is where the
        next step's block reads it. The rows the step does not commit are
        overwritten by the next step's, because this writes at the leaf's
        pre-iteration cache lengths.

        Args:
            taps: ``[rows, n_taps * hidden]`` the target's taps for the rows.
            positions: ``[rows]`` each row's absolute position.
            row_offsets: ``[batch + 1]`` uint32 ragged offsets of the rows.
            kv: The drafter's paged leaf, at its pre-iteration cache lengths.
        """
        projected = self.project_context(taps)
        for index, latent in enumerate(
            self.context_latents(projected, positions)
        ):
            store_context_rows(kv, latent, row_offsets, index)

    def block_from_anchors(self, anchor_embeds: Tensor) -> Tensor:
        """Builds the block's input rows from its anchors' embeddings.

        Row 0 of each block is the anchor token's embedding; the rest are the
        learned per-slot mask rows. Nothing is re-embedded from a drafted
        token: the intra-block dependency lives in the recurrent head, which is
        what keeps the backbone pass parallel.

        This is where the served graph enters, with the rows the target's own
        embedding already produced: the target shards its vocabulary across the
        tensor-parallel ranks, so the drafter cannot look a token up in the
        weight itself (:meth:`embed_block` does, for a one-device weight). The
        block driver's anchors are the tokens the step commits, so every one of
        them is a real token; the out-of-vocabulary case :meth:`embed_block`
        zeroes is a reference-math detail of the standalone entry.

        Args:
            anchor_embeds: ``[batch, hidden]`` each block's anchor embedding,
                in the activation dtype and on every rank.

        Returns:
            ``[batch * depth, hidden]``, request-major.
        """
        batch = anchor_embeds.shape[0]
        masks = F.broadcast_to(
            self.backbone.mask_embedding.unsqueeze(0),
            [batch, self.depth - 1, self.hidden_size],
        )
        block = F.concat([anchor_embeds.unsqueeze(1), masks], axis=1)
        return F.reshape(block, [batch * self.depth, self.hidden_size])

    def embed_block(self, anchor_ids: Tensor, embed_weight: Tensor) -> Tensor:
        """Builds the block's input rows from its anchors, over one weight.

        The one-device form of :meth:`block_from_anchors`: it looks the anchors
        up in a whole ``[vocab, hidden]`` table, which a graph whose vocabulary
        is sharded over tensor parallelism cannot do (a gather from one rank's
        shard needs the rows the other ranks hold). Such a graph embeds through
        the target's own embedding module and enters at
        :meth:`block_from_anchors` instead.

        Args:
            anchor_ids: ``[batch]`` each block's anchor token.
            embed_weight: ``[vocab, hidden]`` the target's token embedding.

        Returns:
            ``[batch * depth, hidden]``, request-major.
        """
        # An anchor outside the vocabulary embeds to zero, as in the reference:
        # the mask token id the checkpoint's own drafting code uses for row 0
        # of an unanchored block is deliberately out of range. The clamp is
        # built from device scalars: a Python bound beside a device tensor is
        # built on the CPU, and the comparison then mixes devices.
        zero = device_scalar(0, anchor_ids)
        # The table's row count is a static weight dim, so it reads as a host
        # ``int``; a ``Dim`` is not a constant the graph builds from.
        vocab = int(embed_weight.shape[0])
        limit = device_scalar(vocab, anchor_ids)
        last = device_scalar(vocab - 1, anchor_ids)
        valid = (anchor_ids >= zero) & (anchor_ids < limit)
        indices = F.where(anchor_ids > last, last, anchor_ids)
        indices = F.where(indices < zero, zero, indices)
        anchors = F.gather(embed_weight, indices.cast(DType.int32), axis=0)
        anchors = F.where(valid.unsqueeze(1), anchors, F.zeros_like(anchors))
        return self.block_from_anchors(anchors)

    def block_positions(self, anchor_positions: Tensor) -> Tensor:
        """Returns each block row's rotated position, ``[batch * depth]``.

        Row ``j`` is rotated at ``anchor_position + offset + j``: the position
        of the token it predicts under the TRT-LLM convention (offset 1), or
        the anchor's own under vLLM's (offset 0).

        Args:
            anchor_positions: ``[batch]`` each block's anchor position.
        """
        batch = anchor_positions.shape[0]
        steps = F.arange(
            0,
            self.depth,
            1,
            out_dim=self.depth,
            dtype=DType.int32,
            device=anchor_positions.mapping,
        )
        grid = (
            F.reshape(anchor_positions.cast(DType.int32), [batch, 1])
            + F.reshape(steps, [1, self.depth])
            + self.block_position_offset
        )
        return F.reshape(grid, [batch * self.depth])

    def forward_block(
        self,
        block_input: Tensor,
        context_keys: list[Tensor],
        context_positions: Tensor,
        positions: Tensor,
        anchor_positions: Tensor,
    ) -> Tensor:
        """Runs the block through every layer and returns its head hidden.

        The layer boundary chain is the eagle schedule: the input is pre-normed
        once with layer 0's ``attention_norm``, each layer adds its attention
        and FFN outputs to one residual stream, and the stream is re-normed
        with the next layer's ``attention_norm`` -- ``final_norm`` after the
        last. The returned rows are therefore already normalized, which is what
        the LM head and the recurrent head consume.

        Args:
            block_input: ``[batch * depth, hidden]`` from :meth:`embed_block`.
            context_keys: One ``[n_ctx, latent_dim]`` tensor per layer.
            context_positions: ``[n_ctx]`` the context rows' positions.
            positions: ``[batch * depth]`` the block rows' positions.
            anchor_positions: ``[batch]`` each block's anchor position, which
                bounds both ends of that block's context window.

        Returns:
            ``[batch * depth, hidden]``.
        """
        mask = block_mask(
            context_positions, anchor_positions, self.window, self.depth
        )
        return self._run_layers(
            block_input,
            lambda index, layer, x: layer.attention(
                x, context_keys[index], positions, mask
            ),
        )

    def block_hidden_paged(
        self,
        block_input: Tensor,
        anchor_positions: Tensor,
        kv: PagedCacheValues,
    ) -> Tensor:
        """Runs the block through every layer, reading the paged window.

        The paged mirror of :meth:`forward_block`: the same eagle chain, over
        the same key set, with each layer's window materialized by slot out of
        the drafter's own leaf instead of masked out of a dense pool. Under the
        block driver this is the ``forward_block`` phase.

        The addressing is the caller's: the slots come from ``anchor_positions``
        alone, so nothing here depends on how the leaf was staged. The clamp
        does: the anchor is bounded by the leaf's ``cache_lengths``
        (:func:`window_reads`), which means the leaf has to be at its *block*
        position -- past the rows this iteration commits, as the block driver
        stages it. The gather resolves a slot through the request's page table
        and guards nothing, so an anchor the leaf cannot serve would be an
        unallocated page rather than a wrong number.

        Args:
            block_input: ``[batch * depth, hidden]`` from :meth:`embed_block`.
            anchor_positions: ``[batch]`` each block's anchor position, the last
                token this iteration commits.
            kv: The drafter's paged leaf, already holding every committed row.

        Returns:
            ``[batch * depth, hidden]``, normalized the same way.
        """
        reads = window_reads(
            anchor_positions, kv.cache_lengths, self.window, self.depth
        )
        positions = self.block_positions(reads.anchor)
        return self._run_layers(
            block_input,
            lambda index, layer, x: layer.attention.forward_paged(
                x,
                kv,
                reads,
                positions,
                depth=self.depth,
                window=self.window,
                layer_idx=index,
            ),
        )

    def _run_layers(
        self,
        block_input: Tensor,
        attend: Callable[[int, DSparkLayer, Tensor], Tensor],
    ) -> Tensor:
        """The eagle layer chain, with each layer's attention supplied.

        Args:
            block_input: ``[batch * depth, hidden]`` the block's input rows.
            attend: ``(layer index, layer, input-normed rows) -> attention
                output``. The dense and paged blocks differ in nothing else.

        Returns:
            ``[batch * depth, hidden]``, normalized by the last layer's exit
            norm.
        """
        layers = self.backbone.layers
        residual = block_input
        x = layers[0].attention_norm(residual)
        for index, layer in enumerate(layers):
            residual = residual + attend(index, layer, x)
            residual = residual + layer.feed_forward(layer.ffn_norm(residual))
            next_norm = (
                layers[index + 1].attention_norm
                if index + 1 < len(layers)
                else self.backbone.final_norm
            )
            x = next_norm(residual)
        return x

    def base_logits(
        self, head_hidden: Tensor, head: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Scores the block with the target's LM head, soft-capped.

        The cap covers these logits only: the recurrent bias added in
        :meth:`draft` is not capped, and neither is the sum.

        The head is the target's, and the target shards its vocabulary across
        the tensor-parallel ranks: its projection returns a rank's columns of
        the vocabulary, which has to be gathered back before anything reads a
        row's full logit vector. The all-gather is the same collective the
        target's own logit projection runs (:func:`project_to_logits`), and
        every rank holds the whole vocabulary afterwards -- the recurrent head
        and the argmax below both read it on every rank.

        Args:
            head_hidden: ``[rows, hidden]`` from :meth:`forward_block`.
            head: ``[rows, hidden] -> [rows, vocab]`` the target's LM head,
                which may return one rank's vocabulary columns per rank. A
                one-device weight goes through :func:`head_from_weight`.

        Returns:
            ``[rows, vocab]`` float32, ``softcap * tanh(x / softcap)``.
        """
        logits = head(head_hidden)
        if any(isinstance(p, Sharded) for p in logits.placements):
            logits = F.allgather(logits, tensor_axis=-1)
        # The head is the target's bf16 projection, so its result carries the
        # activation dtype's rounding before the cap widens it: a float32
        # result here would be *more* precise than the model the drafter was
        # trained beside, and its logits would sit half a bf16 ulp off.
        logits = F.cast(F.cast(logits, head_hidden.dtype), DType.float32)
        cap = self.softcap
        return F.tanh(logits * (1.0 / cap)) * cap

    def draft(
        self,
        base_logits: Tensor,
        head_hidden: Tensor,
        anchor_ids: Tensor,
        sampler: Callable[[Tensor, int], Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unrolls the recurrent chain across the block's rows.

        The state resets to zeros per block: nothing about it persists across
        target steps.

        Args:
            base_logits: ``[batch * depth, vocab]`` from :meth:`base_logits`.
            head_hidden: ``[batch * depth, hidden]`` head rows.
            anchor_ids: ``[batch]`` each block's anchor token, the chain's
                first ``previous``.
            sampler: ``(logits [batch, vocab], position) -> [batch]`` token
                picker; greedy argmax when ``None``.

        Returns:
            ``(tokens [batch, depth], logits [batch, depth, vocab],
            states [batch, depth, markov_rank])``, where ``states[i]`` is the
            state entering step ``i``.
        """
        batch = anchor_ids.shape[0]
        depth = self.depth
        vocab = base_logits.shape[-1]
        base = F.reshape(base_logits, [batch, depth, vocab])
        hidden = F.reshape(head_hidden, [batch, depth, self.hidden_size])
        state = Tensor.zeros(
            [batch, self.autoregressive_head.markov_rank],
            dtype=head_hidden.dtype,
            device=head_hidden.mapping,
        )
        previous = anchor_ids.cast(DType.int32)
        tokens: list[Tensor] = []
        logits: list[Tensor] = []
        states: list[Tensor] = []
        for step in range(depth):
            states.append(state)
            state, bias = self.autoregressive_head(
                state, previous, hidden[:, step, :]
            )
            row = base[:, step, :] + F.cast(bias, DType.float32)
            previous = _argmax(row) if sampler is None else sampler(row, step)
            tokens.append(previous)
            logits.append(row)
        return (
            F.stack(tokens, axis=1),
            F.stack(logits, axis=1),
            F.stack(states, axis=1),
        )

    def forward(
        self,
        taps: Tensor,
        context_positions: Tensor,
        anchor_positions: Tensor,
        anchor_ids: Tensor,
        embed_weight: Tensor,
        head_weight: Tensor,
    ) -> tuple[Tensor, ...]:
        """Runs one whole draft step for a batch of anchors on one context.

        This is the dense entry: every context row the caller holds is
        projected and masked per block, so it is what the parity gate and a
        caller holding a whole prompt's taps run. The serving path is the
        phases (:meth:`project_context`, :meth:`context_latents`,
        :meth:`embed_block`, :meth:`forward_block`, :meth:`base_logits`,
        :meth:`draft`), which cache the context latents and keep the
        intermediates out of the graph's outputs.

        ``taps`` is one pool of context rows shared by the whole batch -- one
        prompt, with as many anchors on it as the batch holds -- because
        :func:`block_mask` narrows that one pool per anchor rather than giving
        each request rows of its own. Padding rows belong at positions outside
        every block's window.

        Args:
            taps: ``[n_ctx, n_taps * hidden]`` the target's taps for the
                context rows.
            context_positions: ``[n_ctx]`` those rows' absolute positions.
            anchor_positions: ``[batch]`` each block's anchor position; a block
                reads the context positions in
                ``[max(0, anchor + 1 - window), anchor]``.
            anchor_ids: ``[batch]`` each block's anchor token.
            embed_weight: ``[vocab, hidden]`` the target's token embedding.
            head_weight: ``[vocab, hidden]`` the target's LM head.

        Returns:
            ``(logits, draft_tokens)`` when built for serving: the logits
            after the recurrent bias ``[batch, depth, vocab]`` and the drafted
            tokens ``[batch, depth]``. With ``emit_intermediates`` it returns
            ``(head_hidden, base_logits, logits, draft_tokens, markov_states,
            *context_latents)`` instead: the block's head rows
            ``[rows, hidden]``, its soft-capped shared logits
            ``[rows, vocab]``, the state each recurrent step entered
            ``[batch, depth, markov_rank]``, and one latent cache tensor per
            layer.
        """
        projected = self.project_context(taps)
        latents = self.context_latents(projected, context_positions)
        block_input = self.embed_block(anchor_ids, embed_weight)
        positions = self.block_positions(anchor_positions)
        head_hidden = self.forward_block(
            block_input, latents, context_positions, positions, anchor_positions
        )
        base = self.base_logits(head_hidden, head_from_weight(head_weight))
        tokens, logits, states = self.draft(base, head_hidden, anchor_ids)
        if not self.emit_intermediates:
            return logits, tokens
        return (
            head_hidden,
            base,
            logits,
            tokens,
            states,
            *latents,
        )

    def forward_block_paged(
        self,
        block_input: Tensor,
        anchor_ids: Tensor,
        anchor_positions: Tensor,
        kv: PagedCacheValues,
        head: Callable[[Tensor], Tensor],
        sampler: Callable[[Tensor, int], Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Runs one whole draft step against the drafter's paged cache.

        The serving mirror of :meth:`forward`: the same block rows, the same
        recurrent chain, and the same key set -- except the window comes out of
        the drafter's own leaf, which :meth:`write_context` filled over the
        steps before this one, instead of from a dense pool the caller holds.
        Nothing about the batch has to be known when the graph is built: the
        rows, the requests and their window are all runtime shapes.

        ``block_input`` arrives already embedded, because the embedding and the
        head are the target's and both are sharded over its vocabulary: the
        caller embeds its anchors through the target's own module and enters
        :meth:`block_from_anchors` there.

        The caller owns the write: call :meth:`write_context` for the rows the
        iteration commits before this, then enter here with the leaf at its
        *block* position -- the same leaf with those rows counted -- which is
        what the block driver stages for the read. The read addresses every row
        through ``anchor_positions``, so the positions themselves come from the
        caller and not from ``cache_lengths``; those lengths bound the anchor
        instead (:func:`window_reads` clamps it to the leaf's last written row),
        so an anchor past them -- an off-by-N caller, or a leaf still staged at
        its ctx position -- is read as the last row rather than as an
        unallocated page.

        Args:
            block_input: ``[batch * depth, hidden]`` the block's rows, from
                :meth:`block_from_anchors`.
            anchor_ids: ``[batch]`` each block's anchor token, which seeds the
                recurrent chain.
            anchor_positions: ``[batch]`` each anchor's absolute position. Its
                own row is the window's last key, so it must be the position of
                the last row :meth:`write_context` wrote for that request.
            kv: The drafter's paged leaf at its block position, holding every
                committed row.
            head: ``[rows, hidden] -> [rows, vocab]`` the target's LM head (see
                :meth:`base_logits`).
            sampler: ``(logits [batch, vocab], position) -> [batch]`` token
                picker; greedy argmax when ``None``, as in :meth:`draft`.

        Returns:
            ``(logits, draft_tokens)``, as :meth:`forward` returns them for a
            serving graph: ``[batch, depth, vocab]`` after the recurrent bias
            and ``[batch, depth]``.
        """
        head_hidden = self.block_hidden_paged(block_input, anchor_positions, kv)
        base = self.base_logits(head_hidden, head)
        tokens, logits, _ = self.draft(base, head_hidden, anchor_ids, sampler)
        return logits, tokens


def _argmax(logits: Tensor) -> Tensor:
    """Returns each row's argmax index as int32, ``[rows]``."""
    return F.squeeze(F.argmax(logits, axis=-1), axis=-1).cast(DType.int32)
