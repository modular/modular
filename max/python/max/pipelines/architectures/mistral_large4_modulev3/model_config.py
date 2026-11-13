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
"""Mistral Large 4 config, read from the checkpoint's ``params.json``.

``mistralai/Mistral-Large-4-1T-A52B-NVFP4`` ships no ``config.json``; a loader
must read ``params.json``, whose per-layer patterns are encoded as repeating
comma-separated strings rather than lists. This module resolves them into
explicit per-layer tuples, so no layer-level decision is re-derived by a
modulo rule at the point of use, and it emits the HF-style ``config.json`` the
rest of MAX and Mach read (:func:`normalized_hf_config`), reproducing
``transformers``' ``convert_mistral_large4_to_hf.py`` output.

``MistralLarge4NormalizedConfig`` is not yet an ``ArchConfig``: M0 ships
metadata only. M1's ``MistralLarge4Config`` (which forks
``DeepseekV3_2Config``) is built from it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# The two quantization conventions this checkpoint family uses. ModelOpt's
# ``FP8_PB_WO`` blocks a weight per 128x128 tile with no activation scale
# (activations are quantized dynamically); ``NVFP4`` packs two E2M1 values per
# byte with a per-16-element E4M3 scale plus two per-tensor ``float32`` scales.
QUANT_ALGO_FP8_PB_WO = "FP8_PB_WO"
QUANT_ALGO_NVFP4 = "NVFP4"

# ``params.json`` abbreviates YaRN's betas in the opposite direction to what
# the names suggest: ``yarn.beta`` is ``beta_fast`` and ``yarn.alpha`` is
# ``beta_slow`` (stated in the HF converter; the 32/1 value pair only fits that
# way round, since ``beta_slow`` is the smaller rotation count).
_YARN_BETA_FAST_KEY = "beta"
_YARN_BETA_SLOW_KEY = "alpha"

# The NoPE cycle, DSA indexer ownership and dense/MoE split are all encoded as
# repeating comma-separated strings in ``params.json``.
_ROPE_TO_FLAG = {"rope": True, "nope": False}
_INDEXER_TO_FLAG = {"full": True, "shared": False}

# The vision tower the loader skips until the vision milestone, declared here
# because the marker token ids belong to the text config.
_VISION_PROJECTOR_HIDDEN_ACT = "gelu"
_VISION_FEATURE_LAYER = -1

# The converter stamps the transformers version that produced the file; the
# base repo's own ``config.json`` carries the same one, and the two ML4 repos
# must agree or a diff between them reads as a schema change.
_HF_TRANSFORMERS_VERSION = "5.21.0.dev0"
_WEIGHTS_INDEX_NAME = "consolidated.safetensors.index.json"

# ``params.json`` keys that only this checkpoint family carries, which is how a
# checkpoint's ``params.json`` is recognized before any of its geometry is read.
_PARAMS_SIGNATURE = frozenset({"nope", "dsa", "kv_sharing", "ssmax_scaling"})

# The token ids the generated config declares. The HF converter reads them off
# ``tekken.json``, which is the base repo's; both repos' ``generation_config``
# agree, and a checkpoint that ships no tokenizer of its own has no other
# source. They are the values the published base config carries.
_BOS_TOKEN_ID = 1
_EOS_TOKEN_ID = 2

TOKENIZER_REPO_ID = "mistralai/Mistral-Large-4-1T-A52B"
"""The repo this family's tokenizer artifacts load from.

The NVFP4 release carries weights and ``params.json``; its ``tokenizer.json``,
``tokenizer_config.json`` and ``chat_template.jinja`` are the base release's,
byte-identical, so every consumer loads them from there.
"""

TOKENIZER_ARTIFACT_REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
"""The revision those artifacts load at, pinned so a moving ``main`` cannot
re-define the tokenizer a deployment boots with.

The tokenizer crate pins the same revision for this family
(``tokenizer-core``'s ``data/pinned_revisions.json``, keyed by this repo id),
which its native family and Universal both resolve; the two must name the same
commit or MAX and Mach tokenize one prompt differently.
"""

ML4_DSPARK_BLOCK_WIDTH = 3
"""Rows per DSpark block: the checkpoint's ``params.json`` ``mtp.depth``.

Every row of a block drafts one token, so a step proposes this many and
verifies one more. The width is the drafter's trained geometry, not a serving
knob: the target's leaf and the drafter's reserve the same count of draft slots
per step (``_agreed_pool`` refuses two leaves that disagree), so a
``--num-speculative-tokens`` that differs is refused rather than silently
sizing one of the two leaves differently.
"""

ML4_DSPARK_VERIFY_ROWS = ML4_DSPARK_BLOCK_WIDTH + 1
"""Rows a DSpark verify scores per request: the block's drafts and the anchor.

The base and spec graphs both size every sparse attention decode launch's
split-K for this many rows, so a one-row decode step reduces each row's keys in
the verify's order and a solo request's greedy output does not depend on
whether speculation is on.
"""


def pins_verify_split(row_invariant_decode: bool, graph_mode: str) -> bool:
    """Whether the sparse MLA op passes the row-invariant split pin.

    The pin and the capture key are two views of one launch: the op takes the
    pin, and the dispatch key has to take the same count, so both read this
    predicate rather than each spelling out the condition. ``mla_decode_graph``
    takes no split pin, so a decode-only graph launches the dispatcher's own
    fitted count and the key must not take the pin's.

    Args:
        row_invariant_decode: Whether the model took the row-invariant pins.
        graph_mode: ``"prefill"``, ``"decode"`` or ``"auto"``.
    """
    return row_invariant_decode and graph_mode != "decode"


def find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    """Returns the dimension at which YaRN blends ``num_rotations`` rotations.

    Mirrors ``modeling_rope_utils.find_correction_dim``. ``dim`` is the rope
    slice's width (``qk_rope_head_dim``), whose half sizes the ramp.

    Args:
        num_rotations: Rotations completed inside the original window.
        dim: Full rope width in dimensions.
        base: RoPE theta.
        max_position_embeddings: The pre-training window (512 here).

    Returns:
        The fractional dimension index.
    """
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def yarn_correction_range(
    *,
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> tuple[int, int]:
    """Returns YaRN's truncated ``(low, high)`` ramp bounds.

    Args:
        beta_fast: The fast-rotation beta (``yarn.beta`` in ``params.json``).
        beta_slow: The slow-rotation beta (``yarn.alpha`` in ``params.json``).
        dim: Full rope width in dimensions.
        base: RoPE theta.
        original_max_position_embeddings: The pre-training window.

    Returns:
        The integer ``(low, high)`` bounds of the extrapolation ramp.

    Raises:
        ValueError: If ``low`` is not below ``high``.
    """
    low = math.floor(
        find_correction_dim(
            beta_fast, dim, base, original_max_position_embeddings
        )
    )
    high = math.ceil(
        find_correction_dim(
            beta_slow, dim, base, original_max_position_embeddings
        )
    )
    low = max(low, 0)
    high = min(high, dim - 1)
    if not low < high:
        raise ValueError(
            f"YaRN ramp is empty: low={low} high={high} for dim={dim}, "
            f"theta={base}, original_max_position_embeddings="
            f"{original_max_position_embeddings}"
        )
    return low, high


@dataclass(frozen=True, kw_only=True)
class MistralLarge4Yarn:
    """YaRN parameters, plus LeRoPE's learned global frequency multiplier.

    ``attention_factor`` is 1.0: the checkpoint applies YaRN with no magnitude
    scaling, so cos/sin are not rescaled and the attention scale stays an
    explicit ``qk_head_dim ** -0.5``. ``lerope_alpha_dim`` is the width of the
    single global ``lerope.alpha`` tensor, which multiplies the finished YaRN
    inverse frequencies and is unrelated to the YaRN betas.
    """

    rope_theta: float
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    attention_factor: float
    lerope: bool
    lerope_alpha_dim: int

    @property
    def correction_range(self) -> tuple[int, int]:
        """The truncated ``(low, high)`` ramp bounds on the rope half-width."""
        return yarn_correction_range(
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            # The ramp sizes off the rope slice, not the full qk head.
            dim=2 * self.lerope_alpha_dim,
            base=self.rope_theta,
            original_max_position_embeddings=(
                self.original_max_position_embeddings
            ),
        )


@dataclass(frozen=True, kw_only=True)
class MistralLarge4LayerSchedule:
    """The per-layer structure, one entry per layer, never a modulo rule.

    ``kv_source`` maps every layer to the layer whose MLA latent it reads: a
    layer that is its own source owns a cache, any other entry is a read-only
    sharer. ``indexer_source`` is the analogous map for the DSA indexer's
    top-k selection, which the checkpoint does not state: 28 layers carry the
    preceding full layer's selection forward.
    """

    rope: tuple[bool, ...]
    owns_indexer: tuple[bool, ...]
    indexer_source: tuple[int, ...]
    kv_source: tuple[int, ...]
    moe: tuple[bool, ...]

    @property
    def n_layers(self) -> int:
        """The number of layers these tuples describe."""
        return len(self.rope)

    @property
    def nope_layers(self) -> tuple[int, ...]:
        """Indices of the layers with no rotary embedding."""
        return tuple(i for i, on in enumerate(self.rope) if not on)

    @property
    def indexer_owner_layers(self) -> tuple[int, ...]:
        """Indices of the layers that own a DSA indexer."""
        return tuple(i for i, on in enumerate(self.owns_indexer) if on)

    @property
    def indexer_sharer_layers(self) -> tuple[int, ...]:
        """Indices of the layers that reuse another layer's top-k selection."""
        return tuple(i for i, on in enumerate(self.owns_indexer) if not on)

    @property
    def kv_owner_layers(self) -> tuple[int, ...]:
        """Indices of the layers that own an MLA latent cache."""
        return tuple(i for i, src in enumerate(self.kv_source) if src == i)

    @property
    def kv_sharer_layers(self) -> tuple[int, ...]:
        """Indices of the layers that read another layer's latent cache."""
        return tuple(i for i, src in enumerate(self.kv_source) if src != i)


@dataclass(frozen=True, kw_only=True)
class MistralLarge4Quantization:
    """ModelOpt per-module quantization, resolved by longest prefix.

    ``modules`` holds the checkpoint's own module paths, so resolution happens
    on checkpoint names rather than on the module names
    :mod:`.weight_adapters` derives.
    """

    method: str
    algo: str
    group_size: int
    modules: Mapping[str, str]
    quantized: bool = True

    def algo_for(self, module: str) -> str:
        """Returns the quantization algorithm covering ``module``.

        Args:
            module: A checkpoint module path, e.g. ``layers.1.experts.0.w1``.

        Returns:
            One of the ``QUANT_ALGO_*`` values, or ``"BF16"`` when no entry
            covers the module.
        """
        best = ""
        for prefix in self.modules:
            if (module == prefix or module.startswith(prefix + ".")) and len(
                prefix
            ) > len(best):
                best = prefix
        return self.modules[best] if best else "BF16"


@dataclass(frozen=True, kw_only=True)
class MistralLarge4Mtp:
    """The in-checkpoint DSpark drafter's declared geometry (M3 work)."""

    n_layers: int
    depth: int
    window: tuple[int, ...]
    target_layers: tuple[int, ...]
    markov_rank: int
    head_type: str
    """``rnn``: the gated recurrent head this module implements."""
    draft_input: str
    """``position_specific``: one learned mask row per intra-block slot."""
    norm_strategy: str
    """``eagle``: pre-norm the block input with layer 0's, then the next
    layer's, ending with ``final_norm``."""


MTP_HEAD_TYPE_RNN = "rnn"
"""The only head this module implements: ``joint_proj -> gate|candidate|
re-emitted`` with a state that resets per block."""

MTP_DRAFT_INPUT_POSITION_SPECIFIC = "position_specific"
"""The only block input this module implements: the anchor embedding plus
``depth - 1`` learned mask rows, one per intra-block slot."""

MTP_NORM_STRATEGY_EAGLE = "eagle"
"""The only norm schedule this module implements: the block input is pre-normed
with layer 0's ``attention_norm``, each layer re-norms the residual with the
next layer's, and the last re-norm is ``final_norm``."""


@dataclass(frozen=True, kw_only=True)
class MistralLarge4NormalizedConfig:
    """Mistral Large 4's model geometry, normalized from ``params.json``."""

    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    kv_lora_rank: int
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    index_norm_eps: float
    """Epsilon of the indexer's two LayerNorms (``dsa.norm_eps``)."""
    final_logit_softcapping: float
    tie_word_embeddings: bool
    ssmax_scaling: bool
    yarn: MistralLarge4Yarn
    layers: MistralLarge4LayerSchedule
    quantization: MistralLarge4Quantization
    # Absent on a minimized fixture too short to host the drafter's five target
    # taps; M1's trunk never reads it.
    mtp: MistralLarge4Mtp | None
    image_token_id: int
    vision_layers: int
    vision_hidden_size: int

    @property
    def qk_head_dim(self) -> int:
        """The full query/key head width: nope and rope slices summed."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attention_scale(self) -> float:
        """The explicit softmax scale every layer uses, NoPE included."""
        return self.qk_head_dim**-0.5

    @staticmethod
    def from_params_json(
        params: Mapping[str, object],
    ) -> MistralLarge4NormalizedConfig:
        """Resolves a checkpoint ``params.json`` into the normalized config.

        Args:
            params: The parsed ``params.json``.

        Returns:
            The normalized config, with every per-layer pattern expanded.

        Raises:
            ValueError: If a per-layer pattern does not cover ``n_layers``, a
                KV-sharing source is not an earlier owning layer, or a
                quantization entry names an unimplemented algorithm.
        """
        n_layers = _int(params, "n_layers")
        dsa = _mapping(params["dsa"], "dsa")
        moe = _mapping(params["moe"], "moe")
        yarn = _mapping(params["yarn"], "yarn")
        vision = _mapping(params["vision_encoder"], "vision_encoder")
        mtp_raw = params.get("mtp")
        if mtp_raw is None:
            mtp = None
        else:
            mtp_block = _mapping(mtp_raw, "mtp")
            dspark = _mapping(mtp_block["dspark"], "mtp.dspark")
            mtp = MistralLarge4Mtp(
                n_layers=_int(mtp_block, "n_layers"),
                depth=_int(mtp_block, "depth"),
                window=_int_tuple(mtp_block, "ragged_attention"),
                target_layers=_int_tuple(mtp_block, "target_activation_layers"),
                markov_rank=_int(dspark, "markov_rank"),
                head_type=_str(dspark, "head_type"),
                draft_input=_str(dspark, "draft_input"),
                norm_strategy=_str(mtp_block, "norm_strategy"),
            )

        rope = _parse_rope_cycle(params["nope"], n_layers)
        owns_indexer = _parse_cycle(
            dsa["sharing"], _INDEXER_TO_FLAG, n_layers, "dsa.sharing"
        )
        kv_sharing = _mapping(params["kv_sharing"], "kv_sharing")
        kv_source = _parse_kv_source(kv_sharing["source_layers_str"], n_layers)
        first_k_dense = _int(moe, "first_k_dense_replace")
        qk_rope_head_dim = _int(params, "qk_rope_head_dim")

        return MistralLarge4NormalizedConfig(
            hidden_size=_int(params, "dim"),
            n_layers=n_layers,
            n_heads=_int(params, "n_heads"),
            n_kv_heads=_int(params, "n_kv_heads"),
            vocab_size=_int(params, "vocab_size"),
            max_position_embeddings=_int(params, "max_position_embeddings"),
            rms_norm_eps=_float(params, "norm_eps"),
            intermediate_size=_int(params, "hidden_dim"),
            moe_intermediate_size=_int(moe, "expert_hidden_dim"),
            n_routed_experts=_int(moe, "num_experts"),
            n_shared_experts=_int(moe, "num_shared_experts"),
            num_experts_per_tok=_int(moe, "num_experts_per_tok"),
            first_k_dense_replace=first_k_dense,
            routed_scaling_factor=_float(moe, "routed_scale"),
            n_group=_int(moe, "num_expert_groups"),
            topk_group=_int(moe, "num_expert_groups_per_tok"),
            kv_lora_rank=_int(params, "kv_lora_rank"),
            q_lora_rank=(
                None
                if params["q_lora_rank"] is None
                else _int(params, "q_lora_rank")
            ),
            qk_nope_head_dim=_int(params, "qk_nope_head_dim"),
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=_int(params, "v_head_dim"),
            index_topk=_int(dsa, "topk"),
            index_n_heads=_int(dsa, "n_heads"),
            index_head_dim=_int(dsa, "head_dim"),
            index_norm_eps=_float(dsa, "norm_eps"),
            final_logit_softcapping=_float(params, "softmax_tanh_gating"),
            tie_word_embeddings=bool(params["tied_embeddings"]),
            ssmax_scaling=_parse_ssmax_scaling(params, dsa),
            yarn=MistralLarge4Yarn(
                rope_theta=_float(params, "rope_theta"),
                factor=_float(yarn, "factor"),
                original_max_position_embeddings=_int(
                    yarn, "original_max_position_embeddings"
                ),
                beta_fast=_float(yarn, _YARN_BETA_FAST_KEY),
                beta_slow=_float(yarn, _YARN_BETA_SLOW_KEY),
                # `apply_scale: false`: no magnitude scaling, so no mscale.
                attention_factor=1.0,
                lerope=bool(params["lerope"]),
                lerope_alpha_dim=qk_rope_head_dim // 2,
            ),
            layers=MistralLarge4LayerSchedule(
                rope=rope,
                owns_indexer=owns_indexer,
                indexer_source=_indexer_source_layers(owns_indexer),
                kv_source=kv_source,
                moe=tuple(i >= first_k_dense for i in range(n_layers)),
            ),
            quantization=_parse_quantization(params.get("quantization_config")),
            mtp=mtp,
            image_token_id=_int(vision, "image_token_id"),
            vision_layers=_int(vision, "num_hidden_layers"),
            vision_hidden_size=_int(vision, "hidden_size"),
        )


def _int(source: Mapping[str, object], key: str) -> int:
    """Reads ``source[key]`` as an int, rejecting a float or a non-number."""
    value = source[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"params.json {key} must be an integer, got {value!r}")
    return value


def _float(source: Mapping[str, object], key: str) -> float:
    """Reads ``source[key]`` as a float, accepting a JSON integer."""
    value = source[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"params.json {key} must be a number, got {value!r}")
    return float(value)


def _str(source: Mapping[str, object], key: str) -> str:
    """Reads ``source[key]`` as a string."""
    value = source[key]
    if not isinstance(value, str):
        raise ValueError(f"params.json {key} must be a string, got {value!r}")
    return value


def _int_tuple(source: Mapping[str, object], key: str) -> tuple[int, ...]:
    """Reads ``source[key]``'s comma-separated value as ints.

    ``params.json`` writes these as strings (``"2048,2048,2048"``) and the
    converter splits them the same way; a rewritten fixture may carry the list
    directly, which reads the same.
    """
    return tuple(int(entry) for entry in _as_strings(source[key], key))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """Returns ``value`` as a mapping, or raises naming ``name``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"params.json {name} must be an object")
    return value


def _as_strings(value: object, name: str) -> tuple[str, ...]:
    """Reads a per-layer pattern, as a comma-separated string or a list.

    The published checkpoints write one comma-separated string per pattern; a
    minimized fixture may write the expanded list instead, which must resolve
    to the same schedule.
    """
    if isinstance(value, str):
        return tuple(value.split(","))
    if isinstance(value, Sequence):
        return tuple(str(entry) for entry in value)
    raise ValueError(
        f"params.json {name} must be a comma-separated string or a list, got "
        f"{value!r}"
    )


def _parse_rope_cycle(value: object, n_layers: int) -> tuple[bool, ...]:
    """Expands the NoPE cycle, which is spelled ``rope|nope`` or ``1|0``."""
    entries = _as_strings(value, "nope")
    flags: list[bool] = []
    for entry in entries:
        if entry in _ROPE_TO_FLAG:
            flags.append(_ROPE_TO_FLAG[entry])
        elif entry in ("0", "1"):
            flags.append(entry == "1")
        else:
            raise ValueError(f"nope has unknown entry {entry!r}")
    if n_layers % len(flags) != 0:
        raise ValueError(
            f"nope has {len(flags)} entries, which does not divide "
            f"n_layers={n_layers}"
        )
    return tuple(flags * (n_layers // len(flags)))


def _parse_cycle(
    value: object,
    flags: Mapping[str, bool],
    n_layers: int,
    name: str,
) -> tuple[bool, ...]:
    """Expands a repeating ``params.json`` pattern to one flag per layer."""
    entries = _as_strings(value, name)
    unknown = sorted(set(entries) - set(flags))
    if unknown:
        raise ValueError(f"{name} has unknown entries {unknown}")
    if n_layers % len(entries) != 0:
        raise ValueError(
            f"{name} has {len(entries)} entries, which does not divide "
            f"n_layers={n_layers}"
        )
    return tuple(flags[entry] for entry in entries * (n_layers // len(entries)))


def _indexer_source_layers(owns_indexer: Sequence[bool]) -> tuple[int, ...]:
    """Maps each layer to the layer whose top-k selection it uses.

    The checkpoint marks 28 layers ``shared`` without naming the selection they
    reuse. Owners and pairs of sharers alternate, so a sharer carries the
    preceding owner's selection forward; a sharer before any owner has nothing
    to carry and is rejected rather than defaulted.
    """
    out: list[int] = []
    last_owner = -1
    for layer, owns in enumerate(owns_indexer):
        if owns:
            last_owner = layer
        if last_owner < 0:
            raise ValueError(
                f"layer {layer} shares an indexer top-k with no earlier owner"
            )
        out.append(last_owner)
    return tuple(out)


def _parse_ssmax_scaling(
    params: Mapping[str, object], dsa: Mapping[str, object]
) -> bool:
    """Reads SSMax from both places ``params.json`` states it.

    The checkpoint repeats the flag at the top level and under ``dsa``. Neither
    is documented as authoritative, so a disagreement is a re-export this
    module has not seen rather than a value to silently pick between. A
    ``params.json`` that states it only at the top level is read without
    complaint.

    Args:
        params: The parsed ``params.json``.
        dsa: Its ``dsa`` object.

    Returns:
        The flag, which both spellings agree on.

    Raises:
        ValueError: If the two spellings disagree.
    """
    top_level = bool(params["ssmax_scaling"])
    nested = dsa.get("ssmax_scaling")
    if nested is not None and bool(nested) != top_level:
        raise ValueError(
            f"params.json ssmax_scaling={top_level} disagrees with "
            f"dsa.ssmax_scaling={nested}"
        )
    return top_level


def _parse_kv_source(value: object, n_layers: int) -> tuple[int, ...]:
    """Parses ``kv_sharing.source_layers_str`` and validates every source."""
    sources = [int(entry) for entry in _as_strings(value, "source_layers_str")]
    if len(sources) != n_layers:
        raise ValueError(
            f"kv_sharing.source_layers_str has {len(sources)} entries, "
            f"expected n_layers={n_layers}"
        )
    for layer, source in enumerate(sources):
        if source < 0:
            raise ValueError(
                f"layer {layer} shares KV with a negative layer {source}"
            )
        if source > layer:
            raise ValueError(
                f"layer {layer} shares KV with a later layer {source}"
            )
        if source != layer and sources[source] != source:
            raise ValueError(
                f"layer {layer} shares KV with layer {source}, which does not "
                "own its cache"
            )
    return tuple(sources)


def _parse_quantization(value: object) -> MistralLarge4Quantization:
    """Resolves ``params.json``'s ModelOpt ``quantization_config``.

    Args:
        value: The ``quantization_config`` object.

    Returns:
        The resolved per-module quantization; ``quantized=False`` when the
        checkpoint declares no ModelOpt mix (the base FP8 checkpoint).

    Raises:
        ValueError: If an entry names an algorithm this module does not
            implement.
    """
    if value is None:
        # The base FP8 checkpoint declares no ModelOpt mix. Its 2-D
        # `qscale_weight` scales are a separate loader; nothing in the
        # normalized metadata depends on it, so the config stays readable.
        return MistralLarge4Quantization(
            method="", algo="", group_size=0, modules={}, quantized=False
        )
    config = _mapping(value, "quantization_config")
    layers = _mapping(
        config["quantized_layers"], "quantization_config.quantized_layers"
    )
    modules: dict[str, str] = {}
    for module, entry in layers.items():
        algo = str(_mapping(entry, f"quantized_layers.{module}")["quant_algo"])
        if algo not in (QUANT_ALGO_FP8_PB_WO, QUANT_ALGO_NVFP4):
            raise ValueError(
                f"quantized_layers.{module} uses unknown quant_algo {algo!r}"
            )
        modules[module] = algo
    return MistralLarge4Quantization(
        method=str(config["quant_method"]),
        algo=str(config["quant_algo"]),
        group_size=_int(config, "group_size"),
        modules=modules,
    )


def read_params_json(path: str | Path) -> dict[str, object]:
    """Reads a checkpoint's ``params.json``.

    Args:
        path: Path to the file.

    Returns:
        The parsed JSON object.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def normalized_hf_config(
    config: MistralLarge4NormalizedConfig,
    *,
    params: Mapping[str, object],
    bos_token_id: int,
    eos_token_id: int,
) -> dict[str, object]:
    """Builds the HF-style ``config.json`` from ``params.json``.

    The output reproduces ``convert_mistral_large4_to_hf.py`` applied to
    ``params.json`` and then serialized by ``MistralLarge4Config``: the same
    keys, the same values, with the class defaults the converter inherits
    (``attention_bias``, ``initializer_range``, ``use_cache``, the vision
    tower's geometry) spelled out. Running it on the base checkpoint's
    ``params.json`` reproduces that repo's own published ``config.json``
    exactly, which is how the reproduction is verified.

    Deliberate differences from the converter's output, for the NVFP4 repo:

    * ``quantization_config`` is absent. The converter only synthesizes an FP8
      block config from ``params.json``'s ``quantization`` block, which this
      checkpoint does not have; its ModelOpt mixture is described by
      ``params.json``'s ``quantization_config`` (a different schema, with a
      per-module map) and is not expressible in HF's ``quantization_config``.
      MAX reads the ModelOpt mix per site instead.
    * ``bos_token_id``/``eos_token_id`` come from the base repo's tokenizer,
      because the NVFP4 repo ships no tokenizer files. The converter reads them
      off ``tekken.json``; the values agree with both repos'
      ``generation_config.json``.

    Args:
        config: The normalized config.
        params: The raw ``params.json``, read for the vision tower's fields and,
            on the base checkpoint, the FP8 ``quantization`` block.
        bos_token_id: Beginning-of-sequence token id.
        eos_token_id: End-of-sequence token id.

    Returns:
        The config as a JSON-serializable dict.
    """
    vision = _mapping(params["vision_encoder"], "vision_encoder")
    text_config: dict[str, object] = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "final_logit_softcapping": config.final_logit_softcapping,
        "first_k_dense_replace": config.first_k_dense_replace,
        # The rope slice's width: `MistralLarge4TextConfig.__post_init__`
        # repoints `head_dim` at it, so the ramp sizes 32 frequencies.
        "head_dim": config.qk_rope_head_dim,
        "hidden_act": "silu",
        "hidden_size": config.hidden_size,
        "index_head_dim": config.index_head_dim,
        "index_n_heads": config.index_n_heads,
        "index_topk": config.index_topk,
        "indexer_norm_eps": config.index_norm_eps,
        "indexer_types": [
            "full" if owns else "shared" for owns in config.layers.owns_indexer
        ],
        "initializer_range": 0.02,
        "intermediate_size": config.intermediate_size,
        "kv_lora_rank": config.kv_lora_rank,
        "kv_source_layers": list(config.layers.kv_source),
        "layer_types": ["indexed_attention"] * config.n_layers,
        "max_position_embeddings": config.max_position_embeddings,
        "mlp_layer_types": [
            "sparse" if is_moe else "dense" for is_moe in config.layers.moe
        ],
        "model_type": "mistral_large4_text",
        "moe_intermediate_size": config.moe_intermediate_size,
        "n_group": config.n_group,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "no_rope_layers": [1 if rope else 0 for rope in config.layers.rope],
        "num_attention_heads": config.n_heads,
        "num_experts_per_tok": config.num_experts_per_tok,
        "num_hidden_layers": config.n_layers,
        "num_key_value_heads": config.n_kv_heads,
        "output_router_logits": False,
        "pad_token_id": None,
        "qk_head_dim": config.qk_head_dim,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_parameters": {
            "attention_factor": config.yarn.attention_factor,
            "beta_fast": config.yarn.beta_fast,
            "beta_slow": config.yarn.beta_slow,
            "factor": config.yarn.factor,
            "original_max_position_embeddings": (
                config.yarn.original_max_position_embeddings
            ),
            "rope_theta": config.yarn.rope_theta,
            "rope_type": "yarn",
        },
        "routed_scaling_factor": config.routed_scaling_factor,
        "tie_word_embeddings": config.tie_word_embeddings,
        "topk_group": config.topk_group,
        "use_cache": True,
        "v_head_dim": config.v_head_dim,
        "vocab_size": config.vocab_size,
    }
    no_rope_interval = _no_rope_interval(config)
    if no_rope_interval is not None:
        text_config["no_rope_layer_interval"] = no_rope_interval

    out: dict[str, object] = {
        "architectures": ["MistralLarge4ForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_index": config.image_token_id,
        "model_type": "mistral_large4",
        "multimodal_projector_bias": bool(vision["adapter_bias"]),
        "projector_hidden_act": _VISION_PROJECTOR_HIDDEN_ACT,
        "spatial_merge_size": _int(vision, "spatial_merge_size"),
        "text_config": text_config,
        "tie_word_embeddings": config.tie_word_embeddings,
        "transformers_version": _HF_TRANSFORMERS_VERSION,
        "transformers_weights": _WEIGHTS_INDEX_NAME,
        "vision_config": {
            "attention_dropout": 0.0,
            "head_dim": config.vision_hidden_size
            // _int(vision, "num_attention_heads"),
            "hidden_act": "silu",
            "hidden_size": config.vision_hidden_size,
            "image_size": _int(vision, "image_size"),
            "initializer_range": 0.02,
            "intermediate_size": _int(vision, "intermediate_size"),
            "model_type": "pixtral",
            "num_attention_heads": _int(vision, "num_attention_heads"),
            "num_channels": _int(vision, "num_channels"),
            "num_hidden_layers": config.vision_layers,
            "patch_size": _int(vision, "patch_size"),
            "rope_parameters": {
                "rope_theta": _float(vision, "rope_theta"),
                "rope_type": "axial",
            },
        },
        "vision_feature_layer": _VISION_FEATURE_LAYER,
    }
    quantization = params.get("quantization")
    if quantization is not None:
        out["quantization_config"] = _fp8_quantization_config(
            _mapping(quantization, "quantization")
        )
    return out


def _no_rope_interval(config: MistralLarge4NormalizedConfig) -> int | None:
    """Returns the NoPE cycle length, or ``None`` when there is no cycle.

    ``no_rope_layer_interval`` only decorates the per-layer list, which this
    generator always writes, so a fixture with fewer than two NoPE layers (or a
    non-periodic schedule) simply omits it.
    """
    nope = config.layers.nope_layers
    if len(nope) < 2:
        return None
    interval = nope[1] - nope[0]
    if any(later - earlier != interval for earlier, later in pairwise(nope)):
        return None
    return interval


def _fp8_quantization_config(
    quantization: Mapping[str, object],
) -> dict[str, object]:
    """Reproduces the converter's FP8 ``quantization_config`` synthesis.

    Only the base checkpoint carries the ``quantization`` block this reads; the
    NVFP4 checkpoint's ModelOpt mixture has no HF spelling.
    """
    qformat_weight = str(quantization["qformat_weight"])
    if qformat_weight != "fp8_e4m3":
        raise ValueError(f"unsupported qformat_weight {qformat_weight!r}")
    qscheme_act = str(quantization["qscheme_act"])
    activation_scheme = {
        "BLOCK": "dynamic",
        "NO_SCALES": "dynamic",
        "TENSOR": "static",
    }[qscheme_act]
    modules_to_not_convert = [
        "model.vision_tower",
        "model.multi_modal_projector",
        "lm_head",
        "kv_b_proj",
        "indexer.wk",
        "indexer.weights_proj",
    ]
    if not bool(quantization["quantize_kv_a"]):
        modules_to_not_convert.append("kv_a_proj_with_mqa")
    return {
        "activation_scheme": activation_scheme,
        "modules_to_not_convert": modules_to_not_convert,
        "quant_method": "fp8",
        "weight_block_size": [
            _int(quantization, "block_size_out"),
            _int(quantization, "block_size_in"),
        ],
    }


def config_json_text(config: Mapping[str, object]) -> str:
    """Returns the on-disk spelling of a generated ``config.json``.

    One spelling, so a config derived on load from a Hub repo and one written
    by this module's CLI into a staged directory are the same bytes.

    Args:
        config: The config to serialize.

    Returns:
        The file's text.
    """
    return json.dumps(config, indent=2, sort_keys=True) + "\n"


def write_hf_config(
    params_path: str | Path,
    out_path: str | Path,
    *,
    bos_token_id: int = _BOS_TOKEN_ID,
    eos_token_id: int = _EOS_TOKEN_ID,
) -> dict[str, object]:
    """Generates the normalized ``config.json`` beside a checkpoint.

    Args:
        params_path: The checkpoint's ``params.json``.
        out_path: Where to write the generated config.
        bos_token_id: Beginning-of-sequence token id.
        eos_token_id: End-of-sequence token id.

    Returns:
        The config written.
    """
    params = read_params_json(params_path)
    config = normalized_hf_config(
        MistralLarge4NormalizedConfig.from_params_json(params),
        params=params,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    Path(out_path).write_text(config_json_text(config), encoding="utf-8")
    return config


def derive_hf_config(params_path: str | Path) -> dict[str, object] | None:
    """Builds a checkpoint's ``config.json`` from its ``params.json``.

    The generator MAX's config loader runs for a Mistral Large 4 repo that ships
    no ``config.json`` -- the NVFP4 release ships ``params.json`` and weights
    only -- and the one this module's CLI writes out for a prepared directory.
    One function, so a config derived on load and one staged by hand carry the
    same fields. The caller decides whether any of this reaches disk; a Hub
    checkpoint is a read-only cache snapshot that no load may write into.

    Args:
        params_path: The path to a checkpoint's ``params.json``, which may or
            may not belong to this family, and may not be readable as one.

    Returns:
        The config, or ``None`` when this is not a Mistral Large 4 checkpoint.
    """
    try:
        params = read_params_json(params_path)
    except (OSError, ValueError):
        # Another family's file of another shape, or not a JSON object at all:
        # this is not a shape this family states its geometry in.
        return None
    if not _PARAMS_SIGNATURE <= params.keys():
        return None
    return normalized_hf_config(
        MistralLarge4NormalizedConfig.from_params_json(params),
        params=params,
        bos_token_id=_BOS_TOKEN_ID,
        eos_token_id=_EOS_TOKEN_ID,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generates an ML4 ``config.json`` from a checkpoint's ``params.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params_json", help="The checkpoint's params.json")
    parser.add_argument(
        "--out", required=True, help="Where to write config.json"
    )
    parser.add_argument("--bos-token-id", type=int, default=_BOS_TOKEN_ID)
    parser.add_argument("--eos-token-id", type=int, default=_EOS_TOKEN_ID)
    args = parser.parse_args(argv)
    write_hf_config(
        args.params_json,
        args.out,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
