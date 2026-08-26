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
"""Configuration dataclass for the Mamba2 SSD architecture.

This is the minimal "shape & policy" config consumed by the Mamba2
weight adapter and the NN modules in :mod:`.mamba2`. It deliberately
does NOT subclass :class:`max.pipelines.lib.MAXModelConfig` /
:class:`ArchConfigWithKVCache` — that integration lands in a later
RFC 0003 item that wires the full pipeline (graph build + cache mgr).
For now the dataclass is the source of truth for:

* the Mamba2 reference defaults (mirrors
  ``mamba_ssm/models/config_mamba.py``), and
* the HF-checkpoint loader that bridges
  :class:`transformers.Mamba2Config` (a.k.a. ``model_type="mamba2"``)
  into our shape vocabulary.

Two-name reality of "Mamba2 config" upstream:

1. ``mamba_ssm.models.config_mamba.MambaConfig`` (reference repo): uses
   ``d_model``, ``n_layer``, ``ssm_cfg`` dict for the mixer knobs
   (``d_state``, ``d_conv``, ``expand``, ``headdim``, ``ngroups``,
   ``chunk_size``).
2. ``transformers.Mamba2Config``: uses ``hidden_size``,
   ``num_hidden_layers``, and exposes the mixer knobs as flat fields
   (``state_size``, ``conv_kernel``, ``expand``, ``head_dim``,
   ``n_groups``, ``chunk_size``). Field naming follows HF conventions
   (``num_heads``, ``head_dim``, ``n_groups``).

:meth:`Mamba2Config.from_huggingface` accepts either flavour and emits
a single canonical Mamba2Config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.lib import upper_bounded_default
from transformers import AutoConfig
from typing_extensions import Self

if TYPE_CHECKING:
    from max.nn.kv_cache.cache_params import KVCacheParamInterface
    from max.pipelines.lib import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig

logger_name = "max.pipelines.mamba2"

# Re-derive the shape tuple from the NN module so config-time validation
# matches what construction would do.
from .mamba2 import mamba2_dims


@dataclass
class Mamba2Config:
    """Minimal Mamba2 shape + policy config.

    The fields mirror ``mamba_ssm.models.config_mamba.MambaConfig`` with
    the mixer-specific keys pulled out of ``ssm_cfg`` into top-level
    fields. Defaults match the reference repo so a bare
    ``Mamba2Config(d_model=..., n_layer=..., vocab_size=...)`` produces
    the same mixer shapes the reference uses for the 2.7B/2.8B SSM
    checkpoints (``d_state=128, d_conv=4, expand=2, headdim=64,
    ngroups=1, chunk_size=256``).

    Constraints enforced at construction time
    (``mamba2_dims`` chokepoint):

    * ``d_inner = expand * d_model`` must be divisible by ``headdim``.
    * ``nheads = d_inner // headdim``.
    * ``ngroups`` must divide ``nheads`` (the SSD kernel currently
      requires ``ngroups == nheads`` — :class:`Mamba2Mixer` enforces
      that stricter rule; the config-level check is the looser
      gcd-divisibility one so adapters that broadcast can still
      construct the config).
    """

    # -- core architecture ------------------------------------------------
    d_model: int
    n_layer: int
    vocab_size: int
    pad_vocab_size_multiple: int = 8

    # -- Mamba2 mixer knobs (mirrors `ssm_cfg` defaults) ------------------
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    # Original ``n_groups`` value from the source HF config, BEFORE the
    # ``from_hf_config`` override that bumps ``ngroups`` up to ``nheads``.
    # The weight adapter needs the original value to correctly compute
    # the HF in_proj row layout (``2*d_mlp + 2*d_inner + 2*src_ngroups*d_state + nheads``)
    # before broadcasting B/C to the kernel's ``ngroups == nheads`` ABI.
    # Defaults to ``None`` for configs constructed without the HF loader;
    # in that case the adapter falls back to ``ngroups``.
    source_ngroups: int | None = None
    chunk_size: int = 256

    # -- normalization / dtype policy -------------------------------------
    rms_norm_eps: float = 1e-5
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    tie_embeddings: bool = True

    # -- mixer toggles (not in reference config; pulled from the mixer's
    # constructor defaults so the dataclass is the single source of truth
    # for everything the adapter / NN module needs) ----------------------
    use_bias: bool = False
    use_conv_bias: bool = True

    # -- init knobs (reference defaults; not consumed at runtime from HF
    # — we mirror the reference values so a future trainer can read the
    # config without a second source). ----------------------------------
    A_init_range: tuple[float, float] = (1.0, 16.0)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    # -- derived dims (filled in __post_init__) ---------------------------
    d_inner: int = field(init=False)
    nheads: int = field(init=False)
    d_in_proj: int = field(init=False)
    conv_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # `mamba2_dims` validates `d_inner % headdim == 0` and the
        # ngroups/nheads divisibility rule, raising a `ValueError`
        # otherwise. Surface the same error here so a malformed config
        # fails fast instead of at module-construction time.
        dims = mamba2_dims(
            self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            ngroups=self.ngroups,
        )
        self.d_inner = dims["d_inner"]
        self.nheads = dims["nheads"]
        self.d_in_proj = dims["d_in_proj"]
        self.conv_dim = dims["conv_dim"]
        self._validate_config()

    def _validate_config(self) -> None:
        """Lightweight self-check on derived dims.

        Catches a class of subtle bugs where the dataclass is constructed
        by hand and a derived field is later mutated to an inconsistent
        value. Kept as a one-liner so the cost is negligible.
        """
        expected_d_inner = self.expand * self.d_model
        if self.d_inner != expected_d_inner:
            raise ValueError(
                f"d_inner ({self.d_inner}) does not match expand*d_model "
                f"({expected_d_inner})"
            )
        expected_nheads = self.d_inner // self.headdim
        if self.nheads != expected_nheads:
            raise ValueError(
                f"nheads ({self.nheads}) does not match d_inner//headdim "
                f"({expected_nheads})"
            )
        expected_d_in_proj = (
            2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        )
        if self.d_in_proj != expected_d_in_proj:
            raise ValueError(
                f"d_in_proj ({self.d_in_proj}) does not match "
                f"2*d_inner + 2*ngroups*d_state + nheads "
                f"({expected_d_in_proj})"
            )
        expected_conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        if self.conv_dim != expected_conv_dim:
            raise ValueError(
                f"conv_dim ({self.conv_dim}) does not match "
                f"d_inner + 2*ngroups*d_state ({expected_conv_dim})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0 (got {self.vocab_size})")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be > 0 (got {self.n_layer})")
        # The SSD kernel currently assumes ngroups == nheads (item-3
        # constraint); the broader gcd rule is already enforced by
        # `mamba2_dims`, but warn loudly here so a config with
        # ngroups < nheads doesn't silently slip into the adapter — the
        # adapter has to broadcast B/C and a non-tiled checkpoint would
        # surface as a shape mismatch much later.
        if (
            self.ngroups != self.nheads
            and math.gcd(self.nheads, self.ngroups) != self.ngroups
        ):
            raise ValueError(
                f"ngroups ({self.ngroups}) must divide nheads ({self.nheads}); "
                "non-divisible groups are not supported by the SSD kernel."
            )

    @property
    def padded_vocab_size(self) -> int:
        """``vocab_size`` rounded up to ``pad_vocab_size_multiple``.

        Mirrors the reference's vocab-padding behaviour (used when
        constructing the embedding/lm_head). Returned as a property so
        the dataclass stays small and ``__post_init__`` doesn't have to
        re-validate when the field changes.
        """
        m = self.pad_vocab_size_multiple
        if m <= 0:
            return self.vocab_size
        return ((self.vocab_size + m - 1) // m) * m

    # -- HF loaders --------------------------------------------------------

    @classmethod
    def from_huggingface(
        cls, name_or_path: str, **overrides: Any
    ) -> Mamba2Config:
        """Load a :class:`Mamba2Config` from an HF model repo or local dir.

        Uses :func:`transformers.AutoConfig.from_pretrained` to load
        whatever config the checkpoint ships and then maps the known
        Mamba2 field names into our vocabulary.

        Two HF dialects are supported:

        * **transformers.Mamba2Config** (``model_type="mamba2"``): flat
          fields — ``hidden_size``, ``num_hidden_layers``, ``vocab_size``,
          ``num_heads``, ``head_dim``, ``state_size``, ``conv_kernel``,
          ``expand``, ``n_groups``, ``chunk_size``, ``layer_norm_epsilon``,
          ``residual_in_fp32``, ``tie_word_embeddings``,
          ``pad_token_id`` (no ``pad_vocab_size_multiple``).
        * **mamba_ssm reference config** (loaded as a generic
          ``PretrainedConfig`` when the repo ships the mamba_ssm
          variant): ``d_model``, ``n_layer``, ``ssm_cfg`` dict,
          ``rms_norm`` toggle, ``fused_add_norm``, ``tie_embeddings``,
          ``pad_vocab_size_multiple``.

        Args:
            name_or_path: HF Hub repo id or local directory path.
            **overrides: Any keyword args here override the values
                pulled from the HF config (useful for tests where the
                checkpoint disagrees with what we want to instantiate).

        Returns:
            A populated :class:`Mamba2Config`.
        """
        # Import here so test-only `Mamba2Config()` use doesn't require
        # transformers at module-import time.
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(name_or_path)
        return cls.from_hf_config(hf_cfg, **overrides)

    @classmethod
    def from_hf_config(cls, hf_cfg: Any, **overrides: Any) -> Mamba2Config:
        """Build a :class:`Mamba2Config` from an already-loaded HF config.

        Separated from :meth:`from_huggingface` so callers that already
        loaded an :class:`AutoConfig` (e.g. via the pipeline-config
        machinery) can reuse it without a second disk/network hit.
        """
        # ssm_cfg defaults: reference repo nests the mixer knobs in a
        # dict. transformers.Mamba2Config lifts them to flat fields.
        # Read both, with the flat fields winning when both are present.
        ssm_cfg: dict[str, Any] = dict(getattr(hf_cfg, "ssm_cfg", None) or {})

        # `d_model` (reference) vs `hidden_size` (transformers) — both
        # surface here; reference repo doesn't have `hidden_size` so the
        # getattr chain prefers the reference name and falls back.
        d_model = getattr(hf_cfg, "d_model", None) or getattr(
            hf_cfg, "hidden_size", None
        )
        if d_model is None:
            raise ValueError(
                "HF config is missing both `d_model` and `hidden_size`; "
                "cannot determine Mamba2 hidden width."
            )

        n_layer = getattr(hf_cfg, "n_layer", None) or getattr(
            hf_cfg, "num_hidden_layers", None
        )
        if n_layer is None:
            raise ValueError(
                "HF config is missing both `n_layer` and "
                "`num_hidden_layers`; cannot determine Mamba2 depth."
            )

        vocab_size = getattr(hf_cfg, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("HF config is missing `vocab_size`.")

        # Mixer knobs: try the transformers field name first, then the
        # ssm_cfg dict, then the reference field name, then the dataclass
        # default. The fallback chain captures both HF dialects without
        # branching on the model_type string (which not every repo
        # bothers to set correctly).
        def _pick(*names: str, default: Any) -> Any:
            for n in names:
                v = getattr(hf_cfg, n, None)
                if v is not None:
                    return v
                if n in ssm_cfg and ssm_cfg[n] is not None:
                    return ssm_cfg[n]
            return default

        d_state = int(_pick("state_size", "d_state", default=128))
        d_conv = int(_pick("conv_kernel", "d_conv", default=4))
        expand = int(_pick("expand", default=2))
        # transformers uses `head_dim`, reference uses `headdim`.
        headdim = int(_pick("head_dim", "headdim", default=64))
        # transformers uses `n_groups`, reference uses `ngroups`.
        # Override to ``nheads`` whenever the checkpoint ships
        # ``n_groups < nheads`` (the common HF case — AntonV/mamba2-130m-hf
        # uses ``n_groups=1`` while ``nheads=24``). The SSD kernel ABI
        # requires ``ngroups == nheads``; the weight adapter's
        # :func:`_broadcast_in_proj_BC` tiles B/C up to ``nheads`` so the
        # in_proj weight matches our mixer's parameter shape. Keeping
        # ``config.ngroups == nheads`` makes the model's ``d_in_proj``
        # consistent with the broadcasted weight tensor.
        nheads_for_groups = (expand * int(d_model)) // headdim
        raw_ngroups = int(_pick("n_groups", "ngroups", default=1))
        ngroups = max(raw_ngroups, nheads_for_groups)
        source_ngroups = raw_ngroups
        chunk_size = int(_pick("chunk_size", default=256))

        # Eps: transformers calls it `layer_norm_epsilon`, reference
        # plumbs it via `rms_norm` toggles (no explicit eps field — the
        # mixer hardcodes 1e-5). Use the explicit value if present.
        rms_norm_eps = float(
            _pick("layer_norm_epsilon", "rms_norm_eps", default=1e-5)
        )

        residual_in_fp32 = bool(_pick("residual_in_fp32", default=True))
        # transformers Mamba2Config doesn't surface `fused_add_norm`;
        # the reference does. Default to True (matches both the
        # reference and what `Mamba2Block` does when constructed with
        # default args).
        fused_add_norm = bool(_pick("fused_add_norm", default=True))

        # transformers uses `tie_word_embeddings`, reference uses
        # `tie_embeddings`. Reference Mamba2 defaults to True; HF
        # `transformers.Mamba2Config` defaults to False — pick whichever
        # the checkpoint actually set, falling back to True (the
        # state-spaces/mamba2 repos all ship with tied embeddings).
        # TODO(verify-vs-hf): the HF default differing from the
        # reference is the most likely source of a silent regression.
        # An integration test should load a real checkpoint and check
        # that `lm_head.weight` resolution matches.
        tie_embeddings = bool(
            _pick("tie_embeddings", "tie_word_embeddings", default=True)
        )

        # Mixer toggles — transformers exposes these directly, reference
        # uses ssm_cfg defaults that match.
        use_bias = bool(_pick("use_bias", "bias", default=False))
        use_conv_bias = bool(_pick("use_conv_bias", "conv_bias", default=True))

        pad_vocab_size_multiple = int(
            _pick("pad_vocab_size_multiple", default=8)
        )

        # Init knobs — reference exposes these in `ssm_cfg`; transformers
        # has flat `time_step_min` / `time_step_max` / `time_step_floor`.
        # TODO(verify-vs-hf): A_init_range isn't in transformers config;
        # we always use the reference default. If a future checkpoint
        # ships with a custom A init the adapter must read it from there
        # instead.
        dt_min = float(_pick("time_step_min", "dt_min", default=0.001))
        dt_max = float(_pick("time_step_max", "dt_max", default=0.1))
        dt_init_floor = float(
            _pick("time_step_floor", "dt_init_floor", default=1e-4)
        )
        a_init_range_raw = _pick("A_init_range", default=(1.0, 16.0))
        if (
            isinstance(a_init_range_raw, (list, tuple))
            and len(a_init_range_raw) == 2
        ):
            a_init_range = (
                float(a_init_range_raw[0]),
                float(a_init_range_raw[1]),
            )
        else:
            a_init_range = (1.0, 16.0)

        kwargs: dict[str, Any] = dict(
            d_model=int(d_model),
            n_layer=int(n_layer),
            vocab_size=int(vocab_size),
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            source_ngroups=source_ngroups,
            chunk_size=chunk_size,
            rms_norm_eps=rms_norm_eps,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            tie_embeddings=tie_embeddings,
            use_bias=use_bias,
            use_conv_bias=use_conv_bias,
            A_init_range=a_init_range,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class Mamba2ArchConfig(Mamba2Config):
    """Registry-facing :class:`Mamba2Config` that satisfies :class:`ArchConfigWithKVCache`.

    :class:`Mamba2Config` is intentionally a pure shape dataclass and does
    not implement the :class:`~max.pipelines.lib.interfaces.ArchConfig`
    protocol (full pipeline-config integration lands in a later RFC 0003
    item). The registry, however, calls ``arch.config.initialize()`` and
    ``arch_config.get_max_seq_len()`` to size the tokenizer. This thin
    subclass bridges that gap by delegating to
    :meth:`Mamba2Config.from_hf_config` for initialization and to the
    HF-config's ``max_position_embeddings`` (capped by the user's
    ``--max-length``) for the sequence-length estimate, matching the
    policy implemented by :meth:`calculate_max_seq_len`.
    """

    # Cached max sequence length plumbed through from `initialize` so
    # `get_max_seq_len` can return it without re-reading the pipeline
    # config. Not part of the shape-config surface — kept as a plain
    # field with a sentinel default for mypy.
    _max_seq_len: int = 0

    # Devices the model runs on. Required by the runtime-checkable
    # ``ModelConfig`` / ``ModelConfigWithKVCache`` protocols consumed by
    # :class:`~max.pipelines.kv_cache.memory_planner.PagedMemoryPlanner`
    # (which ``isinstance``-checks for a ``devices`` attribute). Mirrors
    # Mamba1's :class:`MambaConfig.devices`.
    devices: list[DeviceRef] = field(default_factory=list)

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Build the arch config from a :class:`PipelineConfig`.

        Reads the HF config off ``model_config.huggingface_config`` (or
        ``pipeline_config.model.huggingface_config`` when no explicit
        model config is given) and maps it through
        :meth:`Mamba2Config.from_hf_config`. ``max_seq_len`` is received
        from the caller (the memory plan's VRAM-clamped length, or the
        construction-resolved ``model_config.max_length``), never derived
        here; see :meth:`calculate_max_seq_len` for the bounding policy.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be "
                "loaded."
            )

        shape_cfg = Mamba2Config.from_hf_config(huggingface_config)

        n_devices = len(pipeline_config.model.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs[:n_devices]
        ]

        # Construct via the subclass with all of `Mamba2Config`'s
        # required fields. The derived fields (`d_inner`, `nheads`, ...)
        # are recomputed by `__post_init__` so they match `shape_cfg`.
        return cls(
            d_model=shape_cfg.d_model,
            n_layer=shape_cfg.n_layer,
            vocab_size=shape_cfg.vocab_size,
            pad_vocab_size_multiple=shape_cfg.pad_vocab_size_multiple,
            d_state=shape_cfg.d_state,
            d_conv=shape_cfg.d_conv,
            expand=shape_cfg.expand,
            headdim=shape_cfg.headdim,
            ngroups=shape_cfg.ngroups,
            source_ngroups=shape_cfg.source_ngroups,
            chunk_size=shape_cfg.chunk_size,
            rms_norm_eps=shape_cfg.rms_norm_eps,
            residual_in_fp32=shape_cfg.residual_in_fp32,
            fused_add_norm=shape_cfg.fused_add_norm,
            tie_embeddings=shape_cfg.tie_embeddings,
            use_bias=shape_cfg.use_bias,
            use_conv_bias=shape_cfg.use_conv_bias,
            A_init_range=shape_cfg.A_init_range,
            dt_min=shape_cfg.dt_min,
            dt_max=shape_cfg.dt_max,
            dt_init_floor=shape_cfg.dt_init_floor,
            _max_seq_len=int(max_seq_len),
            devices=device_refs,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bound ``max_length`` by ``max_position_embeddings``.

        Mamba2 has no positional embeddings, so any reasonable upper bound
        works; respect a configured ``max_position_embeddings`` if the
        checkpoint ships one, else default to 2048.
        """
        upper_bound = getattr(
            huggingface_config, "max_position_embeddings", 2048
        )
        try:
            return upper_bounded_default(
                upper_bound=upper_bound,
                default=model_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba2; "
                f"max_length ({model_config.max_length}) exceeds "
                f"max_position_embeddings ({upper_bound})."
            ) from e

    def get_max_seq_len(self) -> int:
        """Maximum sequence length plumbed from the pipeline config."""
        return self._max_seq_len

    def get_kv_params(self) -> KVCacheParamInterface:
        """Dummy KV-cache params so the pipeline allocator has a budget.

        Mamba2's real per-request state lives in :class:`Mamba2SSMStateCache`,
        not in a paged KV cache. But the pipeline's memory estimator runs
        ``_calculate_kv_cache_size`` only when the arch config satisfies
        :class:`ArchConfigWithKVCache`; otherwise it returns 0 and
        ``load_kv_manager`` then refuses to allocate a single page.
        Mirroring Mamba1's :class:`MambaConfig`, we expose a minimal stub
        here (n_kv_heads=1, head_dim=1, num_layers=1) so the allocator
        reserves a few KiB and the rest of the pipeline plumbing works.
        The actual SSM state pool is sized via
        :meth:`Mamba2MemoryPlanner.estimate_activation_memory` (the
        GatedDeltaNetStateCache pattern from qwen3_5).
        """
        return MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=self.devices or [DeviceRef.CPU()],
            page_size=128,
        )
