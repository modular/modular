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
"""MAX model config classes."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import constants as hf_hub_constants
from max.config import ConfigFileModel
from max.driver import DeviceSpec
from max.dtype import DType
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import (
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.context import SamplingParamsGenerationConfigDefaults
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    coerce_device_specs_input,
)
from max.pipelines.lib.weight_loader import (
    WeightLoader,
    _loader_over_weights,
    dict_loader,
)
from max.pipelines.modeling.config_enums import (
    RopeType,
    SupportedEncoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from max.pipelines.weights.weight_path_parser import WeightPathParser
from max.support.human_readable_formatter import to_human_readable_bytes
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)
from transformers import PretrainedConfig
from transformers.generation import GenerationConfig
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.pipeline_args import PipelineArgs

logger = logging.getLogger("max.pipelines")

# Encodings that can be casted to/from each other.
# We currently only support float32 <-> bfloat16 weight type casting.
_ALLOWED_CAST_ENCODINGS = {
    "float32",
    "bfloat16",
}


# ---------------------------------------------------------------------------
# Pure resolution helpers used by MAXModelConfig.__init__: they read config
# values without mutating and return the resolved values.
# ---------------------------------------------------------------------------


def _parse_weight_and_model_paths(
    *,
    model_path: str,
    weight_path: list[Path],
    subfolder: str | None,
    weights_repo_id: str | None,
) -> tuple[list[Path], str, str | None]:
    """Parses ``weight_path``/``model_path`` and applies subfolder prefixing.

    Returns:
        A ``(weight_path, model_path, weights_repo_id)`` tuple.
    """
    weight_path, parsed_repo_id = WeightPathParser.parse(
        model_path, weight_path
    )
    # Only overwrite a seeded weights_repo_id when the parser actually
    # extracts one.  When callers pass a bare filename (to avoid network
    # calls in WeightPathParser), the parser returns None and we must
    # keep the value seeded via __init__.
    if parsed_repo_id is not None:
        weights_repo_id = parsed_repo_id

    # When subfolder is set, user-provided weight paths are relative to
    # the subfolder.  Prepend the subfolder so that all downstream code
    # (encoding detection, validation, downloading) sees repo-relative
    # paths that include the subfolder prefix.
    #
    # Skip this when weights come from a different repo (parsed_repo_id
    # differs from model_path) — cross-repo weight paths are relative to
    # that external repo's root, not the base model's subfolder.
    weights_from_external_repo = (
        parsed_repo_id is not None and parsed_repo_id != model_path
    )
    if subfolder and weight_path and not weights_from_external_repo:
        prefix = subfolder + "/"
        adjusted: list[Path] = []
        for p in weight_path:
            if (
                not p.is_absolute()
                and not p.exists()
                and not str(p).startswith(prefix)
            ):
                adjusted.append(Path(subfolder) / p)
            else:
                adjusted.append(p)
        weight_path = adjusted

    # With an explicit weight_path but no model_path, derive model_path from
    # the parsed weights repo id.
    if weight_path and model_path == "" and weights_repo_id is not None:
        model_path = weights_repo_id

    return weight_path, model_path, weights_repo_id


def _resolve_dtype_cast(
    *,
    from_encoding: SupportedEncoding,
    to_encoding: SupportedEncoding,
    default_device_spec: DeviceSpec,
) -> tuple[SupportedEncoding | None, SupportedEncoding | None]:
    """Validates a dtype cast and returns the ``(from, to)`` bookkeeping.

    Returns ``(None, None)`` when ``from_encoding == to_encoding`` (no cast
    needed).

    Raises:
        ValueError: If the cast isn't an allowed direction, or ``to_encoding``
            isn't supported on ``default_device_spec``.
    """
    if from_encoding == to_encoding:
        return None, None
    elif not (
        from_encoding in _ALLOWED_CAST_ENCODINGS
        and to_encoding in _ALLOWED_CAST_ENCODINGS
    ):
        raise ValueError(
            f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{default_device_spec}'. "
            f"We only support float32 <-> bfloat16 weight type casting."
        )

    if not supported_encoding_supported_on(to_encoding, default_device_spec):
        raise ValueError(
            f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{default_device_spec}' because '{to_encoding}' is not supported on this device."
            f"Please use a different device or a different encoding."
        )
    return from_encoding, to_encoding


def _infer_quantization_encoding(
    config: MAXModelConfig,
) -> tuple[
    SupportedEncoding | None,
    SupportedEncoding | None,
    SupportedEncoding | None,
]:
    """Best-effort inference of ``quantization_encoding`` without architecture info.

    Returns:
        A ``(encoding, applied_dtype_cast_from, applied_dtype_cast_to)``
        tuple. The cast fields are ``None`` unless a float32->bfloat16 GPU
        cast was resolved. ``encoding`` is ``None`` when it cannot be
        unambiguously determined.
    """
    encoding = config.quantization_encoding
    cast_from: SupportedEncoding | None = None
    cast_to: SupportedEncoding | None = None

    if config.weight_path:
        # Try filename-based detection first.
        inferred = parse_supported_encoding_from_file_name(
            str(config.weight_path[0])
        )
        if inferred is None and not os.path.exists(config.weight_path[0]):
            # Remote file — ask the HF repo.
            inferred = config.huggingface_weight_repo.encoding_for_file(
                config.weight_path[0]
            )
        if inferred:
            encoding = inferred
    else:
        # No weight_path — check the repo's supported encodings.
        supported = config.huggingface_weight_repo.supported_encodings
        if len(supported) == 1:
            encoding = supported[0]
        elif (
            len(supported) > 1
            and config.default_device_spec.device_type != "cpu"
        ):
            # GPU preference: most-specific quantized format first,
            # matching _validate_and_resolve_without_given_quantization_encoding.
            if "float4_e2m1fnx2" in supported:
                encoding = "float4_e2m1fnx2"
            elif "float8_e4m3fn" in supported:
                encoding = "float8_e4m3fn"
            elif "bfloat16" in supported:
                encoding = "bfloat16"
            # else: ambiguous — leave as None for architecture to resolve.

    # On GPU, cast float32 → bfloat16 (the natural GPU dtype).
    if (
        encoding == "float32"
        and config.default_device_spec.device_type != "cpu"
    ):
        cast_from, cast_to = _resolve_dtype_cast(
            from_encoding="float32",
            to_encoding="bfloat16",
            default_device_spec=config.default_device_spec,
        )
        encoding = cast_to

    return encoding, cast_from, cast_to


def _infer_weight_path(
    config: MAXModelConfig,
    encoding: SupportedEncoding,
    cast_from: SupportedEncoding | None = None,
) -> list[Path]:
    """Best-effort discovery of weight files without architecture info.

    Takes *encoding* (and optionally *cast_from*) explicitly rather than
    reading ``config.quantization_encoding``/``config._applied_dtype_cast_from``
    so this can be called on a config those fields were never written to
    (e.g. a diffusion component resolved on demand at consumption time,
    without going through architecture-level resolution).

    Prefers safetensors format as default.

    Returns:
        The discovered weight files, or ``[]`` if none are found.
    """
    weight_files = config.huggingface_weight_repo.files_for_encoding(
        encoding=encoding
    )

    if not weight_files and cast_from:
        # We allow ourselves to load float32 safetensors weights as bfloat16.
        weight_files = config.huggingface_weight_repo.files_for_encoding(
            encoding=cast_from
        )

    if (
        not weight_files
        and config.subfolder is not None
        and encoding in ("float16", "bfloat16")
    ):
        # A float16/bfloat16 graph can load float32 weights cast at load
        # time by the component's weight adapter, which lets a
        # mixed-precision diffusion pipeline run (e.g. a bfloat16 text
        # encoder whose checkpoint ships float32 safetensors).
        #
        # Scoped to diffuser sub-components (``subfolder`` set): they skip
        # architecture validation, so this best-effort pass is their only
        # resolution step. Architecture-validated models (LLMs and
        # speculative-decoding draft models) must NOT bind weight_path to
        # the float32 checkpoint here -- the downstream given-encoding
        # validation would then flip quantization_encoding to float32 and
        # drop the requested bfloat16 (breaking e.g. Kimi-K2.6 Eagle3). For
        # those, the identical float32->bfloat16 fallback in
        # ``_resolve_weight_path`` runs after the cast bookkeeping is
        # recorded and resolves it correctly.
        weight_files = config.huggingface_weight_repo.files_for_encoding(
            encoding="float32"
        )

    # Prefer safetensors (reasonable default for diffuser components).
    if safetensors_files := weight_files.get(WeightsFormat.safetensors, []):
        return safetensors_files
    elif weight_files:
        # Fall back to any available format.
        return next(iter(weight_files.values()))
    return []


def _resolve_component_encoding_and_weights(
    config: MAXModelConfig,
) -> tuple[SupportedEncoding | None, list[Path]]:
    """Best-effort resolution of encoding and weight_path for one component.

    Read-only: does not mutate *config*. Intended for callers that consume
    a ``MAXModelConfig`` directly without going through architecture-level
    resolution -- e.g. a diffusion per-component builder -- so they get the
    same best-effort inference diffuser sub-components rely on, without
    depending on mutation having happened first.

    Safe to call even when *config* is already fully resolved (e.g. an LLM
    component whose ``quantization_encoding``/``weight_path`` were set during
    architecture validation): both steps are no-ops once those are set.

    Returns:
        A ``(encoding, weight_path)`` tuple. Either may be left unresolved
        (``None`` / ``[]``) if ambiguous.
    """
    encoding = config.quantization_encoding
    cast_from = config._applied_dtype_cast_from
    weight_path = config.weight_path

    if not encoding:
        try:
            encoding, cast_from, _ = _infer_quantization_encoding(config)
        except Exception:
            logger.debug(
                "Could not infer quantization_encoding for %s.",
                config.model_path,
            )
            encoding = config.quantization_encoding

    if encoding and not weight_path:
        try:
            weight_path = _infer_weight_path(config, encoding, cast_from)
        except Exception:
            logger.debug(
                "Could not resolve weight_path for %s.", config.model_path
            )
            weight_path = config.weight_path

    return encoding, weight_path


def _finalize_gptq_quant_config(
    config: MAXModelConfig,
) -> QuantizationConfig | None:
    """Builds the GPTQ ``QuantizationConfig`` when applicable.

    Requires ``quantization_encoding`` to be set.

    Returns:
        The ``QuantizationConfig`` for a ``"gptq"`` encoding, or ``None``
        for any other encoding.
    """
    assert config.quantization_encoding

    if config.quantization_encoding != "gptq":
        return None

    hf_quant_config = config.huggingface_config.quantization_config

    # This is a bit hacky, but seems like we need it for now.
    # This warning is for the MAX pipeline to alert users about a GPTQ format we don't support yet.
    # Instead of running our GPTQ pipeline on this unsupported format and outputting gibberish, we exit early with a clear error message.
    if str(config.huggingface_config.torch_dtype) not in [
        "float16",
        "torch.float16",
    ]:
        raise ValueError(
            f"{config.huggingface_config.torch_dtype} scales are not supported for GPTQ-quantized models."
        )
    return QuantizationConfig(
        quant_method=hf_quant_config["quant_method"],
        bits=hf_quant_config["bits"],
        group_size=hf_quant_config["group_size"],
        desc_act=hf_quant_config["desc_act"],
        sym=hf_quant_config["sym"],
    )


class MAXModelConfigBase(ConfigFileModel):
    """Abstract base class for MAX model configuration.

    Configures the model used by a pipeline. Subclass this when creating
    specialized model configurations that do not require all fields defined
    in :class:`MAXModelConfig`.
    """

    # Allow arbitrary types (like DeviceRef, AutoConfig) to avoid schema generation errors.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MAXModelConfig(MAXModelConfigBase):
    """Configuration for a pipeline model."""

    use_subgraphs: bool = Field(
        default=True,
        description=(
            "Whether to use subgraphs for the model. This can significantly "
            "reduce compile time, especially for large models with identical "
            "blocks. Default is true."
        ),
    )
    """Whether to use subgraphs for the model."""

    data_parallel_degree: int = Field(
        default=1,
        description=(
            "Data-parallelism parameter. The degree to which the model is "
            "replicated is dependent on the model type."
        ),
    )
    """The degree of data parallelism for replicating the model."""

    pool_embeddings: bool = Field(
        default=True, description="Whether to pool embedding outputs."
    )
    """Whether to pool embedding outputs."""

    max_length: int | None = Field(
        default=None,
        description=(
            "Maximum sequence length the model can process. If not specified, "
            "defaults to the model's ``max_position_embeddings``. May be clamped "
            "during resolution based on available memory."
        ),
    )
    """The maximum sequence length the model can process."""

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int | None) -> int | None:
        """Validate that max_length is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("max_length must be non-negative")
        return v

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = Field(
        default="",
        description=(
            "Accepts either a Hugging Face repository ID "
            "or a local path to the model."
        ),
    )
    """The repository ID of a Hugging Face model to use."""

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "``model_path``."
        ),
    )
    """An optional override for the client-facing model name."""

    weight_path: list[Path] = Field(
        default_factory=list,
        description=(
            "Optional path or URL of the model weights to use. "
            "Overrides default weight discovery."
        ),
    )
    """The path or URL of the model weights to use."""

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description=(
            "Weight encoding type. For GGUF models, the encoding is "
            "auto-detected from the repository when unset; if set, it must "
            "match an available encoding. When the repository contains "
            "multiple quantization formats, set this to choose one."
        ),
    )
    """The weight encoding type."""

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_model_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )
    """The branch or Git revision of the Hugging Face model repository."""

    huggingface_weight_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )
    """The branch or Git revision of the Hugging Face weights repository."""

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modeling files on Hugging Face."
        ),
    )
    """Whether to allow custom modeling files from Hugging Face."""

    subfolder: str | None = Field(
        default=None,
        description=(
            "Subdirectory within the HuggingFace repo to load config and "
            "weights from (for example, ``vae`` or ``text_encoder``). When set, "
            "``config.json`` and weights are resolved from "
            "``{model_path}/{subfolder}/``."
        ),
    )
    """Subdirectory within the HuggingFace repo to load config and weights from."""

    device_specs: list[DeviceSpec] = Field(
        default_factory=_default_device_specs,
        description=(
            "Devices to run inference upon. This option should not be used "
            "directly via the CLI entrypoint."
        ),
    )
    """The devices to run inference on."""

    @field_validator("device_specs", mode="before")
    @classmethod
    def _coerce_device_specs(cls, value: Any) -> list[DeviceSpec]:
        return coerce_device_specs_input(value)

    force_download: bool = Field(
        default=False,
        description=(
            "Whether to force download a given file if it's already present in "
            "the local cache."
        ),
    )
    """Whether to force download a file even if it's already in the local cache."""

    vision_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Model-specific vision configuration overrides. For example, for "
            'InternVL: ``{"max_dynamic_patch": 24}``.'
        ),
    )
    """Model-specific vision configuration overrides."""

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type. Only matters for GGUF weights."
        ),
    )
    """The RoPE type to use, forced regardless of model defaults."""

    sliding_window: int | None = Field(
        default=None,
        description=(
            "If set, overrides the model's attention to use a "
            "sliding-window causal mask of this many tokens. ``None`` "
            "(the default) defers to the HuggingFace config's "
            "``sliding_window`` field, or full causal attention if the "
            "model doesn't advertise one."
        ),
    )
    """Override the attention sliding-window size in tokens."""

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )
    """Whether the model should be built with echo capabilities."""

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config. If a path is provided, the file is "
            "read lazily by the registry when building the tokenizer. If "
            "``None``, the model's default chat template is used."
        ),
    )
    """An optional custom chat template to override the one shipped with the model."""

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The ``KVCacheConfig`` instance.",
    )
    """The KV cache configuration."""

    _applied_dtype_cast_from: SupportedEncoding | None = PrivateAttr(
        default=None
    )
    """Property to track the dtype that safetensor weights were casted from. None means no casting was applied. This should only be set by internal code."""

    _applied_dtype_cast_to: SupportedEncoding | None = PrivateAttr(default=None)
    """Property to track the dtype that safetensor weights were casted to. None means no casting was applied. This should only be set by internal code."""

    _huggingface_config: PretrainedConfig | None = PrivateAttr(default=None)
    """Hugging Face config. This should only be set by internal code."""

    _weights_repo_id: str | None = PrivateAttr(default=None)
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    # TODO(zheng): Refactor QuantizationConfig to be a MAXConfig subclass that
    # also autopopulates default values.
    _quant: QuantizationConfig | None = PrivateAttr(default=None)
    """Optional config for specifying quantization parameters. This should only be set by internal code."""

    _cached_weight_repo: HuggingFaceRepo | None = PrivateAttr(default=None)
    """Cached HuggingFaceRepo for weight files. Avoids recreating instances
    (and redundant HF API calls) on every property access."""

    _cached_model_repo: HuggingFaceRepo | None = PrivateAttr(default=None)
    """Cached HuggingFaceRepo for the model. Avoids recreating instances
    (and redundant HF API calls) on every property access."""

    _config_file_section_name: str = PrivateAttr(default="model_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    # TODO(SERVSYS-1083): This should just be a temporary fix until we can figure out a
    # better way to inject custom PrivateAttrs without relying on a custom
    # constructor.
    # NOTE: We intentionally hide this constructor override from static type
    # checkers so we preserve pydantic's generated `__init__` signature (or the
    # project's mypy plugin behavior) for normal call sites.
    if not TYPE_CHECKING:

        def __init__(self, **data: Any) -> None:
            """Initialize, seeding private attrs and resolving the weight path.

            Private attributes (``PrivateAttr``) aren't accepted as constructor
            kwargs by default, so we pop the seeded ones
            (``_huggingface_config``, ``_weights_repo_id``) here, then resolve
            the weight-path identity eagerly.
            """
            seeded_huggingface_config = data.pop("_huggingface_config", None)
            seeded_weights_repo_id = data.pop("_weights_repo_id", None)
            super().__init__(**data)
            if seeded_huggingface_config is not None:
                self._huggingface_config = seeded_huggingface_config
            if seeded_weights_repo_id is not None:
                self._weights_repo_id = seeded_weights_repo_id

            # Resolve weight-path identity eagerly so the config is fully
            # specified once constructed.
            self.weight_path, self.model_path, self._weights_repo_id = (
                _parse_weight_and_model_paths(
                    model_path=self.model_path,
                    weight_path=self.weight_path,
                    subfolder=self.subfolder,
                    weights_repo_id=self._weights_repo_id,
                )
            )

    # TODO(SERVSYS-1085): Figure out a better way to avoid having to roll our
    # own custom __getstate__/__setstate__ methods.
    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to avoid serializing non-picklable HF config.

        Drops ``_huggingface_config`` from the serialized state to ensure
        the object remains pickleable across processes; it will be
        lazily re-initialized on access via its property.
        """
        # NOTE: In pydantic v2, PrivateAttr values live in `__pydantic_private__`,
        # not necessarily in `__dict__`. Preserve private state across processes,
        # but explicitly drop `_huggingface_config` to avoid serializing possibly
        # non-picklable / remote-code-derived transformer objects.
        state = self.__dict__.copy()
        private = getattr(self, "__pydantic_private__", None)
        if private is not None:
            private_state = dict(private)
            private_state["_huggingface_config"] = None
            # HuggingFaceRepo instances carry cached HF API responses
            # (weight_files, info, etc.) that may not be picklable.
            private_state["_cached_weight_repo"] = None
            private_state["_cached_model_repo"] = None
            state["__pydantic_private__"] = private_state
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state while ensuring ``_huggingface_config`` is reset.

        ``_huggingface_config`` is restored as ``None`` to preserve the lazy
        loading behavior defined in its property.
        """
        private_state = dict(state.pop("__pydantic_private__", None) or {})

        self.__dict__.update(state)

        # Restore pydantic private attrs (and fill any missing defaults).
        private_state.setdefault("_huggingface_config", None)
        private_state.setdefault("_weights_repo_id", None)
        private_state.setdefault("_applied_dtype_cast_from", None)
        private_state.setdefault("_applied_dtype_cast_to", None)
        private_state.setdefault("_quant", None)
        private_state.setdefault("_cached_weight_repo", None)
        private_state.setdefault("_cached_model_repo", None)
        private_state.setdefault("_config_file_section_name", "model_config")
        object.__setattr__(self, "__pydantic_private__", private_state)

    @classmethod
    def from_pipeline_args(cls, args: PipelineArgs) -> Self:
        """Builds a :class:`MAXModelConfig` from a :class:`PipelineArgs`'s flat fields.

        Returns a new object on every call -- ``args`` holds no live handle
        back to it, so mutating the returned object (e.g.
        ``MAXModelConfig.from_pipeline_args(args).foo = x``) has no effect on
        a subsequent call with the same ``args``. Set the corresponding field
        on ``args`` itself instead.
        """
        # Seed ``_weights_repo_id`` (a PrivateAttr) so __init__'s weight-path
        # resolution sees it. Passed via a kwargs dict because the
        # private-attr-seeding __init__ is hidden from type checkers.
        init_kwargs: dict[str, Any] = dict(
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            weight_path=list(args.weight_path),
            quantization_encoding=args.quantization_encoding,
            huggingface_model_revision=args.huggingface_model_revision,
            huggingface_weight_revision=args.huggingface_weight_revision,
            trust_remote_code=args.trust_remote_code,
            subfolder=args.subfolder,
            device_specs=list(args.device_specs),
            force_download=args.force_download,
            vision_config_overrides=dict(args.vision_config_overrides),
            rope_type=args.rope_type,
            sliding_window=args.sliding_window,
            enable_echo=args.enable_echo,
            chat_template=args.chat_template,
            use_subgraphs=args.use_subgraphs,
            data_parallel_degree=args.data_parallel_degree,
            pool_embeddings=args.pool_embeddings,
            max_length=args.max_length,
            kv_cache=args.kv_cache.model_copy(deep=True),
            _weights_repo_id=args._weights_repo_id,
        )
        return cls(**init_kwargs)

    def validate_repo_access(self) -> None:
        """Validates that the model's Hugging Face repo is accessible.

        Deferred out of ``__init__`` so a ``MAXModelConfig`` can be constructed
        offline; invoked from ``PipelineConfig`` construction. A no-op when
        weights are given explicitly (``weight_path``), when no model is
        specified (a placeholder config), or when ``model_path`` is a local
        path -- there is no remote repo to check in those cases. Requiring a
        model to actually run is enforced later, during architecture
        resolution.

        Raises:
            ValueError: If the specified Hugging Face repo is inaccessible.
        """
        if self.weight_path or not self.model_path:
            return
        if not os.path.exists(os.path.expanduser(self.model_path)):
            validate_hf_repo_access(
                repo_id=self.model_path,
                revision=self.huggingface_model_revision,
            )

    @property
    def model_name(self) -> str:
        """Returns the served model name or model path."""
        if self.served_model_name is not None:
            return self.served_model_name
        return self.model_path

    @property
    def graph_quantization_encoding(self) -> QuantizationEncoding | None:
        """Converts the CLI encoding to a MAX Graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.

        Raises:
            ValueError: If no CLI encoding was specified.
        """
        if self.quantization_encoding is None:
            raise ValueError(
                "can't convert `None` CLI encoding to graph quantization encoding"
            )

        return supported_encoding_quantization(self.quantization_encoding)

    def weights_size(self) -> int:
        """Calculates the total size in bytes of all weight files in ``weight_path``.

        Attempts to find the weights locally first to avoid network
        calls, checking in the following order:

        1. If ``repo_type`` is ``"local"``, it checks if the path
           in ``weight_path`` exists directly as a local file path.
        2. Otherwise, if ``repo_type`` is ``"online"``, it first checks the local
           Hugging Face cache using :obj:`huggingface_hub.try_to_load_from_cache()`.
           If not found in the cache, it falls back to querying the Hugging Face
           Hub API via :obj:`HuggingFaceRepo.size_of()`.

        Returns:
            The total size of all weight files in bytes.

        Raises:
            FileNotFoundError: If ``repo_type`` is ``"local"`` and a file
                specified in ``weight_path`` is not found within the local repo
                directory.
            ValueError: If :obj:`HuggingFaceRepo.size_of()` fails to retrieve the
                file size from the Hugging Face Hub API (for example, file metadata
                not available or API error).
            RuntimeError: If the determined ``repo_type`` is unexpected.
        """
        total_weights_size = 0
        repo = self.huggingface_weight_repo

        for file_path in self.weight_path:
            file_path_str = str(file_path)
            full_file_path = Path(repo.repo_id) / file_path

            # 1. Check if the file exists locally (direct path, local repo, or cache)
            if local_file_location := self._local_weight_path(full_file_path):
                total_weights_size += os.path.getsize(local_file_location)
                continue

            # 2. File not found locally or non-existence is cached.
            if repo.repo_type == "local":
                if not self._local_weight_path(full_file_path):
                    raise FileNotFoundError(
                        f"Weight file '{file_path_str}' not found within the local repository path '{repo.repo_id}'"
                    )
            # If it was an online repo, we need to check the API.
            elif repo.repo_type == "online":
                # 3. Fallback: File not local/cached, get size via API for online repos.
                next_size = repo.size_of(file_path_str)
                if next_size is None:
                    # size_of failed (e.g., API error, or file exists in index but metadata failed)
                    raise ValueError(
                        f"Failed to get size of weight file {file_path_str} from repository {repo.repo_id}"
                    )
                total_weights_size += next_size
            else:
                # This case should ideally not be reached due to repo_type validation.
                raise RuntimeError(
                    f"Unexpected repository type: {repo.repo_type}"
                )

        return total_weights_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_weight_repo_id(self) -> str:
        """Returns the Hugging Face repo ID used for weight files."""
        # `_weights_repo_id` is a PrivateAttr. Some construction paths (notably
        # unpickling) can bypass __init__, so the PrivateAttr may be absent.
        weights_repo_id: str | None = getattr(self, "_weights_repo_id", None)
        return weights_repo_id if weights_repo_id else self.model_path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_weight_repo(self) -> HuggingFaceRepo:
        """Returns the Hugging Face repo handle for weight files.

        The result is cached in a PrivateAttr to avoid recreating
        ``HuggingFaceRepo`` instances (and triggering redundant HF API
        calls for file listing, encoding detection, etc.) on every
        access.  The cache is invalidated when the underlying config
        fields change (e.g. after ``model_copy()``).
        """
        weights_repo_id = self.huggingface_weight_repo_id
        # When weights come from an external repo, don't apply the
        # component subfolder — the external repo has its own layout.
        weights_from_external_repo = (
            self._weights_repo_id is not None
            and self._weights_repo_id != self.model_path
        )
        subfolder = None if weights_from_external_repo else self.subfolder

        cached = self._cached_weight_repo
        if (
            cached is not None
            and cached.repo_id == weights_repo_id
            and cached.revision == self.huggingface_weight_revision
            and cached.subfolder == subfolder
        ):
            return cached

        repo = HuggingFaceRepo(
            repo_id=weights_repo_id,
            revision=self.huggingface_weight_revision,
            trust_remote_code=self.trust_remote_code,
            subfolder=subfolder,
        )
        self._cached_weight_repo = repo
        return repo

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_model_repo(self) -> HuggingFaceRepo:
        """Returns the Hugging Face repo handle for the model.

        The result is cached in a PrivateAttr to avoid recreating
        ``HuggingFaceRepo`` instances on every access.  The cache is
        invalidated when the underlying config fields change.
        """
        cached = self._cached_model_repo
        if (
            cached is not None
            and cached.repo_id == self.model_path
            and cached.revision == self.huggingface_model_revision
            and cached.subfolder == self.subfolder
        ):
            return cached

        repo = HuggingFaceRepo(
            repo_id=self.model_path,
            revision=self.huggingface_model_revision,
            trust_remote_code=self.trust_remote_code,
            subfolder=self.subfolder,
        )
        self._cached_model_repo = repo
        return repo

    @property
    def architecture_name(self) -> str | None:
        """Returns the architecture class name from the HuggingFace config.

        For transformers models, returns ``architectures[0]`` from the
        HuggingFace config.
        """
        hf_config = self.huggingface_config
        if hf_config is not None:
            architectures = getattr(hf_config, "architectures", None)
            if architectures:
                return architectures[0]
        return None

    @property
    def huggingface_config(self) -> PretrainedConfig:
        """Returns the Hugging Face model config (loaded on first access).

        For transformers models, returns the ``AutoConfig`` subclass.  For
        non-transformers models (e.g. diffusers components), falls back to
        loading the raw ``config.json`` and wrapping it in a
        ``PretrainedConfig``.

        Raises:
            FileNotFoundError: If no ``config.json`` can be found for the
                model repo/subfolder.
        """
        # Note: For multiprocessing, __getstate__ clears _huggingface_config
        # before pickling. Each worker process reloads fresh, which correctly
        # handles trust_remote_code dynamic class loading.
        if self._huggingface_config is None:
            self._huggingface_config = load_huggingface_config(
                self.huggingface_model_repo
            )
        return self._huggingface_config

    @cached_property
    def generation_config(self) -> GenerationConfig:
        """Retrieves the Hugging Face ``GenerationConfig`` for this model.

        Lazily loads the ``GenerationConfig`` from the model repository
        and caches it to avoid repeated remote fetches.

        Returns:
            The ``GenerationConfig`` for the model, containing generation parameters
            including ``max_length``, ``temperature``, and ``top_p``. If loading
            fails, returns a default ``GenerationConfig``.
        """
        try:
            kwargs: dict[str, Any] = {
                "trust_remote_code": self.huggingface_model_repo.trust_remote_code,
                "revision": self.huggingface_model_repo.revision,
            }
            if self.subfolder is not None:
                kwargs["subfolder"] = self.subfolder
            return GenerationConfig.from_pretrained(
                self.huggingface_model_repo.repo_id,
                **kwargs,
            )
        except Exception as e:
            # This has no material unexpected impact on the user, so we log at debug.
            logger.debug(
                f"Failed to load generation_config from {self.model_name}: {e}. "
                "Using default GenerationConfig."
            )
            return GenerationConfig()

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def sampling_params_defaults(
        self,
    ) -> SamplingParamsGenerationConfigDefaults:
        """Returns sampling defaults derived from the generation config."""
        defaults = {}
        for (
            field_name,
            field_value,
        ) in self.generation_config.to_diff_dict().items():
            if (
                field_name
                in SamplingParamsGenerationConfigDefaults.__dataclass_fields__
            ):
                defaults[field_name] = field_value

        return SamplingParamsGenerationConfigDefaults(**defaults)

    def validate_multi_gpu_supported(self, multi_gpu_supported: bool) -> None:
        """Validates that the model architecture supports multi-GPU inference.

        Args:
            multi_gpu_supported: Whether the model architecture supports multi-GPU inference.
        """
        if (
            not multi_gpu_supported
            and len(self.device_specs) > 1
            and self.default_device_spec.device_type == "gpu"
        ):
            raise ValueError(
                f"Multiple GPU inference is currently not supported for {self.model_path}."
            )

    def validate_and_resolve_quantization_encoding_weight_path(
        self, default_encoding: SupportedEncoding
    ) -> None:
        """Verifies that the quantization encoding and weight path are consistent.

        Args:
            weight_path: The path to the weight file.
            default_encoding: The default encoding to use if no encoding is provided.
        """
        try:
            curr_weights_format = weights_format(self.weight_path)
        except ValueError:
            curr_weights_format = None

        if self.quantization_encoding:
            self._validate_and_resolve_with_given_quantization_encoding(
                weights_format=curr_weights_format
            )
        else:
            self._validate_and_resolve_without_given_quantization_encoding(
                weights_format=curr_weights_format,
                default_encoding=default_encoding,
            )

    def validate_and_resolve_rope_type(self, arch_rope_type: RopeType) -> None:
        """Resolves rope_type from architecture default if not set."""
        if self.rope_type is None:
            self.rope_type = arch_rope_type

    def validate_lora_compatibility(self) -> None:
        """Validates that LoRA configuration is compatible with model settings.

        Raises:
            ValueError: If LoRA is enabled but incompatible with current model configuration.
        """
        if self.kv_cache.enable_prefix_caching:
            raise ValueError(
                "LoRA is not compatible with prefix caching. "
                "Please disable prefix caching by using the --no-enable-prefix-caching flag."
            )

    def validate_and_resolve_with_resolved_quantization_encoding(
        self,
        supported_encodings: set[SupportedEncoding],
        default_weights_format: WeightsFormat,
    ) -> None:
        """Validates model path and weight path against resolved quantization encoding.

        Also finalizes the encoding config.

        Args:
            supported_encodings: A dictionary of supported encodings and their corresponding KV cache strategies.
            default_weights_format: The default weights format to use if no weights format is provided.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        # TODO: This call may be redundant since we do device compatibility
        # validation as they're being set?
        self._validate_quantization_encoding_device_compatibility(
            supported_encodings_list=list(supported_encodings)
        )
        self._finalize_encoding_config()
        self._resolve_weight_path(default_weights_format=default_weights_format)
        self._validate_final_architecture_model_path_weight_path()

    def _validate_and_resolve_dtype_casting(
        self, from_encoding: SupportedEncoding, to_encoding: SupportedEncoding
    ) -> None:
        """Validates dtype casting and resolves quantization_encoding if needed.

        Updates the quantization_encoding to the desired encoding. No-op if
        source and target encodings are the same. We currently only support
        float32 <-> bfloat16 weight type casting.

        Args:
            from_encoding: The current encoding to cast from.
            to_encoding: The desired encoding to cast to.

        Raises:
            ValueError: If the dtype casting is not allowed.
        """
        cast_from, cast_to = _resolve_dtype_cast(
            from_encoding=from_encoding,
            to_encoding=to_encoding,
            default_device_spec=self.default_device_spec,
        )
        if cast_from is None:
            return
        self._applied_dtype_cast_from = cast_from
        self._applied_dtype_cast_to = cast_to
        self.quantization_encoding = cast_to

    def _finalize_encoding_config(self) -> None:
        """Finalizes the encoding config.

        This method should only be called after the quantization encoding has
        been set.
        """
        quant = _finalize_gptq_quant_config(self)
        if quant is not None:
            self._quant = quant

    def _validate_and_resolve_with_given_quantization_encoding(
        self, weights_format: WeightsFormat | None
    ) -> None:
        """Validates quantization encoding when it is provided by the user."""
        assert self.quantization_encoding, (
            "quantization_encoding must be set (given by user)."
        )

        if self.weight_path:
            # Get the encoding of the first weight path file.
            # Try filename-based detection first — it works for both
            # local and remote paths and avoids ambiguity when a repo
            # has multiple dtypes (e.g. NVFP4 repos with F32 norms).
            file_encoding = parse_supported_encoding_from_file_name(
                str(self.weight_path[0])
            )
            if file_encoding is None:
                if os.path.exists(self.weight_path[0]):
                    # Local file with no encoding hint in the name.
                    file_encoding = None
                else:
                    file_encoding = (
                        self.huggingface_weight_repo.encoding_for_file(
                            self.weight_path[0],
                            preferred_encoding=self.quantization_encoding,
                        )
                    )

            if file_encoding and (
                file_encoding in _ALLOWED_CAST_ENCODINGS
                and self.quantization_encoding in _ALLOWED_CAST_ENCODINGS
            ):
                self._validate_and_resolve_dtype_casting(
                    from_encoding=self.quantization_encoding,
                    to_encoding=file_encoding,
                )
        else:
            # Check if the repo only has one quantization_encoding.
            supported_encodings = (
                self.huggingface_weight_repo.supported_encodings
            )
            to_encoding = self.quantization_encoding
            for supported_encoding in supported_encodings:
                from_encoding = supported_encoding

                if not (
                    from_encoding in _ALLOWED_CAST_ENCODINGS
                    and to_encoding in _ALLOWED_CAST_ENCODINGS
                ):
                    continue

                weight_files = self.huggingface_weight_repo.files_for_encoding(
                    encoding=supported_encoding
                )
                if weight_files:
                    self._validate_and_resolve_dtype_casting(
                        from_encoding=from_encoding,
                        to_encoding=to_encoding,
                    )
                    return

    def _validate_and_resolve_without_given_quantization_encoding(
        self,
        weights_format: WeightsFormat | None,
        default_encoding: SupportedEncoding,
    ) -> None:
        """Validates and resolves quantization encoding when not specified by user."""
        assert self.quantization_encoding is None, (
            "quantization_encoding must be None (not specified by user)."
        )

        # If weight path is not None, infer the quantization_encoding from the weight_path.
        if self.weight_path:
            if os.path.exists(self.weight_path[0]):
                # Try a filename hint first (mirrors the best-effort pass,
                # which infers from the filename regardless of local/remote).
                if encoding := parse_supported_encoding_from_file_name(
                    str(self.weight_path[0])
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logger.debug(msg)
                    self.quantization_encoding = encoding
                elif self.weight_path[0].suffix == ".safetensors":
                    # No hint in the name and no header inspection for local
                    # safetensors files: the user must specify explicitly.
                    raise ValueError(
                        "If a local safetensors file is provided, please provide a quantization_encoding."
                    )

            else:
                if encoding := self.huggingface_weight_repo.encoding_for_file(
                    self.weight_path[0]
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logger.debug(msg)
                    self.quantization_encoding = encoding
                else:
                    raise ValueError(
                        f"encoding cannot be inferred from weights file: {self.weight_path[0]}, please pass a quantization_encoding explicitly."
                    )
        else:
            # Check if the repo only has one quantization_encoding.
            supported_encodings = (
                self.huggingface_weight_repo.supported_encodings
            )
            if len(supported_encodings) == 1:
                msg = f"huggingface repo only has '{supported_encodings[0]}' weights, using '{supported_encodings[0]}'"
                logger.debug(msg)
                self.quantization_encoding = supported_encodings[0]
            elif (
                self.default_device_spec.device_type != "cpu"
                and len(supported_encodings) > 1
            ):
                # TODO(AITLIB-137): replace this with more full featured logic.
                # If we are running on an accelerator and the quantization encoding is not set, override to bfloat16.
                if "float4_e2m1fnx2" in supported_encodings:
                    self.quantization_encoding = "float4_e2m1fnx2"
                elif "float8_e4m3fn" in supported_encodings:
                    self.quantization_encoding = "float8_e4m3fn"
                elif "bfloat16" in supported_encodings:
                    self.quantization_encoding = "bfloat16"
            else:
                msg = f"encoding not provided, using default encoding of {default_encoding}"
                logger.debug(msg)
                self.quantization_encoding = default_encoding

        # On GPU, cast float32 -> bfloat16 (the natural GPU dtype). Mirrors
        # the equivalent step in the best-effort pass
        # (_infer_quantization_encoding), which normally applies this cast
        # before architecture resolution ever runs. This is a backstop for
        # cases where quantization_encoding is still unresolved here --
        # e.g. the earlier encoding-inference pass didn't run, or the
        # best-effort pass silently failed -- so architecture validation is
        # self-sufficient rather than depending on the earlier pass.
        if (
            self.quantization_encoding == "float32"
            and self.default_device_spec.device_type != "cpu"
        ):
            self._validate_and_resolve_dtype_casting(
                from_encoding="float32", to_encoding="bfloat16"
            )

    def _validate_quantization_encoding_device_compatibility(
        self,
        supported_encodings_list: list[SupportedEncoding],
    ) -> None:
        """Validates that the quantization encoding is supported on the specified devices.

        Should only be called after the quantization encoding has been set.
        """
        assert self.quantization_encoding, (
            "quantization_encoding must be set by now."
        )
        # If the current encoding is only supported on CPU, and all devices are
        # GPU, switch to CPU automatically. This "downcast" is possible. Going
        # the other way (CPU -> GPU) is not supported and will error out in the
        # loop check below.
        if supported_encoding_supported_devices(self.quantization_encoding) == (
            "cpu",
        ) and all(d.device_type == "gpu" for d in self.device_specs):
            logger.warning(
                f"Encoding '{self.quantization_encoding}' is only supported on CPU. Switching device_specs to CPU."
            )
            self.device_specs = [DeviceSpec.cpu()]
        # Check that the quantization encoding is supported on the specified
        # devices.
        for device_spec in self.device_specs:
            if not supported_encoding_supported_on(
                self.quantization_encoding, device_spec
            ):
                raise ValueError(
                    f"The encoding '{self.quantization_encoding}' is not compatible with the selected device type '{device_spec.device_type}'.\n\n"
                    f"You have two options to resolve this:\n"
                    f"1. Use a different device\n"
                    f"2. Use a different encoding (encodings available for this model: {', '.join(str(enc) for enc in supported_encodings_list)})\n\n"
                    f"Please use the --help flag for more information."
                )

    def _resolve_weight_path(
        self, default_weights_format: WeightsFormat
    ) -> None:
        """Resolves the weight path.

        This method should only be called after the quantization encoding has
        been set.

        Args:
            default_weights_format: The default weights format to use if no weight_path is provided.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        # If no weight_path is provided, we should grab the default.
        if not self.weight_path:
            # Retrieve the default files for each weights format.
            weight_files = self.huggingface_weight_repo.files_for_encoding(
                encoding=self.quantization_encoding
            )

            if not weight_files and self._applied_dtype_cast_from:
                # We allow ourselves to load float32 safetensors weights as bfloat16.
                weight_files = self.huggingface_weight_repo.files_for_encoding(
                    encoding=self._applied_dtype_cast_from
                )

            if not weight_files and self.quantization_encoding in (
                "float16",
                "bfloat16",
            ):
                # A float16/bfloat16 graph can load float32 weights cast at
                # load time by the architecture's weight adapter.
                weight_files = self.huggingface_weight_repo.files_for_encoding(
                    encoding="float32"
                )

            if default_weight_files := weight_files.get(
                default_weights_format, []
            ):
                self.weight_path = default_weight_files
            elif weight_files:
                # Load any available weight file.
                self.weight_path = next(iter(weight_files.values()))

        if not self.weight_path:
            raise ValueError(
                f"compatible weights cannot be found for '{self.quantization_encoding}', in the provided repo: '{self.huggingface_weight_repo.repo_id}'"
            )

    def _validate_final_architecture_model_path_weight_path(self) -> None:
        # Assume at this point, an architecture,
        # a model_path and weight_paths are available.
        assert self.weight_path, "weight_path must be provided."
        repo = self.huggingface_weight_repo
        for path in self.weight_path:
            path_str = str(path)
            # Check if file exists locally (direct, local repo, or cache).
            if self._local_weight_path(path):
                # Found locally: nothing to do.
                continue

            # File not found locally.
            if repo.repo_type == "local":
                if not self._local_weight_path(Path(repo.repo_id) / path):
                    # Helper returning None for local repo means not found.
                    raise FileNotFoundError(
                        f"weight file '{path_str}' not found within the local repository path '{repo.repo_id}'"
                    )
            elif repo.repo_type == "online":
                # Verify that it exists on Huggingface.
                if not repo.file_exists(path_str):
                    raise ValueError(
                        f"weight_path: '{path_str}' does not exist locally or in cache,"
                        f" and '{repo.repo_id}/{path_str}' does"
                        " not exist on HuggingFace."
                    )
            else:
                raise RuntimeError(
                    f"unexpected repository type: {repo.repo_type}"
                )

    def _local_weight_path(self, relative_path: Path) -> str | None:
        """Returns the absolute path if the weight file is found locally.

        Checks locations based on the repository type:
        - If `"local"`, try directly using `relative_path` (absolute or
          CWD-relative).
        - If `"online"`, checks the Hugging Face cache via
          `try_to_load_from_cache()`.

        Args:
            relative_path: The Path object representing the weight file,
                potentially relative to a repo root or cache.

        Returns:
            The absolute path (as a string) to the local file if found, otherwise None.
        """
        repo = self.huggingface_weight_repo

        # Check direct path first (absolute or relative to CWD).
        # NOTE(bduke): do this even for online repositories, because upstream
        # code originating from `huggingface_hub.hf_hub_download` returns
        # absolute paths for cached files.
        if relative_path.exists() and relative_path.is_file():
            return str(relative_path.resolve())

        # 1. Handle local repository paths.
        if repo.repo_type == "local":
            # Not found locally.
            return None

        # 2. Handle online repositories: try cache only.
        elif repo.repo_type == "online":
            # `try_to_load_from_cache` checks the HF cache.
            # Returns absolute path string if found in cache, otherwise None.
            cached_result = try_to_load_from_cache(
                repo_id=repo.repo_id,
                filename=str(relative_path),
                revision=repo.revision,
            )
            if cached_result and not isinstance(
                cached_result, str | os.PathLike
            ):
                # Handle cached non-existent result, which is a special sentinel value.
                raise FileNotFoundError(
                    f"cached non-existent weight file at {relative_path} on Hugging Face"
                )

            return str(cached_result) if cached_result else None
        # 3. Handle unexpected repo type.
        else:
            logger.warning(
                f"Unexpected repository type encountered: {repo.repo_type}"
            )
            return None

    def resolved_weight_paths(
        self, weight_path: list[Path] | None = None
    ) -> list[Path]:
        """Resolve weight paths to absolute local paths, downloading if needed.

        For online repos, downloads weight files from HuggingFace Hub.
        For local repos, constructs absolute paths from the repo root.

        Args:
            weight_path: Weight files to resolve, relative to the repo.
                Defaults to ``self.weight_path``. Pass an explicit,
                already-resolved list for a config whose ``weight_path``
                was never populated during resolution (e.g. a diffusion
                component resolved on demand at consumption time -- see
                :func:`_resolve_component_encoding_and_weights`).

        Returns:
            Absolute paths to weight files on disk.
        """
        if weight_path is None:
            weight_path = self.weight_path
        if not weight_path:
            return []

        weight_repo = self.huggingface_weight_repo
        if weight_repo.repo_type == "online":
            return download_weight_files(
                huggingface_model_id=weight_repo.repo_id,
                filenames=[str(x) for x in weight_path],
                revision=self.huggingface_weight_revision,
                force_download=self.force_download,
            )
        else:
            local_path = Path(weight_repo.repo_id)
            return [local_path / x for x in weight_path]

    def loader(self) -> WeightLoader:
        """Returns a :class:`WeightLoader` over this config's weights.

        The loader's namespace is the raw parameter names from the source
        files (un-prefixed). Pass this directly to a single-model
        pipeline's Module tree; for multi-component pipelines, use
        :meth:`~max.pipelines.lib.model_manifest.ModelManifest.loader`
        which exposes the role-prefixed union across configs.

        Resolution is lazy: the safetensors mmap stays cold for
        parameters the Module never asks for. Inherits the HuggingFace
        download side-effect from :meth:`resolved_weight_paths` for
        online repos.

        Resolves ``quantization_encoding``/``weight_path`` on demand (see
        :func:`_resolve_component_encoding_and_weights`) rather than
        assuming resolution already populated them -- a no-op when
        they're already set.

        Returns an empty loader when there are no weight paths -- common
        for components in a diffusion manifest that are config-only
        (for example, the scheduler).

        Returns:
            A :class:`WeightLoader` over this config's source namespace.
        """
        _, weight_path = _resolve_component_encoding_and_weights(self)
        paths = self.resolved_weight_paths(weight_path)
        if not paths:
            return dict_loader({})
        return _loader_over_weights(load_weights(paths))

    @property
    def default_device_spec(self) -> DeviceSpec:
        """Returns the default device spec for the model.

        This is the first device spec in the list, used for device spec checks
        throughout config validation.

        Returns:
            The default device spec for the model.
        """
        return self.device_specs[0]

    def create_kv_cache_config(self, **kv_cache_kwargs) -> None:
        """Creates and sets the KV cache configuration with the given parameters.

        Creates a new :class:`~max.pipelines.lib.config.KVCacheConfig` from the provided keyword arguments
        and automatically sets the cache_dtype based on the model's quantization
        encoding (or any explicit override in kv_cache_kwargs).

        Args:
            **kv_cache_kwargs: Keyword arguments to pass to the :class:`~max.pipelines.lib.config.KVCacheConfig` constructor.
                Common options include:
                - kv_cache_page_size: Number of tokens per page for paged cache
                - enable_prefix_caching: Whether to enable prefix caching
                - device_memory_utilization: Fraction of device memory to use
                - cache_dtype: Override for the cache data type
        """
        self.kv_cache = KVCacheConfig(**kv_cache_kwargs)
        # Note: the quantization_encoding is possibly not set yet here, so we first check for an explicit override.
        if cache_dtype := self._get_cache_override():
            # Handled by `create_kv_cache_config` but we set it again here to ensure it takes precedence over quantization encoding.
            self.kv_cache._cache_dtype = cache_dtype

    def set_cache_dtype_given_quantization_encoding(
        self,
    ) -> None:
        """Determines the KV cache dtype based on quantization encoding configuration.

        The dtype is determined in the following priority order:

        1. Explicit override from ``kv_cache.kv_cache_format`` (if set).
        2. Derived from the model's ``quantization_encoding``.
        3. Falls back to ``float32`` if no encoding is specified.
        """
        # First check for an explicit override.
        if cache_dtype := self._get_cache_override():
            self.kv_cache._cache_dtype = cache_dtype
            return

        # If there's no quantization encoding return a default value.
        if not self.quantization_encoding:
            self.kv_cache._cache_dtype = DType.float32
            return

        # Otherwise select the default KV cache dtype based on the quantization encoding.
        supported_encoding_to_cache_dtype = {
            "float32": DType.float32,
            "float16": DType.float16,
            "bfloat16": DType.bfloat16,
            "float8_e4m3fn": DType.bfloat16,
            "float4_e2m1fnx2": DType.bfloat16,
            "q4_k": DType.float32,
            "q4_0": DType.float32,
            "q6_k": DType.float32,
            "gptq": DType.bfloat16,
        }
        if self.quantization_encoding in supported_encoding_to_cache_dtype:
            self.kv_cache._cache_dtype = supported_encoding_to_cache_dtype[
                self.quantization_encoding
            ]
            return
        else:
            raise ValueError(
                f"Unsupported quantization encoding for KV cache dtype resolution: {self.quantization_encoding}"
            )

    def _get_cache_override(self) -> DType | None:
        """Check for an explicit KV cache dtype override from kv_cache_format.

        Parses the kv_cache.kv_cache_format string (if set) and converts it
        to the corresponding DType.

        Returns:
            The DType corresponding to the override string, or None if no
            override is set or the string is not recognized. Supported values
            are 'float32', 'bfloat16', and 'float8_e4m3fn' (case-insensitive).
        """
        if self.kv_cache.kv_cache_format is None:
            return None

        dtype_str = self.kv_cache.kv_cache_format.lower()
        cache_format_to_dtype = {
            "float32": DType.float32,
            "bfloat16": DType.bfloat16,
            "float8_e4m3fn": DType.float8_e4m3fn,
        }
        if dtype_str in cache_format_to_dtype:
            return cache_format_to_dtype[dtype_str]
        else:
            raise ValueError(
                f"Unrecognized kv_cache_format override: '{self.kv_cache.kv_cache_format}'. "
                "Supported values are 'float32', 'bfloat16', and 'float8_e4m3fn'."
            )

    def log_model_info(self, role: str) -> None:
        """Logs model configuration information for this config.

        Args:
            role: The semantic role of this model (e.g. ``"main"``,
                ``"draft"``, ``"vae"``).
        """
        logger.info("")
        logger.info("  Model: %s", role)
        separator = "\u2550" * 40  # ═
        logger.info("  %s", separator)

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.device_specs
        )

        quantization_encoding_str = str(self.quantization_encoding)
        if self._applied_dtype_cast_from:
            quantization_encoding_str = (
                f"{quantization_encoding_str}"
                f" (cast from {self._applied_dtype_cast_from})"
            )

        entries: list[tuple[str, Any]] = [
            ("model_path", self.model_path),
        ]

        # Only show subfolder when it is set.
        if self.subfolder:
            entries.append(("subfolder", self.subfolder))

        # Only show weights_repo_id when it differs from model_path.
        weight_repo_id = self.huggingface_weight_repo_id
        if weight_repo_id != self.model_path:
            entries.append(("weights_repo_id", weight_repo_id))

        entries.extend(
            [
                ("huggingface_revision", self.huggingface_model_revision),
                ("quantization_encoding", quantization_encoding_str),
                ("weight_path", _format_weight_path_summary(self.weight_path)),
                ("devices", devices_str),
                ("max_seq_len", self.max_length),
            ]
        )

        for line in _format_config_entries(entries, indent="    "):
            logger.info(line)

        # KVCache configuration
        self._log_kvcache_info()

    def _log_kvcache_info(self) -> None:
        """Logs KV cache configuration details for this model config."""
        kv_config = self.kv_cache
        sub_separator = "\u2500" * 2  # ──
        logger.info("  %s KV Cache %s", sub_separator, sub_separator)

        entries: list[tuple[str, Any]] = [
            ("cache_dtype", kv_config.cache_dtype),
            ("page_size", f"{kv_config.kv_cache_page_size} tokens"),
            ("prefix_caching", kv_config.enable_prefix_caching),
            ("kv_connector", kv_config.kv_connector or "null"),
        ]
        cfg = kv_config.kv_connector_config
        if (
            kv_config.kv_connector
            in (KVConnectorType.local, KVConnectorType.tiered)
            and cfg
        ):
            entries.append(
                ("host_swap_space", f"{cfg.host_kvcache_swap_space_gb} GB")
            )
        entries.append(
            (
                "memory_utilization",
                f"{kv_config.device_memory_utilization:.1%}",
            )
        )

        if kv_config._available_cache_memory is not None:
            entries.append(
                (
                    "available_cache_memory",
                    to_human_readable_bytes(kv_config._available_cache_memory),
                )
            )

        for line in _format_config_entries(entries, indent="    "):
            logger.info(line)


def _format_weight_path_summary(weight_paths: list[Path]) -> str:
    """Format weight paths as a compact single-line summary.

    Args:
        weight_paths: List of weight file paths.

    Returns:
        A human-readable summary string, e.g.
        ``"model-*.safetensors (10 files)"``.
    """
    if len(weight_paths) == 0:
        return "(none)"
    if len(weight_paths) == 1:
        return str(weight_paths[0])

    # Find common prefix and extension to build a glob-like summary.
    from os.path import commonprefix

    str_paths = [str(p) for p in weight_paths]
    prefix = commonprefix(str_paths)
    extensions = {p.rsplit(".", 1)[-1] for p in str_paths if "." in p}
    ext = f".{extensions.pop()}" if len(extensions) == 1 else ""
    # Trim prefix to last separator for a cleaner glob.
    for sep in ("/", "-", "_"):
        idx = prefix.rfind(sep)
        if idx != -1:
            prefix = prefix[: idx + 1]
            break
    return f"{prefix}*{ext} ({len(weight_paths)} files)"


def _format_config_entries(
    entries: list[tuple[str, Any]], indent: str = "    "
) -> list[str]:
    """Format key-value config entries with aligned colons.

    Args:
        entries: List of (key, value) tuples to format.
        indent: Prefix string for each line.

    Returns:
        A list of formatted strings with keys left-aligned and colons
        vertically aligned based on the longest key.
    """
    max_key_len = max(len(key) for key, _ in entries)
    return [f"{indent}{key:<{max_key_len}} : {value}" for key, value in entries]
