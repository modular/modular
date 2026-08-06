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

"""User-facing input arguments for a MAX pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from max.config import ConfigFileModel
from max.driver import DeviceSpec
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.config.profiling_config import ProfilingConfig
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    coerce_device_specs_input,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.config_enums import (
    RopeType,
    SupportedEncoding,
)
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.config.config import PipelineConfig


class PipelineArgs(ConfigFileModel):
    """User-settable input arguments for a pipeline.

    ``PipelineArgs`` is the user-facing input to the pipeline system. It
    holds flat model-level fields plus nested sub-configs mirroring the
    :class:`PipelineConfig` schema (``runtime``, ``sampling``,
    ``profiling``) and a small number of cohesive sub-config objects
    (``kv_cache``, ``lora``, ``speculative``, ``draft_model``).

    Multi-component pipelines (e.g. diffusion) that require a pre-built
    :class:`~max.pipelines.lib.model_manifest.ModelManifest` may pass
    ``models=<manifest>`` to the constructor. That manifest is stored as a
    private override and used verbatim by :meth:`PipelineConfig.from_args`
    instead of constructing one from the flat scalar fields.

    Call :meth:`PipelineConfig.from_args` to obtain a fully-constructed
    :class:`PipelineConfig` ready for architecture-driven resolution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------ #
    # Top-level pipeline fields
    # ------------------------------------------------------------------ #

    model_override: list[str] = Field(
        default_factory=list,
        description=(
            "Per-component overrides for the ModelManifest, in the format "
            "``component.field=value``. Applied before resolution. Repeatable."
        ),
    )

    task: PipelineTask = Field(
        default=PipelineTask.UNDEFINED,
        description=(
            "The pipeline task to run (e.g. ``text_generation``, "
            "``embeddings_generation``). Used to disambiguate architectures "
            "registered under the same name for multiple tasks."
        ),
    )

    debug_verify_replay: bool = Field(
        default=False,
        description=(
            "When ``device_graph_capture`` is enabled, execute eager launch-trace "
            "verification before replay. Intended for debugging only."
        ),
    )

    # ------------------------------------------------------------------ #
    # Fields from MAXModelConfig
    # ------------------------------------------------------------------ #

    model_path: str = Field(
        default="",
        description=(
            "Accepts either a Hugging Face repository ID "
            "or a local path to the model."
        ),
    )

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "``model_path``."
        ),
    )

    weight_path: list[Path] = Field(
        default_factory=list,
        description=(
            "Optional path or URL of the model weights to use. "
            "Overrides default weight discovery."
        ),
    )

    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description=(
            "Weight encoding type. For GGUF models, the encoding is "
            "auto-detected from the repository when unset."
        ),
    )

    huggingface_model_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )

    huggingface_weight_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face weight repository to use."
        ),
    )

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modeling files on Hugging Face."
        ),
    )

    subfolder: str | None = Field(
        default=None,
        description=(
            "Subdirectory within the HuggingFace repo to load config and "
            "weights from."
        ),
    )

    device_specs: list[DeviceSpec] = Field(
        default_factory=_default_device_specs,
        description=("Devices to run inference upon."),
    )

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

    vision_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=("Model-specific vision configuration overrides."),
    )

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type. Only matters for GGUF weights."
        ),
    )

    sliding_window: int | None = Field(
        default=None,
        description=(
            "If set, overrides the model's attention to use a "
            "sliding-window causal mask of this many tokens."
        ),
    )

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config."
        ),
    )

    use_subgraphs: bool = Field(
        default=True,
        description=("Whether to use subgraphs for the model."),
    )

    data_parallel_degree: int = Field(
        default=1,
        description=("Data-parallelism parameter."),
    )

    pool_embeddings: bool = Field(
        default=True,
        description="Whether to pool embedding outputs.",
    )

    max_length: int | None = Field(
        default=None,
        description=("Maximum sequence length the model can process."),
    )

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The ``KVCacheConfig`` instance.",
    )

    # ------------------------------------------------------------------ #
    # Sub-configs mirroring the PipelineConfig schema
    # ------------------------------------------------------------------ #

    runtime: PipelineRuntimeConfig = Field(
        default_factory=PipelineRuntimeConfig,
        description="Runtime and scheduling configuration.",
    )

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Token sampling configuration.",
    )

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig,
        description="Profiling configuration.",
    )

    # ------------------------------------------------------------------ #
    # Sub-config objects (kept cohesive)
    # ------------------------------------------------------------------ #

    lora: LoRAConfig | None = Field(
        default=None,
        description="The LoRA config.",
    )

    speculative: SpeculativeConfig | None = Field(
        default=None,
        description="The SpeculativeConfig.",
    )

    draft_model: MAXModelConfig | None = Field(
        default=None,
        description=(
            "Draft model configuration for speculative decoding. "
            "Replaces the ``models['draft']`` entry in a :class:`PipelineConfig`."
        ),
    )

    # Escape hatch for multi-component pipelines (e.g. diffusion) where
    # a pre-built ModelManifest is required. When set,
    # PipelineConfig.from_args() uses this manifest directly instead of
    # constructing one from flat fields.
    _manifest_override: ModelManifest | None = PrivateAttr(default=None)

    # Cross-repo weight source (e.g. a bartowski GGUF repo supplying weights
    # for a meta-llama config repo). Not a user-settable input field -- set
    # directly on the instance (``args._weights_repo_id = ...``) by callers
    # that need it, then re-seeded onto the built MAXModelConfig by
    # MAXModelConfig.from_pipeline_args(), since that returns a fresh object
    # each call.
    _weights_repo_id: str | None = PrivateAttr(default=None)

    def __init__(
        self, *, models: ModelManifest | None = None, **data: Any
    ) -> None:
        super().__init__(**data)
        if models is not None:
            object.__setattr__(self, "_manifest_override", models)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def main_architecture_name(self) -> str:
        """Returns the HuggingFace architecture class name for the main model.

        Reads ``architectures[0]`` from the model's HuggingFace config without
        constructing a full :class:`PipelineConfig`.

        Raises:
            ValueError: If the architecture name cannot be determined.
        """
        if self._manifest_override is not None:
            return self._manifest_override.main_architecture_name
        arch = MAXModelConfig.from_pipeline_args(self).architecture_name
        if arch is None:
            raise ValueError(
                f"Cannot determine architecture name for {self.model_path!r}: "
                "HuggingFace config has no 'architectures' field."
            )
        return arch

    @classmethod
    def from_flat_kwargs(cls, **kwargs: Any) -> Self:
        """Construct a :class:`PipelineArgs` from a flat CLI kwargs namespace.

        Routes flat kwargs (the same format accepted by
        :meth:`PipelineConfig.from_flat_kwargs`) into the appropriate flat
        fields of :class:`PipelineArgs`. Delegates to
        :meth:`PipelineConfig.from_flat_kwargs` for the heavy-lifting of
        routing and sub-config construction, then extracts the user-facing
        fields.

        Args:
            **kwargs: Flat keyword arguments, e.g. ``model_path``,
                ``kv_cache_size``, ``enable_lora``.

        Returns:
            A :class:`PipelineArgs` populated from the flat kwargs.
        """
        from max.pipelines.lib.config.config import PipelineConfig

        pipeline_config = PipelineConfig.from_flat_kwargs(**kwargs)
        return cls.from_pipeline_config(pipeline_config)

    @classmethod
    def from_pipeline_config(cls, pipeline_config: PipelineConfig) -> Self:
        """Construct a :class:`PipelineArgs` from an existing :class:`PipelineConfig`.

        Extracts the user-facing fields from a :class:`PipelineConfig`
        and returns a :class:`PipelineArgs` populated from them.

        This exists to let :meth:`from_flat_kwargs` reuse
        :meth:`PipelineConfig.from_flat_kwargs`'s flat-kwarg routing logic
        (parsing ``--model-override``, building the draft model config,
        etc.) instead of duplicating it. It is not a general round-trip:
        ``pipeline_config`` is expected to be freshly constructed and not
        yet resolved. Resolution-derived state (e.g. an applied dtype cast
        recorded during architecture-level resolution) is *not* preserved --
        :class:`PipelineArgs` is deliberately isolated from resolution
        mutations (see #90128), so passing an already-resolved
        ``pipeline_config`` here will silently drop that state.

        Args:
            pipeline_config: The source :class:`PipelineConfig` to extract
                from. Should not have had :meth:`PipelineConfig.resolve`
                called on it.

        Returns:
            A :class:`PipelineArgs` populated from the given config.
        """
        main = pipeline_config.models.get("main") or MAXModelConfig()
        # Multi-component (diffusion) manifests have no "main" entry; their
        # per-component configs can't be reconstructed from the flat fields,
        # so carry the manifest through verbatim.
        manifest = (
            pipeline_config.models
            if "main" not in pipeline_config.models
            else None
        )
        return cls(
            models=manifest,
            # top-level
            model_override=list(pipeline_config.model_override),
            task=pipeline_config.task,
            debug_verify_replay=pipeline_config.debug_verify_replay,
            # MAXModelConfig fields
            model_path=main.model_path,
            served_model_name=main.served_model_name,
            weight_path=list(main.weight_path),
            quantization_encoding=main.quantization_encoding,
            huggingface_model_revision=main.huggingface_model_revision,
            huggingface_weight_revision=main.huggingface_weight_revision,
            trust_remote_code=main.trust_remote_code,
            subfolder=main.subfolder,
            device_specs=list(main.device_specs),
            force_download=main.force_download,
            vision_config_overrides=dict(main.vision_config_overrides),
            rope_type=main.rope_type,
            sliding_window=main.sliding_window,
            enable_echo=main.enable_echo,
            chat_template=main.chat_template,
            use_subgraphs=main.use_subgraphs,
            data_parallel_degree=main.data_parallel_degree,
            pool_embeddings=main.pool_embeddings,
            max_length=main.max_length,
            kv_cache=main.kv_cache.model_copy(deep=True),
            # sub-configs
            runtime=pipeline_config.runtime.model_copy(deep=True),
            sampling=pipeline_config.sampling.model_copy(deep=True),
            profiling=pipeline_config.profiling.model_copy(deep=True),
            lora=pipeline_config.lora.model_copy(deep=True)
            if pipeline_config.lora
            else None,
            speculative=pipeline_config.speculative.model_copy(deep=True)
            if pipeline_config.speculative
            else None,
            draft_model=pipeline_config.draft_model.model_copy(deep=True)
            if pipeline_config.draft_model is not None
            else None,
        )
