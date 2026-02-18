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
from max.driver import DeviceSpec, devices_exist, scan_available_devices
from max.dtype import DType
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightsFormat, weights_format
from max.interfaces import SamplingParamsGenerationConfigDefaults
from max.nn.legacy.kv_cache import KVCacheStrategy
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)
from transformers import AutoConfig
from transformers.generation import GenerationConfig

from .config_enums import RepoType, RopeType, SupportedEncoding
from .device_specs import coerce_device_specs_input
from .hf_utils import (
    HuggingFaceRepo,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from .kv_cache_config import KVCacheConfig
from .registry import PIPELINE_REGISTRY
from .weight_path_parser import WeightPathParser

logger = logging.getLogger("max.pipelines")


# Encodings that can be casted to/from each other.
# We currently only support float32 <-> bfloat16 weight type casting.
_ALLOWED_CAST_ENCODINGS = {
    SupportedEncoding.float32,
    SupportedEncoding.bfloat16,
}


class MAXModelConfigBase(ConfigFileModel):
    """Abstract base class for all (required) MAX model configs.

    This base class is used to configure a model to use for a pipeline, but also
    handy to sidestep the need to pass in optional fields when subclassing
    MAXModelConfig.
    """

    # Allow arbitrary types (like DeviceRef, AutoConfig) to avoid schema generation errors.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MAXModelConfig(MAXModelConfigBase):
    use_subgraphs: bool = Field(
        default=True,
        description=(
            "Whether to use subgraphs for the model. This can significantly "
            "reduce compile time, especially for large models with identical "
            "blocks. Default is true."
        ),
    )

    data_parallel_degree: int = Field(
        default=1,
        description=(
            "Data-parallelism parameter. The degree to which the model is "
            "replicated is dependent on the model type."
        ),
    )

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = Field(
        default="",
        description=(
            "The repository ID of a Hugging Face model to use. "
            "The `--model` option also works as an alias."
        ),
    )

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "model_path."
        ),
    )

    weight_path: list[Path] = Field(
        default_factory=list,
        description="Optional path or url of the model weights to use.",
    )

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description="Weight encoding type.",
    )

    allow_safetensors_weights_fp32_bf6_bidirectional_cast: bool = Field(
        default=False,
        description=(
            "Whether to allow automatic float32 to/from bfloat16 safetensors "
            "weight type casting, if needed. Currently only supported in "
            "Llama3 models."
        ),
    )

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_model_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )

    huggingface_weight_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modelling files on Hugging Face."
        ),
    )

    device_specs: list[DeviceSpec] = Field(
        default_factory=scan_available_devices,
        description=(
            "Devices to run inference upon. This option should not be used "
            "directly via the CLI entrypoint."
        ),
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
        description=(
            "Model-specific vision configuration overrides. For example, for "
            'InternVL: {"max_dynamic_patch": 24}.'
        ),
    )

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type: none, normal, or neox. Only "
            "matters for GGUF weights."
        ),
    )

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The KVCache config.",
    )

    _applied_dtype_cast_from: SupportedEncoding | None = PrivateAttr(
        default=None
    )
    """Property to track the dtype that safetensor weights were casted from. None means no casting was applied. This should only be set by internal code."""

    _applied_dtype_cast_to: SupportedEncoding | None = PrivateAttr(default=None)
    """Property to track the dtype that safetensor weights were casted to. None means no casting was applied. This should only be set by internal code."""

    _huggingface_config: AutoConfig | None = PrivateAttr(default=None)
    """Hugging Face config. This should only be set by internal code."""

    _diffusers_config: dict[str, Any] | None = PrivateAttr(default=None)
    """Diffusers config for diffusion pipelines. This should only be set by internal code."""

    _weights_repo_id: str | None = PrivateAttr(default=None)
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    # TODO(zheng): Refactor QuantizationConfig to be a MAXConfig subclass that
    # also autopopulates default values.
    _quant: QuantizationConfig | None = PrivateAttr(default=None)
    """Optional config for specifying quantization parameters. This should only be set by internal code."""

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
            """Initialize config, allowing tests/internal callers to seed PrivateAttrs.

            Pydantic PrivateAttrs are not regular model fields, so they are not
            accepted as constructor kwargs by default. Some tests (and debugging
            utilities) intentionally seed `_huggingface_config` to avoid network
            access and to validate config override plumbing. Hence, we need to
            explicitly define this __init__ method to seed the PrivateAttr(s).
            """
            seeded_huggingface_config = data.pop("_huggingface_config", None)
            seeded_diffusers_config = data.pop("_diffusers_config", None)
            super().__init__(**data)
            if seeded_huggingface_config is not None:
                self._huggingface_config = seeded_huggingface_config
            if seeded_diffusers_config is not None:
                self._diffusers_config = seeded_diffusers_config

    # TODO(SERVSYS-1085): Figure out a better way to avoid having to roll our
    # own custom __getstate__/__setstate__ methods.
    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to avoid serializing non-picklable HF config.

        Drops `_huggingface_config` and `_diffusers_config` from the serialized state to ensure
        the object remains pickleable across processes; they will be
        lazily re-initialized on access via their respective properties.
        """
        # NOTE: In pydantic v2, PrivateAttr values live in `__pydantic_private__`,
        # not necessarily in `__dict__`. Preserve private state across processes,
        # but explicitly drop `_huggingface_config` and `_diffusers_config` to avoid serializing possibly
        # non-picklable / remote-code-derived transformer objects.
        state = self.__dict__.copy()
        private = getattr(self, "__pydantic_private__", None)
        if private is not None:
            private_state = dict(private)
            private_state["_huggingface_config"] = None
            private_state["_diffusers_config"] = None
            state["__pydantic_private__"] = private_state
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state while ensuring `_huggingface_config` and `_diffusers_config` are reset.

        `_huggingface_config` and `_diffusers_config` are restored as None to preserve the lazy
        loading behavior defined in their respective properties.
        """
        private_state = dict(state.pop("__pydantic_private__", None) or {})

        self.__dict__.update(state)

        # Restore pydantic private attrs (and fill any missing defaults).
        private_state.setdefault("_huggingface_config", None)
        private_state.setdefault("_diffusers_config", None)
        private_state.setdefault("_weights_repo_id", None)
        private_state.setdefault("_applied_dtype_cast_from", None)
        private_state.setdefault("_applied_dtype_cast_to", None)
        private_state.setdefault("_quant", None)
        private_state.setdefault("_config_file_section_name", "model_config")
        object.__setattr__(self, "__pydantic_private__", private_state)

    # TODO(zheng): This can't just be a __post_init__ method, because we need to
    # it also sets and updates other fields which may not be determined /
    # initialized in the default factory.
    # Realistically, this shouldn't become a problem in the long term once we
    # instantiate these MAXConfigs with probably DAG dependency flows in our
    # larger config refactor.
    def resolve(self) -> None:
        """Validates and resolves the config.

        This method is called after the model config is initialized, to ensure that all
        config fields have been initialized to a valid state. It will also set
        and update other fields which may not be determined / initialized in the
        default factory.

        In order:
        1. Validate that the device_specs provided are available
        2. Parse the weight path(s) and initialize the _weights_repo_id
        """
        # Validate that --quantization-encoding is given when --allow-safetensors-weights-fp32-bf6-bidirectional-cast is True
        if (
            self.allow_safetensors_weights_fp32_bf6_bidirectional_cast
            and self.quantization_encoding is None
        ):
            raise ValueError(
                "--quantization-encoding must be provided when --allow-safetensors-weights-fp32-bf6-bidirectional-cast is enabled"
            )

        # Validate that the device_specs provided are available
        if not devices_exist(self.device_specs):
            available_devices = scan_available_devices()
            raise ValueError(
                f"device specs provided ({self.device_specs}) do not exist.\n"
                f"available devices: {available_devices}"
            )

        self.weight_path, self._weights_repo_id = WeightPathParser.parse(
            self.model_path, self.weight_path
        )

        # If we cannot infer the weight path, we lean on the model_path
        # to provide it.
        if len(self.weight_path) == 0:
            if self.model_path == "":
                raise ValueError(
                    "model must be provided and must be a valid Hugging Face repository"
                )
            elif not os.path.exists(os.path.expanduser(self.model_path)):
                # Check if the model_path is a valid HuggingFace repository
                validate_hf_repo_access(
                    repo_id=self.model_path,
                    revision=self.huggingface_model_revision,
                )
        elif self.model_path == "" and self._weights_repo_id is not None:
            # weight_path is used and we should derive the repo_id from it.
            # At this point, we should have a resolved weight path - be it local or remote HF.
            # weight_path should not be used directly anymore.
            self.model_path = self._weights_repo_id

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

        return self.quantization_encoding.quantization_encoding

    def weights_size(self) -> int:
        """Calculates the total size in bytes of all weight files in ``weight_path``.

        Attempts to find the weights locally first to avoid network
        calls, checking in the following order:

        1. If `repo_type` is :obj:`RepoType.local`, it checks if the path
           in `weight_path` exists directly as a local file path.
        2. Otherwise, if `repo_type` is :obj:`RepoType.online`, it first checks the local
           Hugging Face cache using :obj:`huggingface_hub.try_to_load_from_cache()`.
           If not found in the cache, it falls back to querying the Hugging Face
           Hub API via :obj:`HuggingFaceRepo.size_of()`.

        Returns:
            The total size of all weight files in bytes.

        Raises:
            FileNotFoundError: If `repo_type` is :obj:`RepoType.local` and a file
                specified in `weight_path` is not found within the local repo
                directory.
            ValueError: If :obj:`HuggingFaceRepo.size_of()` fails to retrieve the
                file size from the Hugging Face Hub API (e.g., file metadata
                not available or API error).
            RuntimeError: If the determined `repo_type` is unexpected.
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
            if repo.repo_type == RepoType.local:
                if not self._local_weight_path(full_file_path):
                    raise FileNotFoundError(
                        f"Weight file '{file_path_str}' not found within the local repository path '{repo.repo_id}'"
                    )
            # If it was an online repo, we need to check the API.
            elif repo.repo_type == RepoType.online:
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
        """Returns the Hugging Face repo handle for weight files."""
        return HuggingFaceRepo(
            repo_id=self.huggingface_weight_repo_id,
            revision=self.huggingface_weight_revision,
            trust_remote_code=self.trust_remote_code,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_model_repo(self) -> HuggingFaceRepo:
        """Returns the Hugging Face repo handle for the model."""
        return HuggingFaceRepo(
            repo_id=self.model_path,
            revision=self.huggingface_model_revision,
            trust_remote_code=self.trust_remote_code,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_config(self) -> AutoConfig | None:
        """Returns the Hugging Face model config (loaded on first access)."""
        # Note: For multiprocessing, __getstate__ clears _huggingface_config
        # before pickling. Each worker process will reload the config fresh,
        # which properly handles trust_remote_code dynamic class loading.
        if self._huggingface_config is None:
            try:
                self._huggingface_config = (
                    PIPELINE_REGISTRY.get_active_huggingface_config(
                        huggingface_repo=self.huggingface_model_repo
                    )
                )
            except Exception as e:
                # Not a transformers-style model (e.g., diffusers model)
                logger.debug(
                    f"Could not load HuggingFace config for "
                    f"{self.model_path}: {e}"
                )
                return None
        return self._huggingface_config

    @computed_field  # type: ignore[prop-decorator]
    @property
    def diffusers_config(self) -> dict[str, Any] | None:
        """Retrieve the diffusers config for diffusion pipelines.

        Note: For multiprocessing, __getstate__ clears _diffusers_config
        before pickling. Each worker process will reload the config fresh.

        Returns:
            The diffusers config dict if this is a diffusion pipeline, None otherwise.
            The dict will have a structure with "_class_name" and "components" keys,
            where each component includes "class_name" and "config_dict" fields.
        """
        if self._diffusers_config is None:
            model_index = PIPELINE_REGISTRY.get_active_diffusers_config(
                huggingface_repo=self.huggingface_model_repo
            )
            if model_index is not None:
                # Enhance the model_index with component configs
                self._diffusers_config = self._load_diffusers_components(
                    model_index
                )
            else:
                self._diffusers_config = None
        return self._diffusers_config

    def _load_diffusers_components(
        self, model_index: dict[str, Any]
    ) -> dict[str, Any]:
        """Load component configs for a diffusers pipeline.

        Args:
            model_index: The raw model_index.json dict from HuggingFace.

        Returns:
            Enhanced config dict with "components" key containing loaded configs.
        """
        import json

        from huggingface_hub import hf_hub_download, list_repo_files

        # Extract class name and version
        class_name = model_index.get("_class_name")
        diffusers_version = model_index.get("_diffusers_version")

        # Build components dict with loaded configs
        components = {}
        repo = self.huggingface_model_repo
        repo_root: Path | None = None
        if repo.repo_type == RepoType.local:
            repo_root = Path(repo.repo_id)
            assert repo_root.exists(), (
                "Local Hugging Face repository path does not exist: "
                f"{repo_root}"
            )
            repo_files = [
                path.relative_to(repo_root).as_posix()
                for path in repo_root.rglob("*")
                if path.is_file()
            ]
        else:
            repo_files = list_repo_files(
                repo_id=repo.repo_id,
                revision=repo.revision,
            )

        component_configs = {}
        for file_name in repo_files:
            if "/" in file_name:
                component_name, target = file_name.split("/")
                if "_" in component_name:
                    key = component_name.split("_")[0]
                else:
                    key = component_name
                if target in ["config.json", f"{key}_config.json"]:
                    try:
                        if repo.repo_type == RepoType.local:
                            assert repo_root is not None, (
                                "repo_root must be set for local repo types."
                            )
                            cfg_path = repo_root / file_name
                        else:
                            cfg_path = Path(
                                hf_hub_download(
                                    repo_id=repo.repo_id,
                                    filename=file_name,
                                    revision=repo.revision,
                                )
                            )
                        with open(cfg_path) as f:
                            component_configs[component_name] = json.load(f)
                    except Exception as e:
                        logger.debug(
                            f"Could not load config for {file_name}: {e}"
                        )

        for component_name, component_info in model_index.items():
            if component_name.startswith("_"):
                continue

            if not isinstance(component_info, list) or len(component_info) != 2:
                continue

            library, class_type = component_info

            components[component_name] = {
                "library": library,
                "class_name": class_type,
                "config_dict": component_configs.get(component_name, {}),
            }

        # Build the final config structure
        return {
            "_class_name": class_name,
            "_diffusers_version": diffusers_version,
            "components": components,
        }

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def generation_config(self) -> GenerationConfig:
        """Retrieve the Hugging Face GenerationConfig for this model.

        This property lazily loads the GenerationConfig from the model repository
        and caches it to avoid repeated remote fetches.

        Returns:
            The GenerationConfig for the model, containing generation parameters
            like max_length, temperature, top_p, etc. If loading fails, returns
            a default GenerationConfig.
        """
        try:
            return GenerationConfig.from_pretrained(
                self.huggingface_model_repo.repo_id,
                trust_remote_code=self.huggingface_model_repo.trust_remote_code,
                revision=self.huggingface_model_repo.revision,
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
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        default_weights_format: WeightsFormat,
    ) -> None:
        """Validates model path and weight path against resolved quantization encoding.

        Also resolves the KV cache strategy and finalizes the encoding config.

        Args:
            supported_encodings: A dictionary of supported encodings and their corresponding KV cache strategies.
            default_weights_format: The default weights format to use if no weights format is provided.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        # TODO: This call may be redundant since we do device compatibility
        # validation as they're being set?
        self._validate_quantization_encoding_device_compatibility(
            supported_encodings_list=list(supported_encodings.keys())
        )
        self._finalize_encoding_config()
        self._resolve_weight_path(default_weights_format=default_weights_format)
        self._resolve_kv_cache_strategy(supported_encodings=supported_encodings)
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
        assert self.allow_safetensors_weights_fp32_bf6_bidirectional_cast, (
            "allow_safetensors_weights_fp32_bf6_bidirectional_cast must be set to True"
        )

        if from_encoding == to_encoding:
            return
        elif not (
            from_encoding in _ALLOWED_CAST_ENCODINGS
            and to_encoding in _ALLOWED_CAST_ENCODINGS
        ):
            raise ValueError(
                f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{self.default_device_spec}'. "
                f"We only support float32 <-> bfloat16 weight type casting."
            )

        if not to_encoding.supported_on(device_spec=self.default_device_spec):
            raise ValueError(
                f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{self.default_device_spec}' because '{to_encoding}' is not supported on this device."
                f"Please use a different device or a different encoding."
            )
        self._applied_dtype_cast_from = from_encoding
        self._applied_dtype_cast_to = to_encoding
        self.quantization_encoding = to_encoding

    def _validate_and_resolve_with_given_quantization_encoding(
        self, weights_format: WeightsFormat | None
    ) -> None:
        """Validates quantization encoding when it is provided by the user."""
        assert self.quantization_encoding, (
            "quantization_encoding must be set (given by user)."
        )

        if self.weight_path:
            # Get the encoding of the first weight path file.
            if os.path.exists(self.weight_path[0]):
                file_encoding = SupportedEncoding.parse_from_file_name(
                    str(self.weight_path[0])
                )
            else:
                file_encoding = self.huggingface_weight_repo.encoding_for_file(
                    self.weight_path[0]
                )

            if file_encoding:
                if self.allow_safetensors_weights_fp32_bf6_bidirectional_cast:
                    self._validate_and_resolve_dtype_casting(
                        from_encoding=self.quantization_encoding,
                        to_encoding=file_encoding,
                    )
                # For cases where they do not match but with allow_safetensors_weights_fp32_bf6_bidirectional_cast set to False, we raise an error.
                elif file_encoding != self.quantization_encoding:
                    raise ValueError(
                        f"weight_path provided '{self.weight_path[0]}' has an inconsistent encoding '{file_encoding}' than quantization_encoding provided '{self.quantization_encoding}'. Please update one."
                    )
        else:
            if self.allow_safetensors_weights_fp32_bf6_bidirectional_cast:
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

                    weight_files = (
                        self.huggingface_weight_repo.files_for_encoding(
                            encoding=supported_encoding
                        )
                    )
                    if weight_files:
                        self._validate_and_resolve_dtype_casting(
                            from_encoding=from_encoding,
                            to_encoding=to_encoding,
                        )
                        return
            else:
                weight_files = self.huggingface_weight_repo.files_for_encoding(
                    encoding=self.quantization_encoding
                )
                if not weight_files:
                    raise ValueError(
                        f"quantization_encoding '{self.quantization_encoding}' is not supported by the repo '{self.huggingface_weight_repo.repo_id}'"
                    )

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
                # Not currently supported. Infer encoding from local path.
                if self.weight_path[0].suffix == ".safetensors":
                    raise ValueError(
                        "If a local safetensors file is provided, please provide a quantization_encoding."
                    )

                if encoding := SupportedEncoding.parse_from_file_name(
                    str(self.weight_path[0])
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logger.debug(msg)
                    self.quantization_encoding = encoding

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
                if SupportedEncoding.float4_e2m1fnx2 in supported_encodings:
                    self.quantization_encoding = (
                        SupportedEncoding.float4_e2m1fnx2
                    )
                elif SupportedEncoding.float8_e4m3fn in supported_encodings:
                    self.quantization_encoding = SupportedEncoding.float8_e4m3fn
                elif SupportedEncoding.bfloat16 in supported_encodings:
                    self.quantization_encoding = SupportedEncoding.bfloat16
            else:
                msg = f"encoding not provided, using default encoding of {default_encoding}"
                logger.debug(msg)
                self.quantization_encoding = default_encoding

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
        if self.quantization_encoding.supported_devices == ("cpu",) and all(
            d.device_type == "gpu" for d in self.device_specs
        ):
            logger.warning(
                f"Encoding '{self.quantization_encoding}' is only supported on CPU. Switching device_specs to CPU."
            )
            self.device_specs = [DeviceSpec.cpu()]
        # Check that the quantization encoding is supported on the specified
        # devices.
        for device_spec in self.device_specs:
            if not self.quantization_encoding.supported_on(device_spec):
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

    def _resolve_kv_cache_strategy(
        self,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
    ) -> None:
        """Resolves the KVCacheStrategy.

        This method should only be called after the quantization encoding has
        been set / resolved.

        Args:
            supported_encodings: A dictionary of supported encodings and their corresponding KV cache strategies.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        # Check supported_cache_strategy
        supported_cache_strategies = supported_encodings.get(
            self.quantization_encoding, []
        )
        if (
            self.kv_cache.cache_strategy == "model_default"
            and supported_cache_strategies
        ):
            default_strategy = supported_cache_strategies[0]
            msg = f"default cache_strategy of '{default_strategy}' enabled"
            logger.debug(msg)

            self.kv_cache.cache_strategy = default_strategy
        elif (
            supported_cache_strategies
            and self.kv_cache.cache_strategy not in supported_cache_strategies
        ):
            supported_strategy = supported_cache_strategies[0]

            msg = f"cache_strategy = '{self.kv_cache.cache_strategy}' not supported for '{self.quantization_encoding}', using '{supported_strategy}' cache strategy."
            logger.warning(msg)

            self.kv_cache.cache_strategy = supported_strategy

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
            if repo.repo_type == RepoType.local:
                if not self._local_weight_path(Path(repo.repo_id) / path):
                    # Helper returning None for local repo means not found.
                    raise FileNotFoundError(
                        f"weight file '{path_str}' not found within the local repository path '{repo.repo_id}'"
                    )
            elif repo.repo_type == RepoType.online:
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

    def _finalize_encoding_config(self) -> None:
        """Finalizes the encoding config.

        This method should only be called after the quantization encoding has
        been set.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        if self.quantization_encoding == SupportedEncoding.gptq:
            if self.huggingface_config is None:
                raise ValueError(
                    f"GPTQ quantization requires a HuggingFace config for '{self.model_path}', "
                    "but config could not be loaded. "
                    "Please ensure the model repository contains a valid config.json with quantization_config."
                )
            hf_quant_config = self.huggingface_config.quantization_config

            # This is a bit hacky, but seems like we need it for now.
            # This warning is for the MAX pipeline to alert users about a GPTQ format we don't support yet.
            # Instead of running our GPTQ pipeline on this unsupported format and outputting gibberish, we exit early with a clear error message.
            if str(self.huggingface_config.torch_dtype) not in [
                "float16",
                "torch.float16",
            ]:
                raise ValueError(
                    f"{self.huggingface_config.torch_dtype} scales are not supported for GPTQ-quantized models."
                )
            default_quantization_config = QuantizationConfig(
                quant_method=hf_quant_config["quant_method"],
                bits=hf_quant_config["bits"],
                group_size=hf_quant_config["group_size"],
                desc_act=hf_quant_config["desc_act"],
                sym=hf_quant_config["sym"],
            )
            self._quant = default_quantization_config

    def _local_weight_path(self, relative_path: Path) -> str | None:
        """Returns the absolute path if the weight file is found locally.

        Checks locations based on the repository type:
        - If `RepoType.local`, try directly using `relative_path` (absolute or
          CWD-relative).
        - If `RepoType.online`, checks the Hugging Face cache via
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
        if repo.repo_type == RepoType.local:
            # Not found locally.
            return None

        # 2. Handle online repositories: try cache only.
        elif repo.repo_type == RepoType.online:
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
        """Create and set the KV cache configuration with the given parameters.

        This method creates a new KVCacheConfig from the provided keyword arguments
        and automatically sets the cache_dtype based on the model's quantization
        encoding (or any explicit override in kv_cache_kwargs).

        Args:
            **kv_cache_kwargs: Keyword arguments to pass to KVCacheConfig constructor.
                Common options include:
                - cache_strategy: The KV cache strategy (continuous, paged, etc.)
                - kv_cache_page_size: Number of tokens per page for paged cache
                - enable_prefix_caching: Whether to enable prefix caching
                - device_memory_utilization: Fraction of device memory to use
                - cache_dtype: Override for the cache data type
        """
        self.kv_cache = KVCacheConfig(**kv_cache_kwargs)
        # Note: the quantization_encoding is possibly not set yet here, so we first check for an explicit override.
        if cache_dtype := self._get_cache_override():
            self.kv_cache._cache_dtype = cache_dtype

    def set_cache_dtype_given_quantization_encoding(
        self,
    ) -> None:
        """Determine the KV cache dtype based on quantization encoding configuration.

        The dtype is determined in the following priority order:
        1. Explicit override from kv_cache.kv_cache_format (if set)
        2. Derived from the model's quantization_encoding
        3. Falls back to float32 if no encoding is specified

        Returns:
            The DType to use for the KV cache. Typical values are:
            - DType.float32 for float32, q4_k, q4_0, q6_k encodings
            - DType.bfloat16 for bfloat16, float8_e4m3fn, float4_e2m1fnx2, gptq encodings
        """
        # First check for an explicit override.
        if self.kv_cache.kv_cache_format is not None:
            return  # No default needed, override is set.

        # If there's no quantization encoding return a default value.
        if not self.quantization_encoding:
            self.kv_cache._cache_dtype = DType.float32
            return

        # Otherwise select the default KV cache dtype based on the quantization encoding.
        supported_encoding_to_cache_dtype = {
            SupportedEncoding.float32: DType.float32,
            SupportedEncoding.bfloat16: DType.bfloat16,
            SupportedEncoding.float8_e4m3fn: DType.bfloat16,
            SupportedEncoding.float4_e2m1fnx2: DType.bfloat16,
            SupportedEncoding.q4_k: DType.float32,
            SupportedEncoding.q4_0: DType.float32,
            SupportedEncoding.q6_k: DType.float32,
            SupportedEncoding.gptq: DType.bfloat16,
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
