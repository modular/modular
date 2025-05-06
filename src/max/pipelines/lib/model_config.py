# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""MAX config classes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional

import huggingface_hub
import torch
from huggingface_hub import constants as hf_hub_constants
from huggingface_hub import try_to_load_from_cache
from max.driver import DeviceSpec, devices_exist, scan_available_devices
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightsFormat, weights_format
from max.nn.kv_cache import KVCacheStrategy
from transformers import AutoConfig

from .config_enums import (
    _ALTERNATE_ENCODINGS,
    RepoType,
    RopeType,
    SupportedEncoding,
)
from .hf_utils import HuggingFaceRepo, repo_exists_with_retry
from .max_config import KVCacheConfig, MAXConfig
from .registry import PIPELINE_REGISTRY

logger = logging.getLogger("max.pipelines")


@dataclass
class MAXModelConfigBase(MAXConfig):
    """Abstract base class for all (required) MAX model configs.

    This base class is used to configure a model to use for a pipeline, but also
    handy to sidestep the need to pass in optional fields when subclassing
    MAXModelConfig.
    """

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MAXModelConfig(MAXModelConfigBase):
    """Abstract base class for all MAX model configs.

    This class is used to configure a model to use for a pipeline.
    """

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = ""
    """:obj:`repo_id` of a Hugging Face model repository to use."""

    weight_path: list[Path] = field(default_factory=list)
    """Optional path or url of the model weights to use."""

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: Optional[SupportedEncoding] = None
    """Weight encoding type."""

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_model_revision: str = hf_hub_constants.DEFAULT_REVISION
    """Branch or Git revision of Hugging Face model repository to use."""

    huggingface_weight_revision: str = hf_hub_constants.DEFAULT_REVISION
    """Branch or Git revision of Hugging Face model repository to use."""

    trust_remote_code: bool = False
    """Whether or not to allow for custom modelling files on Hugging Face."""

    device_specs: list[DeviceSpec] = field(
        default_factory=scan_available_devices
    )
    """Devices to run inference upon. This option is not documented in :obj:`help()` as it shouldn't be used directly via the CLI entrypoint."""

    force_download: bool = False
    """Whether to force download a given file if it's already present in the local cache."""

    rope_type: Optional[RopeType] = None
    """Force using a specific rope type: `none` | `normal` | `neox`. Only matters for GGUF weights."""

    use_subgraphs: bool = False
    """Whether to use subgraphs for the model."""

    _huggingface_config: Optional[AutoConfig] = None
    """Hugging Face config. This should only be set by internal code."""

    _weights_repo_id: Optional[str] = None
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    # TODO(zheng): Refactor QuantizationConfig to be a MAXConfig subclass that
    # also autopopulates default values.
    _quant_config: Optional[QuantizationConfig] = None
    """Optional config for specifying quantization parameters. This should only be set by internal code."""

    _kv_cache_config: KVCacheConfig = field(default_factory=KVCacheConfig)
    """The KVCache config."""

    # TODO(zheng): This can't just be a __post_init__ method, because we need to
    # it also sets and updates other fields which may not be determined /
    # initialized in the default factory.
    # Realistically, this shouldn't become a problem in the long term once we
    # instantiate these MAXConfigs with probably DAG depedency flows in our
    # larger config refactor.
    def resolve(self) -> None:
        """Validates and resolves the config.

        This method is called after the model config is initialized, to ensure that all
        config fields have been initialized to a valid state. It will also set
        and update other fields which may not be determined / initialized in the
        default factory.
        """
        # Validate that the device_specs provided are available
        if not devices_exist(self.device_specs):
            available_devices = scan_available_devices()
            msg = f"device specs provided ({self.device_specs}) do not exist."
            msg += f"\navailable devices: {available_devices}"
            raise ValueError(msg)

        # Replit model_paths are kinda broken due to transformers
        # version mismatch. We manually update trust_remote_code to True
        # because the modularai version does not have the custom Python code needed
        # Without this, we get:
        #     ValueError: `attn_type` has to be either `multihead_attention` or
        #     `multiquery_attention`. Received: grouped_query_attention
        # Another reason why we override this flag here is because at PipelineConfig
        # instantiation below, we'll call AutoConfig.from_pretrained, which will
        # trigger the error above if not set to True.
        if "replit" in self.model_path.lower():
            self.trust_remote_code = True

        # Validate that if weight_paths are passed as strings, they are converted to Path.
        if isinstance(self.weight_path, tuple):
            self.weight_path = list(self.weight_path)
        elif not isinstance(self.weight_path, list):
            self.weight_path = [self.weight_path]
        weight_paths = []
        # Validate that if weight_paths are passed as strings, they are converted to Path.
        for path in self.weight_path:
            if isinstance(path, str):
                path = Path(path)
            elif not isinstance(path, Path):
                raise ValueError(
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )
            elif path.is_file():
                # If we already exist on the OS. Dont parse the path, just continue.
                weight_paths.append(path)
                continue

            # If the path, looks like it may start with a Hugging Face repo id,
            # check if the repo_id is the same as the one provided.
            # If it is the same, set the weight_path to just be the file_name post repo_id
            # If it is different, set the _weights_repo_id to be that repo_id
            # and set the path to be the file_name without the repo_id.
            if path_pieces := str(path).split("/"):
                if len(path_pieces) >= 3:
                    repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
                    file_name = "/".join(path_pieces[2:])
                    if self.model_path != "" and repo_id == self.model_path:
                        path = Path(file_name)
                    elif huggingface_hub.file_exists(repo_id, file_name):
                        self._weights_repo_id = repo_id
                        path = Path(file_name)
                elif self.model_path == "":
                    raise ValueError(
                        "Unable to derive model_path from weight_path, "
                        "please provide a valid Hugging Face repository id."
                    )

            weight_paths.append(path)

        self.weight_path = weight_paths

        # If we cannot infer the weight path, we lean on the model_path
        # to provide it.
        if len(self.weight_path) == 0:
            if self.model_path == "":
                raise ValueError(
                    "model_path must be provided and must be a valid Hugging Face repository"
                )
            elif (not os.path.exists(self.model_path)) and (
                not repo_exists_with_retry(
                    repo_id=self.model_path,
                    revision=self.huggingface_model_revision,
                )
            ):
                raise ValueError(
                    f"{self.model_path} is not a valid Hugging Face repository"
                )
        elif self.model_path == "" and self._weights_repo_id is not None:
            # weight_path is used and we should derive the repo_id from it.
            # At this point, we should have a resolved weight path - be it local or remote HF.
            # weight_path should not be used directly anymore.
            self.model_path = self._weights_repo_id

        # TODO(KERN-1737): Restore to default after fixing sharded
        # output linear shapes.
        # Set device memory utilization conservatively when tensor parallelism
        # degree is greater than 1.
        # This works around assumptions about uniform weight sharding that would
        # otherwise cause a CUDA OOM.
        if len(self.device_specs) > 1:
            self._kv_cache_config.device_memory_utilization = min(
                0.8, self._kv_cache_config.device_memory_utilization
            )

    @property
    def kv_cache_config(self) -> KVCacheConfig:
        return self._kv_cache_config

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
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
        """Calculates the total size in bytes of all weight files specified in
        `weight_path`.

        This method attempts to find the weights locally first to avoid network
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

            # 1. Check if the file exists locally (direct path, local repo, or cache)
            if local_file_location := self._local_weight_path(file_path):
                total_weights_size += os.path.getsize(local_file_location)
                continue

            # 2. File not found locally or non-existence is cached.
            if repo.repo_type == RepoType.local:
                if not self._local_weight_path(Path(repo.repo_id) / file_path):
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

    @property
    def huggingface_weight_repo_id(self) -> str:
        return (
            self._weights_repo_id if self._weights_repo_id else self.model_path
        )

    @cached_property
    def huggingface_weight_repo(self) -> HuggingFaceRepo:
        return HuggingFaceRepo(
            repo_id=self.huggingface_weight_repo_id,
            revision=self.huggingface_weight_revision,
            trust_remote_code=self.trust_remote_code,
        )

    @cached_property
    def huggingface_model_repo(self) -> HuggingFaceRepo:
        return HuggingFaceRepo(
            repo_id=self.model_path,
            revision=self.huggingface_model_revision,
            trust_remote_code=self.trust_remote_code,
        )

    @property
    def huggingface_config(self) -> AutoConfig:
        # TODO: This is a hack to get around the fact that in serving and the
        # way we instantiate multiprocess model workers, pickling AutoConfig will
        # not work and AutoConfig.from_pretrained will need to be called again
        # when trust_remote_code=True.
        if self.trust_remote_code:
            return AutoConfig.from_pretrained(
                self.huggingface_model_repo.repo_id,
                trust_remote_code=self.huggingface_model_repo.trust_remote_code,
                revision=self.huggingface_model_repo.revision,
            )
        if self._huggingface_config is None:
            self._huggingface_config = (
                PIPELINE_REGISTRY.get_active_huggingface_config(
                    huggingface_repo=self.huggingface_model_repo
                )
            )
        return self._huggingface_config

    def validate_multi_gpu_supported(self, multi_gpu_supported: bool) -> None:
        """Validates that the model architecture supports multi-GPU inference.

        Args:
            multi_gpu_supported: Whether the model architecture supports multi-GPU inference.
        """
        if (
            not multi_gpu_supported
            and len(self.device_specs) > 1
            and self.device_specs[0].device_type == "gpu"
        ):
            raise ValueError(
                f"Multiple GPU inference is currently not supported for {self.model_path}."
            )

    def validate_and_resolve_quantization_encoding_weight_path(
        self, default_encoding: SupportedEncoding
    ) -> None:
        """Verifies that the quantization encoding and weight path provided
        are consistent.
        """

        if os.environ.get("MODULAR_DISABLE_HF_NETWORK_ACCESS", None):
            return

        # If weight_path and quantization_encoding are provided, verify that they are consistent.
        try:
            _weights_format = weights_format(self.weight_path)
        except ValueError:
            _weights_format = None
        if (
            self.weight_path
            and self.quantization_encoding
            # Cannot validate quantization_encoding for pytorch.
            and _weights_format != WeightsFormat.pytorch
        ):
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
                if file_encoding != self.quantization_encoding:
                    msg = f"weight_path provided '{self.weight_path[0]}' has an inconsistent encoding '{file_encoding}' than quantization_encoding provided '{self.quantization_encoding}'. Please update one."
                    raise ValueError(msg)
        # If weight path is not None, infer the quantization_encoding from the weight_path.
        elif (
            self.weight_path
            and not self.quantization_encoding
            and _weights_format != WeightsFormat.pytorch
        ):
            if os.path.exists(self.weight_path[0]):
                # Not currently supported. Infer encoding from local path.
                if self.weight_path[0].suffix == ".safetensors":
                    msg = "If a local safetensors file is provided, please provide a quantization_encoding."
                    raise ValueError(msg)

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
                    msg = f"encoding cannot be inferred from weights file: {self.weight_path[0]}, please pass a quantization_encoding explictly."
                    raise ValueError(msg)
        elif not self.quantization_encoding:
            # Check if the repo only has one quantization_encoding.
            supported_encodings = (
                self.huggingface_weight_repo.supported_encodings
            )
            if len(supported_encodings) == 1:
                msg = f"huggingface repo only has '{supported_encodings[0]}' weights, using '{supported_encodings[0]}'"
                logger.debug(msg)
                self.quantization_encoding = supported_encodings[0]
            elif not self.device_specs[0].device_type == "cpu":
                # TODO(AITLIB-137): replace this with more full featured logic.
                # If we are running on an accelerator and the quantiziation encoding is not set, override to bfloat16.
                if SupportedEncoding.float8_e4m3fn in supported_encodings:
                    self.quantization_encoding = SupportedEncoding.float8_e4m3fn
                elif SupportedEncoding.bfloat16 in supported_encodings:
                    self.quantization_encoding = SupportedEncoding.bfloat16
            else:
                msg = f"encoding not provided, using default encoding of {default_encoding}"
                logger.debug(msg)
                self.quantization_encoding = default_encoding

    def validate_and_resolve_rope_type(self, arch_rope_type: RopeType) -> None:
        if self.rope_type is None:
            self.rope_type = arch_rope_type

    def validate_and_resolve_with_set_quantization_encoding(
        self,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        default_weights_format: WeightsFormat,
    ) -> None:
        """
        Validates that the model path, and weight path
        provided are consistent with a set quantization encoding. Also resolves
        the KV cache strategy and finalizes the encoding config.
        """
        self._validate_quantization_encoding_device_compatibility(
            supported_encodings_list=list(supported_encodings.keys()),
        )
        self._finalize_encoding_config()
        self._resolve_weight_path(default_weights_format=default_weights_format)
        self._resolve_kv_cache_strategy(supported_encodings=supported_encodings)
        self._validate_final_architecture_model_path_weight_path()

    def _validate_quantization_encoding_device_compatibility(
        self,
        supported_encodings_list: list[SupportedEncoding],
    ) -> None:
        """
        Validates that the quantization encoding is supported on the specified
        devices.

        This method should only be called after the quantization encoding has
        been set.
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
                msg = (
                    f"The encoding '{self.quantization_encoding}' is not compatible with the selected device type '{device_spec.device_type}'.\n\n"
                    f"You have two options to resolve this:\n"
                    f"1. Use a different device\n"
                    f"2. Use a different encoding (encodings available for this model: {', '.join(str(enc) for enc in supported_encodings_list)})\n\n"
                    f"Please use the --help flag for more information."
                )

                raise ValueError(msg)

    def _resolve_weight_path(
        self, default_weights_format: WeightsFormat
    ) -> None:
        """
        Resolves the weight path.

        This method should only be called after the quantization encoding has
        been set.

        Args:
            default_weights_format: The default weights format to use if no weight_path is provided.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."

        # If no weight_path is provided, we should grab the default.
        if not self.weight_path:
            # Retrieve the default files for each weights format.

            # Get alternate encoding (e.g. if float32 is requested and there are
            # only bfloat16 weights, allow retrieving the bfloat16 weights
            # because they can be cast to float32).
            if self.quantization_encoding:
                alternate_encoding = _ALTERNATE_ENCODINGS.get(
                    self.quantization_encoding
                )
            else:
                alternate_encoding = None

            weight_files = self.huggingface_weight_repo.files_for_encoding(
                encoding=self.quantization_encoding,
                alternate_encoding=alternate_encoding,
            )

            if default_weight_files := weight_files.get(
                default_weights_format, []
            ):
                self.weight_path = default_weight_files
            elif weight_files:
                # Load any available weight file.
                self.weight_path = next(iter(weight_files.values()))

        if not self.weight_path:
            if self.quantization_encoding not in [
                SupportedEncoding.bfloat16,
                SupportedEncoding.float32,
            ]:
                msg = f"compatible weights cannot be found for '{self.quantization_encoding}' in 'gguf' format, in the provided repo: '{self.huggingface_weight_repo.repo_id}'"
                raise ValueError(msg)
            else:
                msg = f"compatible weights cannot be found for '{self.quantization_encoding}'"
                raise ValueError(msg)

    def _resolve_kv_cache_strategy(
        self,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
    ) -> None:
        """
        Resolves the KVCacheStrategy.

        This method should only be called after the quantization encoding has
        been set.
        """
        assert self.quantization_encoding, "quantization_encoding must be set."
        # Check supported_cache_strategy
        supported_cache_strategies = supported_encodings.get(
            self.quantization_encoding, []
        )
        if (
            self.kv_cache_config.cache_strategy == KVCacheStrategy.MODEL_DEFAULT
            and supported_cache_strategies
        ):
            default_strategy = supported_cache_strategies[0]
            msg = f"default cache_strategy of '{default_strategy}' enabled"
            logger.debug(msg)

            self.kv_cache_config.cache_strategy = default_strategy
        elif (
            supported_cache_strategies
            and self.kv_cache_config.cache_strategy
            not in supported_cache_strategies
        ):
            supported_strategy = supported_cache_strategies[0]

            msg = f"cache_strategy = '{self.kv_cache_config.cache_strategy}' not supported for '{self.quantization_encoding}', using '{supported_strategy}' cache strategy."
            logger.warning(msg)

            self.kv_cache_config.cache_strategy = supported_strategy

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
                    msg = (
                        f"weight_path: '{path_str}' does not exist locally or in cache,"  # noqa: E501
                        f" and '{repo.repo_id}/{path_str}' does"
                        " not exist on HuggingFace."
                    )
                    raise ValueError(msg)
            else:
                raise RuntimeError(
                    f"unexpected repository type: {repo.repo_type}"
                )

    def _finalize_encoding_config(self):
        """
        Finalizes the encoding config.

        This method should only be called after the quantization encoding has
        been set.
        """
        if self.quantization_encoding == SupportedEncoding.gptq:
            hf_quant_config = self.huggingface_config.quantization_config

            if self.huggingface_config.torch_dtype is not torch.float16:
                raise ValueError(
                    "bfloat16 scales are not supported for GPTQ-quantized models."
                )
            default_quantization_config = QuantizationConfig(
                quant_method=hf_quant_config["quant_method"],
                bits=hf_quant_config["bits"],
                group_size=hf_quant_config["group_size"],
                desc_act=hf_quant_config["desc_act"],
                sym=hf_quant_config["sym"],
            )
            self._quant_config = default_quantization_config

    def _local_weight_path(self, relative_path: Path) -> str | None:
        """Checks common local locations for a weight file and returns its
        absolute path if found.

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
                cached_result, (str, os.PathLike)
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

    @staticmethod
    def help() -> dict[str, str]:
        max_model_help = {
            "model_path": "Specify the repository ID of a Hugging Face model repository to use. This is used to load both Tokenizers, architectures and model weights.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "quantization_encoding": "Define the weight encoding type for quantization. This can help optimize performance and memory usage during inference. ie. q4_k, bfloat16 etc.",
            "huggingface_model_revision": "Branch or Git revision of Hugging Face model repository to use.",
            "huggingface_weight_revision": "Branch or Git revision of Hugging Face weight repository to use.",
            "trust_remote_code": "Indicate whether to allow custom modelling files from Hugging Face repositories. Set this to true with caution, as it may introduce security risks.",
            "force_download": "Specify whether to forcefully download a file even if it already exists in local cache. Set this to true if you want to ensure you have the latest version.",
        }

        config_help = KVCacheConfig.help()
        for key in config_help:
            if key in max_model_help:
                raise ValueError(
                    f"Duplicate help key '{key}' found in {KVCacheConfig.__name__}"
                )
        max_model_help.update(config_help)
        return max_model_help
