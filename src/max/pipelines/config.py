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

"""Standardized config for Pipeline Inference."""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import random
import struct
import time
from abc import abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union, cast, get_type_hints

import huggingface_hub
import torch
from huggingface_hub import constants as hf_hub_constants
from huggingface_hub import errors as hf_hub_errors
from huggingface_hub.utils import tqdm as hf_tqdm
from max.driver import (
    DeviceSpec,
    devices_exist,
    scan_available_devices,
)
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm.contrib.concurrent import thread_map
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")


class PipelineEngine(str, Enum):
    MAX = "max"
    HUGGINGFACE = "huggingface"


class SupportedEncoding(str, Enum):
    """All possible encodings which may be supported by a particular model."""

    float32 = "float32"
    bfloat16 = "bfloat16"
    q4_k = "q4_k"
    q4_0 = "q4_0"
    q6_k = "q6_k"
    gptq = "gptq"

    @classmethod
    def parse_from_file_name(cls, name: str):
        # TODO(AITLIB-127): Robustify detection of quantization encoding
        name = name.lower()
        if "f32" in name or "float32" in name:
            return SupportedEncoding.float32
        elif "bf16" in name or "bfloat16" in name:
            return SupportedEncoding.bfloat16
        elif "q4_k_m" in name:
            return SupportedEncoding.q4_k
        elif "q4_0" in name:
            return SupportedEncoding.q4_0
        elif "q6_k" in name:
            return SupportedEncoding.q6_k
        elif "gptq" in name:
            return SupportedEncoding.gptq
        else:
            return None

    @property
    def quantization_encoding(self) -> Optional[QuantizationEncoding]:
        if self not in _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding QuantizationEncoding."
            )
        return _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING[self]

    @property
    def dtype(self) -> DType:
        """The underlying model dtype associated with a quantization_encoding."""
        if self not in _SUPPORTED_ENCODING_TO_DTYPE:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding dtype."
            )
        return _SUPPORTED_ENCODING_TO_DTYPE[self]

    @property
    def cache_dtype(self) -> DType:
        """The underlying dtype used in the kvcache for correctness."""
        if self not in _SUPPORTED_ENCODING_TO_CACHE_DTYPE:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding cache dtype."
            )

        return _SUPPORTED_ENCODING_TO_CACHE_DTYPE[self]

    def supported_on(self, device_spec: DeviceSpec) -> bool:
        """Returns whether this quantization encoding is supported on a device."""
        return device_spec.device_type in _SUPPORTED_DEVICES[self]


_SUPPORTED_ENCODING_TO_DTYPE = {
    SupportedEncoding.float32: DType.float32,
    SupportedEncoding.bfloat16: DType.bfloat16,
    SupportedEncoding.q4_k: DType.uint8,
    SupportedEncoding.q4_0: DType.uint8,
    SupportedEncoding.q6_k: DType.uint8,
    SupportedEncoding.gptq: DType.uint8,
}


_SUPPORTED_ENCODING_TO_CACHE_DTYPE = {
    SupportedEncoding.float32: DType.float32,
    SupportedEncoding.bfloat16: DType.bfloat16,
    SupportedEncoding.q4_k: DType.float32,
    SupportedEncoding.q4_0: DType.float32,
    SupportedEncoding.q6_k: DType.float32,
    SupportedEncoding.gptq: DType.bfloat16,
}

_SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING = {
    SupportedEncoding.float32: None,
    SupportedEncoding.bfloat16: None,
    SupportedEncoding.q4_k: QuantizationEncoding.Q4_K,
    SupportedEncoding.q4_0: QuantizationEncoding.Q4_0,
    SupportedEncoding.q6_k: QuantizationEncoding.Q6_K,
    SupportedEncoding.gptq: QuantizationEncoding.GPTQ,
}


# Basic validation for supported devices for each type of encoding.
_SUPPORTED_DEVICES: dict[SupportedEncoding, tuple[str, ...]] = {
    SupportedEncoding.float32: ("cpu", "gpu"),
    SupportedEncoding.bfloat16: ("gpu",),
    SupportedEncoding.q4_k: ("cpu",),
    SupportedEncoding.q4_0: ("cpu",),
    SupportedEncoding.q6_k: ("cpu",),
    SupportedEncoding.gptq: ("gpu",),
}


class RepoType(str, Enum):
    online = "online"
    local = "local"


# Reference: https://github.com/ggerganov/llama.cpp/blob/eb5c3dc64bd967f2e23c87d9dec195f45468de60/src/llama.cpp#L20778
class RopeType(str, Enum):
    none = "none"
    normal = "normal"
    neox = "neox"


def _repo_exists_with_retry(repo_id: str) -> bool:
    """
    Wrapper around huggingface_hub.repo_exists with retry logic.
    Uses exponential backoff with 25% jitter, starting at 1s and doubling each retry.

    See huggingface_hub.repo_exists for details
    """
    max_attempts = 5
    base_delays = [2**i for i in range(max_attempts)]
    retry_delays_in_seconds = [
        d * (1 + random.uniform(-0.25, 0.25)) for d in base_delays
    ]

    for attempt, delay_in_seconds in enumerate(retry_delays_in_seconds):
        try:
            return huggingface_hub.repo_exists(repo_id)
        except (
            hf_hub_errors.RepositoryNotFoundError,
            hf_hub_errors.GatedRepoError,
            hf_hub_errors.RevisionNotFoundError,
            hf_hub_errors.EntryNotFoundError,
        ) as e:
            # Forward these specific errors to the user
            logger.error(f"Hugging Face repository error: {str(e)}")
            raise
        except (hf_hub_errors.HfHubHTTPError, RequestsConnectionError) as e:
            if attempt == max_attempts - 1:
                logger.error(
                    f"Failed to connect to Hugging Face Hub after {max_attempts} attempts: {str(e)}"
                )
                raise

            logger.warning(
                f"Transient Hugging Face Hub connection error (attempt {attempt + 1}/{max_attempts}): {str(e)}"
            )
            logger.warning(
                f"Retrying Hugging Face connection in {delay_in_seconds} seconds..."
            )
            time.sleep(delay_in_seconds)

    assert False, (
        "This should never be reached due to the raise in the last attempt"
    )


@dataclass
class HuggingFaceRepo:
    repo_id: str
    trust_remote_code: bool = False
    repo_type: Optional[RepoType] = None

    def __post_init__(self) -> None:
        # Get repo type.
        if not self.repo_type:
            if os.path.exists(self.repo_id):
                self.repo_type = RepoType.local
            else:
                self.repo_type = RepoType.online

        if self.repo_type == RepoType.online and not _repo_exists_with_retry(
            self.repo_id
        ):
            raise ValueError(f"model_path: {self.repo_id} does not exist")

    def __str__(self) -> str:
        return self.repo_id

    def __repr__(self) -> str:
        return self.repo_id

    @cached_property
    def info(self) -> huggingface_hub.ModelInfo:
        if self.repo_type == RepoType.local:
            raise ValueError(
                "using model info, on local repos is not supported."
            )
        elif self.repo_type == RepoType.online:
            return huggingface_hub.model_info(
                self.repo_id, files_metadata=False
            )
        else:
            raise ValueError(f"Unsupported repo type: {self.repo_type}")

    @cached_property
    def weight_files(self) -> dict[WeightsFormat, list[str]]:
        safetensor_search_pattern = "*.safetensors"
        gguf_search_pattern = "*.gguf"
        pytorch_search_pattern = "*.bin"

        weight_files = {}
        if self.repo_type == RepoType.local:
            safetensor_paths = glob.glob(
                os.path.join(self.repo_id, safetensor_search_pattern)
            )
            gguf_paths = glob.glob(
                os.path.join(self.repo_id, gguf_search_pattern)
            )
            pytorch_paths = glob.glob(
                os.path.join(self.repo_id, pytorch_search_pattern)
            )
        elif self.repo_type == RepoType.online:
            fs = huggingface_hub.HfFileSystem()
            safetensor_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{safetensor_search_pattern}"),
            )
            gguf_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{gguf_search_pattern}"),
            )
            pytorch_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{pytorch_search_pattern}"),
            )
        else:
            raise ValueError(f"Unsupported repo type: {self.repo_type}")

        if safetensor_paths:
            if len(safetensor_paths) == 1:
                # If there is only one weight allow any name.
                weight_files[WeightsFormat.safetensors] = [
                    safetensor_paths[0].replace(f"{self.repo_id}/", "")
                ]
            else:
                # If there is more than one weight, ignore consolidated tensors.
                weight_files[WeightsFormat.safetensors] = [
                    f.replace(f"{self.repo_id}/", "")
                    for f in safetensor_paths
                    if "consolidated" not in f
                ]

        if gguf_paths:
            weight_files[WeightsFormat.gguf] = [
                f.replace(f"{self.repo_id}/", "") for f in gguf_paths
            ]

        if pytorch_paths:
            weight_files[WeightsFormat.pytorch] = [
                f.replace(f"{self.repo_id}/", "") for f in pytorch_paths
            ]

        return weight_files

    def size_of(self, filename: str) -> Union[int, None]:
        if self.repo_type == RepoType.online:
            url = huggingface_hub.hf_hub_url(self.repo_id, filename)
            metadata = huggingface_hub.get_hf_file_metadata(url)
            return metadata.size
        raise NotImplementedError("not implemented for non-online repos.")

    @cached_property
    def supported_encodings(self) -> list[SupportedEncoding]:
        # TODO(AITLIB-128): Detection of supported encodings in weights can be cleaned up
        supported_encodings = set([])

        # Parse gguf file names.
        for gguf_path in self.weight_files.get(WeightsFormat.gguf, []):
            encoding = SupportedEncoding.parse_from_file_name(gguf_path)
            if encoding:
                supported_encodings.add(encoding)

        # Get Safetensor Metadata.
        if WeightsFormat.safetensors in self.weight_files:
            if self.repo_type == RepoType.local:
                # Safetensor repos are assumed to only have one encoding in them.
                with open(
                    os.path.join(
                        self.repo_id,
                        self.weight_files[WeightsFormat.safetensors][0],
                    ),
                    "rb",
                ) as file:
                    # Read the first 8 bytes of the file
                    length_bytes = file.read(8)
                    # Interpret the bytes as a little-endian unsigned 64-bit integer
                    length_of_header = struct.unpack("<Q", length_bytes)[0]
                    # Read length_of_header bytes
                    header_bytes = file.read(length_of_header)
                    # Interpret the bytes as a JSON object
                    header = json.loads(header_bytes)

                    encoding = None
                    for weight_value in header.values():
                        if weight_dtype := weight_value.get("dtype", None):
                            if weight_dtype == "F32":
                                supported_encodings.add(
                                    SupportedEncoding.float32
                                )
                            elif weight_dtype == "BF16":
                                supported_encodings.add(
                                    SupportedEncoding.bfloat16
                                )
                            else:
                                logger.warning(
                                    f"unknown dtype found in safetensors file: {weight_dtype}"
                                )

            elif self.repo_type == RepoType.online:
                if safetensors_info := self.info.safetensors:
                    for params in safetensors_info.parameters:
                        if "BF16" in params:
                            supported_encodings.add(SupportedEncoding.bfloat16)
                        elif "F32" in params:
                            supported_encodings.add(SupportedEncoding.float32)
                if safetensors_config := self.info.config:
                    if quant_config := safetensors_config.get(
                        "quantization_config"
                    ):
                        if quant_config["quant_method"] == "gptq":
                            supported_encodings.add(SupportedEncoding.gptq)
            else:
                raise ValueError(f"Unsupported repo_type: {self.repo_type}")

        # Get torch dtype for pytorch files.
        if WeightsFormat.pytorch in self.formats_available:
            cfg = AutoConfig.from_pretrained(
                self.repo_id, trust_remote_code=self.trust_remote_code
            )

            if torch_dtype := getattr(cfg, "torch_dtype", None):
                if torch_dtype == torch.float32:
                    supported_encodings.add(SupportedEncoding.float32)
                elif torch_dtype == torch.bfloat16:
                    supported_encodings.add(SupportedEncoding.bfloat16)
            else:
                logger.warning(
                    "torch_dtype not available, cant infer encoding from config.json"
                )

        return list(supported_encodings)

    def _get_gguf_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        files = []
        for gguf_file in self.weight_files.get(WeightsFormat.gguf, []):
            file_encoding = SupportedEncoding.parse_from_file_name(gguf_file)
            if file_encoding == encoding:
                files.append(Path(gguf_file))

        if files:
            return {WeightsFormat.gguf: files}
        else:
            return {}

    def _get_safetensor_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if (
            WeightsFormat.safetensors in self.weight_files
            and encoding == self.supported_encodings[0]
        ):
            return {
                WeightsFormat.safetensors: [
                    Path(f)
                    for f in self.weight_files[WeightsFormat.safetensors]
                ]
            }

        return {}

    def _get_pytorch_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if (
            WeightsFormat.pytorch in self.weight_files
            and encoding == self.supported_encodings[0]
        ):
            return {
                WeightsFormat.pytorch: [
                    Path(f) for f in self.weight_files[WeightsFormat.pytorch]
                ]
            }

        return {}

    def files_for_encoding(
        self,
        encoding: SupportedEncoding,
        weights_format: Optional[WeightsFormat] = None,
        alternate_encoding: Optional[SupportedEncoding] = None,
    ) -> dict[WeightsFormat, list[Path]]:
        if weights_format == WeightsFormat.pytorch:
            logger.warning(
                "cannot infer encoding from .bin files, returning all bin files"
            )
            return self._get_pytorch_files_for_encoding(encoding)

        if weights_format is WeightsFormat.gguf:
            return self._get_gguf_files_for_encoding(encoding)
        elif weights_format == WeightsFormat.safetensors:
            return self._get_safetensor_files_for_encoding(encoding)

        gguf_files = self._get_gguf_files_for_encoding(encoding)

        safetensor_files = self._get_safetensor_files_for_encoding(encoding)
        gguf_files.update(safetensor_files)

        pytorch_files = self._get_pytorch_files_for_encoding(encoding)
        gguf_files.update(pytorch_files)

        if not gguf_files and alternate_encoding:
            logger.warning(
                "Could not find checkpoint with %s encoding, searching for %s files instead.",
                encoding,
                alternate_encoding,
            )
            return self.files_for_encoding(alternate_encoding, weights_format)
        return gguf_files

    def file_exists(self, filename: str) -> bool:
        return huggingface_hub.file_exists(self.repo_id, filename)

    def download(self, filename: str, force_download: bool = False) -> Path:
        return Path(
            huggingface_hub.hf_hub_download(
                self.repo_id, filename, force_download=force_download
            )
        )

    @property
    def formats_available(self) -> list[WeightsFormat]:
        return list(self.weight_files.keys())

    def encoding_for_file(self, file: Union[str, Path]) -> SupportedEncoding:
        if str(file).endswith(".safetensors"):
            # If this file is safetensors, return the first encoding, as Safetensor repos can only have one.
            return self.supported_encodings[0]
        elif str(file).endswith(".gguf"):
            encoding = SupportedEncoding.parse_from_file_name(str(file))
            if encoding:
                return encoding

            raise ValueError(
                f"gguf file, but encoding not found in file name: {file}"
            )
        elif str(file).endswith(".bin"):
            # If this file is pytorch, return the first encoding, as Pytorch repos only likely have one.
            return self.supported_encodings[0]
        else:
            raise ValueError(
                f"weight path: {file} not gguf or safetensors, cannot infer encoding from file."
            )


@dataclass
class MAXConfig:
    """Abstract base class for all MAX configs.

    There are some invariants that MAXConfig classes should follow:
    - All config classes should be dataclasses.
    - All config classes should have a help() method that returns a dictionary of config
    options and their descriptions.
    - All config classes dataclass fields should have default values, and hence
    can be trivially initialized via `cls()`.
    - All config classes should be frozen (except KVCacheConfig for now), to
    avoid accidental modification of config objects.
    - All config classes must have mutually exclusive dataclass fields among
    themselves.

    """

    @abstractmethod
    def help(self) -> dict[str, str]:
        """Documentation for this config class. Return a dictionary of config
        options and their descriptions."""
        ...


@dataclass
class MAXModelConfig(MAXConfig):
    """Abstract base class for all MAX model configs.

    This class is used to configure a model to use for a pipeline.
    """

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = ""
    """repo_id of a Hugging Face model repository to use."""

    huggingface_repo_id: str = ""
    """DEPRECATED: repo_id of a Hugging Face model repository to use. Use `model_path` instead."""

    weight_path: list[Path] = field(default_factory=list)
    """Optional path or url of the model weights to use."""

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: Optional[SupportedEncoding] = None
    """Weight encoding type."""

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_revision: str = hf_hub_constants.DEFAULT_REVISION
    """Branch or Git revision of Hugging Face model repository to use."""

    trust_remote_code: bool = False
    """Whether or not to allow for custom modelling files on Hugging Face."""

    device_specs: list[DeviceSpec] = field(
        default_factory=scan_available_devices
    )
    """Devices to run inference upon. This option is not documented in help() as it shouldn't be used directly via the CLI entrypoint."""

    force_download: bool = False
    """Whether to force download a given file if it's already present in the local cache."""

    _weights_repo_id: Optional[str] = None
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    # TODO(zheng): Refactor QuantizationConfig to be a MAXConfig subclass that
    # also autopopulates default values.
    _quant_config: Optional[QuantizationConfig] = None
    """Optional config for specifying quantization parameters. This should only be set by internal code."""

    # TODO(zheng): This can't just be a __post_init__ method, because we need to
    # it also sets and updates other fields which may not be determined /
    # initialized in the default factory.
    # Realistically, this shouldn't become a problem in the long term once we
    # instantiate these MAXConfigs with probably DAG depedency flows in our
    # larger config refactor.
    def validate(self):
        """
        Validate the config.

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

        if self.huggingface_repo_id != "":
            logger.warning(
                "--huggingface-repo-id is deprecated, use `--model-path` instead. This setting will stop working in a future release."
            )
            self.model_path = self.huggingface_repo_id

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
                not _repo_exists_with_retry(self.model_path)
            ):
                raise ValueError(
                    f"{self.model_path} is not a valid Hugging Face repository"
                )
        elif self.model_path == "" and self._weights_repo_id is not None:
            # weight_path is used and we should derive the repo_id from it.
            # At this point, we should have a resolved weight path - be it local or remote HF.
            # weight_path should not be used directly anymore.
            self.model_path = self._weights_repo_id

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """Converts the CLI encoding to a MAX graph quantization encoding.

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

    def finalize_encoding_config(self):
        if self.quantization_encoding == SupportedEncoding.gptq:
            hf_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                revision=self.huggingface_revision,
            )
            hf_quant_config = hf_config.quantization_config

            if hf_config.torch_dtype is not torch.float16:
                raise ValueError(
                    "bfloat16 scales are not supported for GPTQ-quantized models."
                )

            self._quant_config = QuantizationConfig(
                quant_method=hf_quant_config["quant_method"],
                bits=hf_quant_config["bits"],
                group_size=hf_quant_config["group_size"],
                desc_act=hf_quant_config["desc_act"],
                sym=hf_quant_config["sym"],
            )

    @property
    def cache_dtype(self) -> DType:
        if self.quantization_encoding is None:
            raise ValueError(
                "quantization_encoding must be provided to infer cache dtype."
            )

        return self.quantization_encoding.cache_dtype

    def weights_size(self) -> int:
        size = 0
        hf_repo = HuggingFaceRepo(
            (
                self._weights_repo_id
                if self._weights_repo_id
                else self.model_path
            ),
            trust_remote_code=self.trust_remote_code,
        )
        for file_path in self.weight_path:
            if os.path.exists(file_path):
                size += os.path.getsize(file_path)
                continue

            next_size = hf_repo.size_of(str(file_path))

            if next_size is None:
                raise ValueError(
                    f"Failed to get size of weight file {file_path}"
                )
            size += next_size

        return size

    def download_weights(self) -> None:
        # Try to load locally.
        if all([os.path.exists(file_path) for file_path in self.weight_path]):
            logger.info("All files exist locally, skipping download.")
            return

        start_time = datetime.datetime.now()
        weights_repo_id = (
            self._weights_repo_id if self._weights_repo_id else self.model_path
        )
        logger.info(f"Starting download of model: {weights_repo_id}")
        # max_workers=8 setting copied from default for
        # huggingface_hub.snapshot_download.
        self.weight_path = list(
            thread_map(
                lambda filename: Path(
                    huggingface_hub.hf_hub_download(
                        weights_repo_id,
                        str(filename),
                        revision=self.huggingface_revision,
                        force_download=self.force_download,
                    )
                ),
                self.weight_path,
                max_workers=8,
                tqdm_class=hf_tqdm,
            )
        )

        logger.info(
            f"Finished download of model: {weights_repo_id} in {(datetime.datetime.now() - start_time).total_seconds()} seconds."
        )

    def huggingface_weights_repo(self) -> HuggingFaceRepo:
        return HuggingFaceRepo(
            (
                self._weights_repo_id
                if self._weights_repo_id
                else self.model_path
            ),
            trust_remote_code=self.trust_remote_code,
        )

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "model_path": "Specify the repository ID of a Hugging Face model repository to use. This is used to load both Tokenizers, architectures and model weights.",
            "huggingface_repo_id": "DEPRECATED: Use `model_path` instead.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "quantization_encoding": "Define the weight encoding type for quantization. This can help optimize performance and memory usage during inference. ie. q4_k, bfloat16 etc.",
            "huggingface_revision": "Branch or Git revision of Hugging Face model repository to use.",
            "trust_remote_code": "Indicate whether to allow custom modelling files from Hugging Face repositories. Set this to true with caution, as it may introduce security risks.",
            "force_download": "Specify whether to forcefully download a file even if it already exists in local cache. Set this to true if you want to ensure you have the latest version.",
        }


@dataclass
class SamplingConfig(MAXConfig):
    top_k: int = 1
    """Limits the sampling to the K most probable tokens. This defaults to 1, which enables greedy sampling."""

    enable_structured_output: bool = False
    """Enable structured generation/guided decoding for the server. This allows the user to pass a json
    schema in the response_format field, which the LLM will adhere to."""

    in_dtype: DType = DType.float32
    """The data type of the input tokens."""
    out_dtype: DType = DType.float32
    """The data type of the output logits."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "top_k": "Limit sampling to the top K most probable tokens during generation. This can help control randomness and improve output quality. This defaults to 1, which defaults to greedy sampling.",
            "enable_structured_output": "Whether to enable constrained decoding in the text generation pipeline. This defaults to false.",
        }


# frozen is False (for now) because of _available_cache_memory being set by
# internal code.
@dataclass(frozen=False)
class KVCacheConfig(MAXConfig):
    cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT
    """The cache strategy to use. This defaults to `model_default`, which will set the cache
    strategy based on the default strategy for the architecture requested.

    You can also force the engine to use a specific caching strategy: `naive` | `continuous` | `paged`.
    """

    kv_cache_page_size: int = 128
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    device_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace:
        kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size
    """

    _available_cache_memory: Optional[int] = None
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "cache_strategy": "Force a specific cache strategy: 'naive' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 512.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "device_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size. Default is set to 0.9.",
        }


@dataclass
class ProfilingConfig(MAXConfig):
    gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF
    """Whether to enable GPU profiling of the model."""

    def __post_init__(self):
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == GPUProfilingMode.OFF:
            if gpu_profiling_env not in GPUProfilingMode:
                raise ValueError(
                    "gpu_profiling must be one of: "
                    + ", ".join(GPUProfilingMode)
                )
            self.gpu_profiling = GPUProfilingMode(gpu_profiling_env)

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "gpu_profiling": "Whether to turn on GPU profiling for the model. This defaults to 'off'.",
        }


@dataclass(frozen=False)
class PipelineConfig(MAXConfig):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    engine: Optional[PipelineEngine] = None
    """Engine backend to use for serving, 'max' for the max engine, or 'huggingface' as fallback option for improved model coverage."""

    serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    save_to_serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    max_length: Optional[int] = None
    """Maximum sequence length of the model."""

    max_new_tokens: int = -1
    """Maximum number of new tokens to generate during a single inference pass of the model."""

    max_batch_size: Optional[int] = None
    """Maximum batch size to execute with the model.
    This is set to one, to minimize memory consumption for the base case, in which a person is
    running a local server to test out MAX. For users launching in a server scenario, the expectation
    is that this value should be set higher based on server capacity."""

    max_ce_batch_size: int = 192
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and `max_batch_size`."""

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to split context encoding requests into multiple chunks
    based on 'target_num_new_tokens'."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context
    encoding requests. Requires chunked prefill."""

    max_num_steps: int = -1
    """The number of steps to run for multi-step scheduling. -1 specifies a default value based on
    configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding
    models)."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    target_num_new_tokens: Optional[int] = None
    """The target number of un-encoded tokens to include in each batch.
    If not set, this will be set to a best-guess optimal value based on model, hardware, and available memory."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    rope_type: Optional[RopeType] = None
    """Force using a specific rope type: `none` | `normal` | `neox`. Only matters for GGUF weights."""

    pool_embeddings: bool = True
    """Whether to pool embedding outputs."""

    _weight_adapters: dict[WeightsFormat, WeightsAdapter] = field(
        default_factory=dict
    )
    """Weight adapter for the provided `weight_path`."""

    max_cache_batch_size: Optional[int] = None
    """DEPRECATED: The maximum cache batch size to use for the model. Use max_batch_size instead."""

    use_experimental_kernels: str = os.environ.get(
        "USE_EXPERIMENTAL_KERNELS", "false"
    )

    draft_model: Optional[str] = None
    """Draft model for use during Speculative Decoding."""

    _model_config: MAXModelConfig = field(default_factory=MAXModelConfig)
    """The model config."""

    _kv_cache_config: KVCacheConfig = field(default_factory=KVCacheConfig)
    """The KVCache config."""

    _sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    """The sampling config."""

    _profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)

    """The profiling config."""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        # Check if any kwargs are meant for other MAXConfig classes
        unmatched_kwargs: dict[str, Any] = {}
        # Then process kwargs which override defaults
        for key, value in list(kwargs.items()):
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
                del kwargs[key]
            else:
                unmatched_kwargs[key] = value

        # Try to match unmatched kwargs with other config classes
        if unmatched_kwargs:
            # TODO(zheng): Make this more efficient by using MaxConfig instance
            # instead of hardcoding the config names.
            for config_name in [
                "_sampling_config",
                "_kv_cache_config",
                "_profiling_config",
                # TODO(zheng): Remove this once backward compatibility is no
                # longer needed for MAXModelConfig.
                "_model_config",
            ]:
                config_class = get_type_hints(self.__class__)[config_name]
                matched_kwargs = {}
                for key, value in unmatched_kwargs.items():
                    if key in config_class.__dataclass_fields__:
                        matched_kwargs[key] = value
                if matched_kwargs:
                    setattr(self, config_name, config_class(**matched_kwargs))
                    # Remove matched kwargs
                    for key in matched_kwargs:
                        del unmatched_kwargs[key]

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        self.validate()

    def validate(self) -> None:
        """
        Validate the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
        """
        self._model_config.validate()

        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

        if self.max_cache_batch_size is not None:
            logger.warning(
                "--max-cache-batch-size is deprecated, use `--max-batch-size` instead. This setting will stop working in a future release."
            )
            self.max_batch_size = self.max_cache_batch_size

        # Set sensible defaults. These are platform-specific.
        if self.max_num_steps < 0:
            if (
                self.sampling_config.enable_structured_output
                or self._model_config.device_specs[0] == DeviceSpec.cpu()
            ):
                self.max_num_steps = 1
            else:
                self.max_num_steps = 10

        if (
            self.max_num_steps > 1
            and self.sampling_config.enable_structured_output
        ):
            raise ValueError(
                "max_num_steps > 1 not supported when enable_structured_output = True"
            )

        if self.sampling_config.enable_structured_output:
            if self._model_config.device_specs[0] == DeviceSpec.cpu():
                raise ValueError(
                    "enable_structured_output is not currently supported on CPU."
                )

        if self.draft_model:
            if not _repo_exists_with_retry(self.draft_model):
                raise ValueError(
                    "draft_model provided does not exist on HuggingFace."
                    "Only public HuggingFace draft models currently supported."
                )

    def __getstate__(self) -> dict[str, Any]:
        """Override `__getstate__` to exclude the Hugging Face config."""
        state = self.__dict__.copy()
        return state

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """Converts the CLI encoding to a MAX graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.
        """
        return self._model_config.graph_quantization_encoding

    def finalize_encoding_config(self):
        """Depending on the encoding picked, we get some more parameters from the hf config"""
        self._model_config.finalize_encoding_config()

    @staticmethod
    def help() -> dict[str, str]:
        pipeline_help = {
            "engine": "Specify the engine backend to use for serving the model. Options include `max` for the MAX engine, or `huggingface` as a fallback option that provides improved model coverage.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "save_to_serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and `max-cache-batch-size`. Default is 32.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `target-num-new-tokens`",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests. Requires chunked prefill.",
            "max_cache_batch_size": "DEPRECATED: Use `max_batch_size` instead.",
            "rope_type": "Force using a specific rope type, `none` | `normal' | `nexo`. Only matters for GGUF weights.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "pad_to_multiple_of": "Pad input tensors to be a multiple of value provided. Default is set to 2.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
            "draft_model": "Draft model for use in speculative decoding.",
        }

        # Add help text for all MAX config classes
        # TODO(zheng): Make this more efficient by using MaxConfig instance
        # instead of hardcoding the config names.
        for config_class in [SamplingConfig, KVCacheConfig, ProfilingConfig]:
            config_help = config_class.help()  # type: ignore
            for key in config_help:
                if key in pipeline_help:
                    raise ValueError(
                        f"Duplicate help key '{key}' found in {config_class.__name__}"
                    )
            pipeline_help.update(config_help)
        return pipeline_help

    @property
    def model_config(self) -> MAXModelConfig:
        return self._model_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @property
    def kv_cache_config(self) -> KVCacheConfig:
        return self._kv_cache_config

    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling_config
