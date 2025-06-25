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

"""Utilities for interacting with HuggingFace Files/Repos."""

from __future__ import annotations

import contextlib
import datetime
import glob
import json
import logging
import os
import random
import struct
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union, cast

import huggingface_hub
import torch
from huggingface_hub import errors as hf_hub_errors
from huggingface_hub.utils import tqdm as hf_tqdm
from max.graph.weights import WeightsFormat
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm.contrib.concurrent import thread_map
from tqdm.std import TqdmDefaultWriteLock
from transformers import AutoConfig

from .config_enums import (
    RepoType,
    SupportedEncoding,
)

logger = logging.getLogger("max.pipelines")


def _create_gated_repo_error_message(repo_id: str, original_error: str) -> str:
    """Create a user-friendly error message for gated repository access issues."""
    return (
        f"Repository '{repo_id}' exists but requires authentication. "
        f"This is a gated/private repository that requires an access token. "
        f"Please ensure you have:\n"
        f"1. A valid Hugging Face access token with appropriate permissions\n"
        f"2. The token is properly configured (via 'huggingface-cli login' or HF_TOKEN environment variable)\n"
        f"3. You have been granted access to this model\n\n"
        f"Original error: {original_error}"
    )


def _create_repo_not_found_error_message(
    repo_id: str, revision: str, original_error: str
) -> str:
    """Create a user-friendly error message for repository not found issues."""
    return (
        f"Repository '{repo_id}' not found. Please check:\n"
        f"1. The repository ID is correct\n"
        f"2. The repository exists on Hugging Face\n"
        f"3. The revision '{revision}' exists\n\n"
        f"Original error: {original_error}"
    )


def _create_repo_access_fallback_error_message(
    repo_id: str, original_error: str
) -> str:
    """Create a user-friendly fallback error message for repository access issues."""
    return (
        f"Failed to access repository '{repo_id}'. "
        f"This could be due to network issues, invalid repository, or authentication problems.\n\n"
        f"Original error: {original_error}"
    )


def _create_repo_not_exists_error_message(repo_id: str, revision: str) -> str:
    """Create a user-friendly error message when _repo_exists_with_retry returns False."""
    return (
        f"Repository '{repo_id}' not found. Please check:\n"
        f"1. The repository ID is correct\n"
        f"2. The repository exists on Hugging Face\n"
        f"3. The revision '{revision}' exists"
    )


def try_to_load_from_cache(
    repo_id: str, filename: str, revision: str
) -> Union[str, Any, None]:
    """
    Wrapper around huggingface_hub.try_to_load_from_cache. We also validate
    that the repo exists.

    validate_hf_repo_access is called before this function to ensure the repo
    exists.
    """
    validate_hf_repo_access(repo_id=repo_id, revision=revision)
    return huggingface_hub.try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )


def validate_hf_repo_access(repo_id: str, revision: str) -> None:
    """
    Validate repository access and raise clear, user-friendly errors.

    Args:
        repo_id: The HuggingFace repository ID to validate
        revision: The revision/branch to validate

    Raises:
        ValueError: With user-friendly error messages for various access issues
    """
    try:
        repo_exists = _repo_exists_with_retry(
            repo_id=repo_id, revision=revision
        )
        if not repo_exists:
            raise ValueError(
                _create_repo_not_exists_error_message(repo_id, revision)
            )
    except hf_hub_errors.GatedRepoError as e:
        raise ValueError(
            _create_gated_repo_error_message(repo_id, str(e))
        ) from e
    except (
        hf_hub_errors.RepositoryNotFoundError,
        hf_hub_errors.RevisionNotFoundError,
        hf_hub_errors.EntryNotFoundError,
    ) as e:
        raise ValueError(
            _create_repo_not_found_error_message(repo_id, revision, str(e))
        ) from e
    except Exception as e:
        # Fallback for other HuggingFace or network errors
        raise ValueError(
            _create_repo_access_fallback_error_message(repo_id, str(e))
        ) from e


class _ThreadingOnlyTqdmLock(TqdmDefaultWriteLock):
    """A version of TqdmDefaultWriteLock that only uses threading locks.

    The tqdm write lock will not be enforced across processes.
    """

    mp_lock = None


@contextlib.contextmanager
def _hf_tqdm_using_threading_only_lock():
    """Use a threading-only lock if there is no existing write lock.

    If a write lock already exists, it is not replaced.  The sole purpose of
    this is to override the default creation of a lock that is problematic in
    this context (as we cannot always ensure proper shutdown of a
    multiprocessing lock, in some cases causing leaks).

    This function exists rather than another hf_tqdm subclass directly
    replacing _lock because Hugging Face internals still use hf_tqdm, and tqdm
    uses class-resident state NOT shared across subclasses, so we have to
    override hf_tqdm directly and cannot use a subclass.
    """
    # N.B.: _lock nonpresence is treated differently than presence with a None
    # value.  Make sure we go down the default path even for None; we only
    # replace the lock if the attribute is not present.  We can't use the
    # public get_lock API for this since that creates the lock we're trying to
    # avoid in the first place.
    if hasattr(hf_tqdm, "_lock"):
        yield
        return
    setattr(hf_tqdm, "_lock", _ThreadingOnlyTqdmLock())
    try:
        yield
    finally:
        delattr(hf_tqdm, "_lock")


def download_weight_files(
    huggingface_model_id: str,
    filenames: list[str],
    revision: Optional[str] = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> list[Path]:
    """Provided a HuggingFace model id, and filenames, download weight files
        and return the list of local paths.

    Args:
        huggingface_model_id:
          The huggingface model identifier, ie. `modularai/Llama-3.1-8B-Instruct-GGUF`

        filenames:
          A list of file paths relative to the root of the HuggingFace repo.
          If files provided are available locally, download is skipped, and
          the local files are used.

        revision:
          The HuggingFace revision to use. If provided, we check our cache
          directly without needing to go to HuggingFace directly, saving a
          network call.

        force_download:
          A boolean, indicating whether we should force the files to be
          redownloaded, even if they are already available in our local cache,
          or a provided path.

        max_workers:
          The number of worker threads to concurrently download files.

    """
    if not force_download and all(
        os.path.exists(Path(filename)) for filename in filenames
    ):
        logger.info("All files exist locally, skipping download.")
        return [Path(filename) for filename in filenames]

    start_time = datetime.datetime.now()
    logger.info(f"Starting download of model: {huggingface_model_id}")
    with _hf_tqdm_using_threading_only_lock():
        weight_paths = list(
            thread_map(
                lambda filename: Path(
                    huggingface_hub.hf_hub_download(
                        huggingface_model_id,
                        filename,
                        revision=revision,
                        force_download=force_download,
                    )
                ),
                filenames,
                max_workers=max_workers,
                tqdm_class=hf_tqdm,
            )
        )

    logger.info(
        f"Finished download of model: {huggingface_model_id} in {(datetime.datetime.now() - start_time).total_seconds()} seconds."
    )

    return weight_paths


def _repo_exists_with_retry(repo_id: str, revision: str) -> bool:
    """
    Wrapper around huggingface_hub.revision_exists with retry logic.
    Uses exponential backoff with 25% jitter, starting at 1s and doubling each retry.

    We use revision_exists here instead of repo_exists because repo_exists
    does not take in a revision parameter.

    See huggingface_hub.revision_exists for details
    """

    max_attempts = 5
    base_delays = [2**i for i in range(max_attempts)]
    retry_delays_in_seconds = [
        d * (1 + random.uniform(-0.25, 0.25)) for d in base_delays
    ]

    for attempt, delay_in_seconds in enumerate(retry_delays_in_seconds):
        try:
            return huggingface_hub.revision_exists(
                repo_id=repo_id, revision=revision
            )
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
            # Do not retry if Too Many Requests error received
            if e.response.status_code == 429:
                logger.error(e)
                raise

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


@dataclass(frozen=True)
class HuggingFaceRepo:
    """
    A class for interacting with HuggingFace Repos.
    """

    repo_id: str
    """The HuggingFace repo id. While it's called repo_id, it can be a HF
    remote or local path altogether."""

    revision: str = huggingface_hub.constants.DEFAULT_REVISION
    """The revision to use for the repo."""

    trust_remote_code: bool = False
    """Whether to trust remote code."""

    repo_type: Optional[RepoType] = None
    """The type of repo. This is inferred from the repo_id."""

    def __post_init__(self) -> None:
        # Get repo type.
        if not self.repo_type:
            if os.path.exists(self.repo_id):
                object.__setattr__(self, "repo_type", RepoType.local)
            else:
                object.__setattr__(self, "repo_type", RepoType.online)

        if self.repo_type == RepoType.online:
            validate_hf_repo_access(
                repo_id=self.repo_id, revision=self.revision
            )

    def __str__(self) -> str:
        return self.repo_id

    def __repr__(self) -> str:
        return self.repo_id

    def __hash__(self) -> int:
        return hash(
            (
                self.repo_id,
                self.revision,
                self.trust_remote_code,
                self.repo_type,
            )
        )

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
                list[str], fs.glob(f"{self.repo_id}/{gguf_search_pattern}")
            )
            pytorch_paths = cast(
                list[str], fs.glob(f"{self.repo_id}/{pytorch_search_pattern}")
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
                            elif weight_dtype == "F8_E4M3":
                                supported_encodings.add(
                                    SupportedEncoding.float8_e4m3fn
                                )
                            else:
                                logger.warning(
                                    f"unknown dtype found in safetensors file: {weight_dtype}"
                                )

            elif self.repo_type == RepoType.online:
                if safetensors_info := self.info.safetensors:
                    for params in safetensors_info.parameters:
                        if "F8_E4M3" in params:
                            supported_encodings.add(
                                SupportedEncoding.float8_e4m3fn
                            )
                        elif "BF16" in params:
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
                self.repo_id,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
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
            and encoding in self.supported_encodings
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


# TODO: Over time we'd like to extend this into a new HFAssetResolver class that
# automatically handles locally cached vs. remotely fetched artifacts via
# specified repo_ids and revisions.
def generate_local_model_path(repo_id: str, revision: str) -> str:
    """Generate the local filesystem path where a HuggingFace model repo is cached.

    This function takes a HuggingFace repository ID and revision hash and returns the full local
    filesystem path where the model files are cached by the huggingface_hub library. The path
    follows the standard HuggingFace caching convention of:
    ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{revision}

    Args:
        repo_id: The HuggingFace repository ID in the format "org/model"
                (e.g. "HuggingFaceTB/SmolLM2-135M")
        revision: The specific model revision hash to use, typically from a repo lock file

    Returns:
        str: The absolute path to the cached model files for the specified revision.
            For example: "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/abc123"

    Raises:
        FileNotFoundError: If the model path does not exist locally
    """
    # Convert repo_id to format used in local path
    temp_model_path = repo_id.replace("/", "--")
    # Build full path with revision hash and expand user directory
    path = (
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        / "hub"
        / f"models--{temp_model_path}"
        / "snapshots"
        / revision
    ).expanduser()

    if not path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    return str(path)
