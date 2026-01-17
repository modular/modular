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

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
import requests
from huggingface_hub.utils import EntryNotFoundError, OfflineModeIsEnabled
from max.config import load_config
from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    Pipeline,
    RequestID,
)
from requests.exceptions import HTTPError

from ..config_enums import RepoType
from ..interfaces import DiffusionPipeline

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class ImageGenerationPipeline(
    Pipeline[ImageGenerationInputs, ImageGenerationOutput],
):
    """Pipeline wrapper for diffusion image generation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        diffusion_pipeline: type[DiffusionPipeline],
    ) -> None:
        # Download checkpoints if required
        # NOTE: Unlike TextGenerationPipeline where each file,
        # such as configs and weights, are downloaded individually,
        # DiffusionPipeline downloads the entire snapshot at once,
        # since it normally contains multiple components.
        pretrained_model_name_or_path = (
            pipeline_config.model.huggingface_model_repo.repo_id
        )
        if (
            pipeline_config.model.huggingface_model_repo.repo_type
            == RepoType.online
        ):
            cached_folder = self.download(
                pretrained_model_name_or_path,
                config_name=diffusion_pipeline.config_name,
                force_download=pipeline_config.model.force_download,
                revision=pipeline_config.model.huggingface_model_revision,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        self._diffusion_pipeline = diffusion_pipeline(
            pipeline_config, cached_folder
        )

    def download(
        self,
        pretrained_model_name: str | os.PathLike,
        config_name: str | None,
        force_download: bool = False,
        revision: str | None = None,
    ) -> str:
        """Download the pipeline components from the Hugging Face Hub.

        Args:
            pretrained_model_name: Model identifier.
            config_name: Pipeline config filename in the repo.
            force_download: Whether to force download.
            revision: Model revision.

        Returns:
            Path to the downloaded model folder.
        """
        try:
            info = huggingface_hub.model_info(
                pretrained_model_name, revision=revision
            )
        except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
            logger.warning(
                f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache."
            )
            model_info_call_error = (
                e  # save error to reraise it if model is not cached locally
            )

        if config_name is None:
            raise ValueError(
                f"config_name for {pretrained_model_name} pipeline is not set. "
                "Please set proper config file name from huggingface hub."
            )
        try:
            config_file = huggingface_hub.hf_hub_download(
                pretrained_model_name,
                config_name,
                revision=revision,
                force_download=force_download,
            )
        except EntryNotFoundError as e:
            raise ValueError(
                f"config file {config_name} not found for {pretrained_model_name} pipeline. "
                "Please check if the config file name is correct."
            ) from e

        config_dict = load_config(config_file)
        ignore_filenames = config_dict.pop("_ignore_files", [])

        filenames = {sibling.rfilename for sibling in info.siblings}
        filenames = set(filenames) - set(ignore_filenames)

        ignore_patterns = [
            "*.bin",
            "*.msgpack",
            "*.onnx",
            "*.pb",
            "*.bin.index.*json",
            "*.msgpack.index.*json",
            "*.onnx.index.*json",
            "*.pb.index.*json",
        ]

        allow_patterns = ["*/*"]
        allow_patterns += [
            "scheduler_config.json",
            "config.json",
            config_name,
        ]
        re_ignore_pattern = [
            re.compile(fnmatch.translate(p)) for p in ignore_patterns
        ]
        re_allow_pattern = [
            re.compile(fnmatch.translate(p)) for p in allow_patterns
        ]

        expected_files = [
            f
            for f in filenames
            if not any(p.match(f) for p in re_ignore_pattern)
        ]
        expected_files = [
            f
            for f in expected_files
            if any(p.match(f) for p in re_allow_pattern)
        ]

        snapshot_folder = Path(config_file).parent
        pipeline_is_cached = all(
            (snapshot_folder / f).is_file() for f in expected_files
        )

        if pipeline_is_cached and not force_download:
            # if the pipeline is cached, we can directly return it
            # else call snapshot_download
            return snapshot_folder

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = huggingface_hub.snapshot_download(
                pretrained_model_name,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise OSError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

    def execute(self, inputs: ImageGenerationInputs) -> ImageGenerationOutput:
        outputs = self._diffusion_pipeline(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            true_cfg_scale=inputs.true_cfg_scale,
            height=inputs.height,
            width=inputs.width,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            num_images_per_prompt=inputs.num_images_per_prompt,
        )
        return ImageGenerationOutput(images=outputs.images)

    def release(self, request_id: RequestID) -> None:
        pass
