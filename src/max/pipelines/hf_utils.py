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

import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import (
    file_exists,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")


def get_architectures_from_huggingface_repo(
    model_path: str, trust_remote_code: bool = False
) -> list[str]:
    # Retrieve architecture from model config.
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    return getattr(config, "architectures", [])


@dataclass(frozen=True)
class HuggingFaceFile:
    """A simple object for tracking huggingface model metadata.
    The repo_id will frequently be used to load a tokenizer,
    whereas the filename is used to download model weights."""

    repo_id: str
    filename: str
    revision: str | None = None

    def download(self, force_download: bool = False) -> Path:
        """Download the file and return the file path where the data is saved locally."""
        return Path(
            hf_hub_download(
                self.repo_id,
                self.filename,
                revision=self.revision,
                force_download=force_download,
            )
        )

    def size(self) -> int | None:
        url = hf_hub_url(self.repo_id, self.filename, revision=self.revision)
        metadata = get_hf_file_metadata(url)
        return metadata.size

    def exists(self) -> bool:
        return file_exists(self.repo_id, self.filename, revision=self.revision)


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
          The huggingface model identifier, ie. `modularai/llama-3.1`

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
    weight_paths = list(
        thread_map(
            lambda filename: Path(
                hf_hub_download(
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
