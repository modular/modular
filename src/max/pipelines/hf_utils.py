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

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import (
    file_exists,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from transformers import AutoConfig


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
