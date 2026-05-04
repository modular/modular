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

from __future__ import annotations

import logging
import os
import random
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest, encode_image_from_file_path

logger = logging.getLogger(__name__)


class _BatchJobImageURL(BaseModel):
    url: str


class _BatchJobTextContent(BaseModel):
    type: Literal["text"]
    text: str


class _BatchJobImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: _BatchJobImageURL


_BatchJobContentItem = Annotated[
    _BatchJobTextContent | _BatchJobImageContent,
    Field(discriminator="type"),
]


class _BatchJobMessage(BaseModel):
    role: str
    content: str | list[_BatchJobContentItem]


class _BatchJobBody(BaseModel):
    messages: list[_BatchJobMessage]
    max_tokens: int | None = None


class _BatchJob(BaseModel):
    body: _BatchJobBody


class BatchJobBenchmarkDataset(LocalBenchmarkDataset):
    """Benchmark dataset for OpenAI Batch API format (tar files with jobs.jsonl and images).

    This dataset supports tar files containing:
    - jobs.jsonl: JSONL file with chat completion requests
    - files/: Directory with images referenced in the requests

    The dataset can operate in two modes:
    1. Embedded mode (default): Images are base64-encoded and included in requests
    2. File reference mode: Images are copied to a server-accessible directory
    """

    def _extract_dataset(self) -> None:
        if not self.dataset_path:
            raise ValueError(
                "BatchJobBenchmarkDataset requires a dataset_path be set"
            )

        if os.path.isdir(self.dataset_path):
            # Dataset is already extracted
            self._dataset_dir = self.dataset_path
            self._jobs_file = os.path.join(self._dataset_dir, "jobs.jsonl")
            self._files_dir = os.path.join(self._dataset_dir, "files")

            if not os.path.exists(self._jobs_file):
                raise ValueError(
                    f"jobs.jsonl not found in directory at {self._jobs_file}"
                )
        else:
            # Extract tar file to temporary directory
            self._extract_dir = tempfile.TemporaryDirectory(prefix="batch_job_")
            extract_path = Path(self._extract_dir.name)

            logger.info(f"Extracting tar file to: {extract_path}")
            with tarfile.open(self.dataset_path, "r") as tar:
                tar.extractall(extract_path)

            # Find the jobs.jsonl file
            extracted_items = os.listdir(extract_path)
            if len(extracted_items) == 1 and os.path.isdir(
                os.path.join(extract_path, extracted_items[0])
            ):
                # Tar contained a single directory
                self._dataset_dir = os.path.join(
                    extract_path, extracted_items[0]
                )
            else:
                # Tar contents extracted directly
                self._dataset_dir = str(extract_path)

            self._jobs_file = os.path.join(self._dataset_dir, "jobs.jsonl")
            self._files_dir = os.path.join(self._dataset_dir, "files")

            if not os.path.exists(self._jobs_file):
                raise ValueError(
                    f"jobs.jsonl not found in extracted tar at {self._jobs_file}"
                )

    def _process_image_content(
        self,
        image_dir: str | None,
        image_url: str,
        processed_content: list[dict[str, Any]],
    ) -> None:
        # Handle file: references
        if image_url.startswith("file:"):
            # Extract the relative path (e.g., "files/image00000000-00000000.jpeg")
            relative_path = image_url.replace("file:", "")
            image_filename = os.path.basename(relative_path)
            source_image = os.path.join(self._dataset_dir, relative_path)

            if image_dir:
                # File reference mode: copy image to server directory
                dest_image = os.path.join(image_dir, image_filename)
                if not os.path.exists(dest_image):
                    shutil.copy2(source_image, dest_image)

                # Keep file reference (server will load from image_dir)
                processed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file:{dest_image}"},
                    }
                )
            else:
                # Embedded mode: encode image as base64
                if os.path.exists(source_image):
                    encoded_img = encode_image_from_file_path(source_image)
                    processed_content.append(dict(encoded_img))
                else:
                    logger.warning(f"Image not found: {source_image}")
        else:
            # Pass through URLs that are already data: or http:
            processed_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = False,
        **kwargs,
    ) -> RequestSamples:
        """Sample requests from batch job dataset.

        Args:
            num_requests: Number of requests to sample
            tokenizer: Tokenizer for computing token lengths
            output_lengths: Optional list of output lengths
            shuffle: Whether to shuffle the dataset
            **kwargs: Additional parameters including:
                - image_dir: Directory where server can access images (for file reference mode)
        """
        # Get optional parameters
        image_dir = kwargs.get("image_dir")

        self._extract_dataset()

        # Load all batch job entries
        batch_jobs: list[_BatchJob] = []
        with open(self._jobs_file) as f:
            for line in f:
                batch_jobs.append(_BatchJob.model_validate_json(line))

        # Shuffle if requested
        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(batch_jobs)

        # Prepare image directory if in file reference mode
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
            logger.info(
                f"Using file reference mode with image directory: {image_dir}"
            )
        else:
            logger.info("Using embedded image mode (base64 encoding)")

        # Sample requests
        sampled_requests: list[SampledRequest] = []
        for i in range(min(num_requests, len(batch_jobs))):
            job = batch_jobs[i]

            # Process the messages to handle images
            processed_messages: list[dict[str, Any]] = []

            for message in job.body.messages:
                processed_content: list[dict[str, Any]] | str
                if isinstance(message.content, list):
                    content_list: list[dict[str, Any]] = []
                    for content_item in message.content:
                        if isinstance(content_item, _BatchJobTextContent):
                            content_list.append(content_item.model_dump())
                        else:
                            self._process_image_content(
                                image_dir,
                                content_item.image_url.url,
                                content_list,
                            )
                    processed_content = content_list
                else:
                    processed_content = message.content

                processed_messages.append(
                    {"role": message.role, "content": processed_content}
                )

            # Compute prompt length (tokenize just the text content)
            text_content = ""
            for msg in processed_messages:
                if isinstance(msg["content"], str):
                    text_content += msg["content"] + " "
                elif isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            text_content += item["text"] + " "

            # TODO: This should also include the image token count.
            prompt_len = len(
                tokenizer.encode(text_content, add_special_tokens=False)
            )

            # Get output length from the batch job's max_tokens
            output_len: int | None
            if output_lengths is not None:
                output_len = output_lengths[i]
                ignore_eos = True
            else:
                # Use max_tokens from the prompt if available
                output_len = job.body.max_tokens
                ignore_eos = False

            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=processed_messages,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=ignore_eos,
                )
            )

        logger.info(
            f"Sampled {len(sampled_requests)} requests from batch job dataset"
        )
        return RequestSamples(requests=sampled_requests)
