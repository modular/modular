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

import importlib.resources
import json
from collections.abc import Sequence

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest


class AxolotlBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Axolotl dataset from local file."""
        # Set default dataset path if not provided
        if self.dataset_path is None:
            with importlib.resources.path(
                ".".join(__name__.split(".")[:-1]) + ".packaged_data",
                "axolotl_dummy.json",
            ) as path:
                # FIXME: In some installation scenarios, this path is a
                # temporary file that will be deleted when we exit the context
                # manager.
                self.dataset_path = str(path)

        # Call parent fetch method to validate path exists
        super().fetch()

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        """Sample requests from an Axolotl-formatted dataset.
        The dataset should be in the following JSON format:
        [
            {
                "segments": [
                    {
                        "label": true,
                        "text": "human text..."
                    },
                    {
                        "label": false,
                        "text": "assistant text..."
                    }
                ]
            },
            ...
        ]
        This function extracts all text segments where label is false (assistant responses).
        Reference:
        https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/template_free.html
        Args:
            num_requests: Number of requests to sample
            tokenizer: Tokenizer for computing token lengths
            output_lengths: Optional list of request lengths for outputs
        Returns:
            Sampled requests
        """
        assert self.dataset_path is not None, (
            "dataset_path must be provided for AxolotlBenchmarkDataset"
        )
        # Load the dataset
        with open(self.dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        # Extract all text segments where label is false
        prompts = []
        for conversation in dataset:
            for segment in conversation["segments"]:
                if not segment["label"]:
                    prompts.append(segment["text"])

        print("Total number of prompts:", len(prompts))

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            # Randomly sample with replacement
            sampled_prompts = np.random.choice(
                prompts, size=num_requests, replace=True
            )
        else:
            num_repeats = int(np.ceil(num_requests / len(prompts)))
            sampled_prompts = np.array((prompts * num_repeats)[0:num_requests])

        sampled_requests: list[SampledRequest] = []
        for i, prompt in enumerate(sampled_prompts):
            prompt_len = len(tokenizer(prompt).input_ids)
            output_len = None if output_lengths is None else output_lengths[i]
            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=(output_len is not None),
                )
            )
        return RequestSamples(requests=sampled_requests)
