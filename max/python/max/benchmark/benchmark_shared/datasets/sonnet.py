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
import random
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest


class SonnetBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Sonnet dataset from local file."""
        # Set default dataset path if not provided
        if self.dataset_path is None:
            with importlib.resources.path(
                ".".join(__name__.split(".")[:-1]) + ".packaged_data",
                "sonnet_4x.txt",
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
        # Extract required parameters from kwargs
        input_len = kwargs.get("input_len")
        prefix_len = kwargs.get("prefix_len")
        apply_chat_template = kwargs.get("apply_chat_template")

        # Validate required parameters
        if input_len is None:
            raise ValueError("input_len is required for SonnetBenchmarkDataset")
        if prefix_len is None:
            raise ValueError(
                "prefix_len is required for SonnetBenchmarkDataset"
            )
        if apply_chat_template is None:
            raise ValueError(
                "apply_chat_template is required for SonnetBenchmarkDataset"
            )

        assert input_len > prefix_len, (
            "input_len must be greater than prefix_len."
        )

        assert self.dataset_path is not None, (
            "dataset_path must be provided for SonnetBenchmarkDataset"
        )

        # Load the dataset.
        with open(self.dataset_path) as f:
            poem_lines = f.readlines()

        # Tokenize the poem lines.
        poem_token_ids = tokenizer(poem_lines).input_ids
        average_poem_len = sum(
            len(token_ids) for token_ids in poem_token_ids
        ) / len(poem_token_ids)

        # Base prefix for all requests.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_message = [
            {
                "role": "user",
                "content": base_prompt,
            }
        ]
        base_prompt_formatted = tokenizer.apply_chat_template(
            base_message, add_generation_prompt=True, tokenize=False
        )
        base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

        assert input_len > base_prompt_offset, (
            f"input_len must be greater than {base_prompt_offset}."
        )
        num_input_lines = round(
            (input_len - base_prompt_offset) / average_poem_len
        )

        # First approximately `prefix_len` number of tokens in the
        # prompt are fixed poem lines.
        assert prefix_len > base_prompt_offset, (
            f"prefix_len must be greater than {base_prompt_offset}."
        )

        num_prefix_lines = round(
            (prefix_len - base_prompt_offset) / average_poem_len
        )
        prefix_lines = poem_lines[:num_prefix_lines]

        # Sample the rest of lines per request.
        sampled_requests: list[SampledRequest] = []
        for i in range(num_requests):
            sampled_lines = "".join(
                prefix_lines
                + random.sample(poem_lines, num_input_lines - num_prefix_lines)
            )

            prompt = f"{base_prompt}{sampled_lines}"
            message = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )
            # TODO: Figure out why MyPy can't figure this type out otherwise
            assert isinstance(prompt_formatted, str)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            prompt_out = prompt_formatted if apply_chat_template else prompt
            output_len = None if output_lengths is None else output_lengths[i]
            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=prompt_out,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=(output_len is not None),
                )
            )

        return RequestSamples(requests=sampled_requests)
