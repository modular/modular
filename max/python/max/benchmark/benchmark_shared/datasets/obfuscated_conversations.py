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

import random
from collections.abc import Sequence

import msgspec
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest


class ObfuscatedConversationsLine(msgspec.Struct):
    timestamp: str
    conversation_id: str
    messages: str


class ObfuscatedConversationsBenchmarkDataset(LocalBenchmarkDataset):
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        # Extract required parameters from kwargs
        seed = kwargs.get("seed")
        if seed is None:
            raise ValueError(
                "seed is required for ObfuscatedConversationsBenchmarkDataset"
            )

        # Validate required parameters
        if output_lengths is None:
            raise ValueError(
                "output_lengths is required for"
                " ObfuscatedConversationsBenchmarkDataset"
            )

        assert self.dataset_path is not None, (
            "dataset_path must be provided for"
            " ObfuscatedConversationsBenchmarkDataset"
        )
        random.seed(seed)
        np.random.seed(seed)

        with open(self.dataset_path) as jsonl_file:
            decoded_lines = [
                msgspec.json.decode(json_line, type=ObfuscatedConversationsLine)
                for json_line in jsonl_file
            ]

        if len(decoded_lines) < num_requests:
            raise ValueError(
                f"Dataset has {len(decoded_lines)} conversations but"
                f" {num_requests} were requested"
            )

        if shuffle:
            conversation_indices = random.choices(
                range(len(decoded_lines)), k=num_requests
            )
        else:
            max_start = max(0, len(decoded_lines) - num_requests)
            start_idx = random.randint(0, max_start)
            conversation_indices = list(
                range(start_idx, start_idx + num_requests)
            )

        sampled_requests: list[SampledRequest] = []
        for i, conversation_idx in enumerate(conversation_indices):
            item = decoded_lines[conversation_idx]
            prompt = item.messages
            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_lengths[i],
                    encoded_images=[],
                    ignore_eos=(output_lengths[i] is not None),
                )
            )
        return RequestSamples(requests=sampled_requests)
