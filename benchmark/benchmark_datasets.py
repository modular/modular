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

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import msgspec
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from sample_workload_utils import (
    CODE_DEBUG_TEMPLATE,
    ChatSession,
    CodeDebugLine,
    SampledRequest,
    build_chat_message,
    encode_image,
)
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("benchmark_datasets")


@dataclass
class DatasetRegistryEntry:
    """Registry entry for a benchmark dataset.

    Attributes:
        class_name: The name of the BenchmarkDataset subclass that implements this dataset.
        has_multiturn_chat_support: Whether this dataset supports multiturn chat scenarios.
    """

    class_name: str
    has_multiturn_chat_support: bool


"""Registry mapping dataset names to their implementation metadata and capabilities.

This registry serves as the central configuration for all supported benchmark datasets
in the serving benchmarking framework. It enables dynamic class resolution and
capability discovery without requiring explicit imports or hardcoded conditionals.

Structure:
    Each registry entry maps a dataset name (str) to a DatasetRegistryEntry containing:

    - class_name (str): The name of the BenchmarkDataset subclass that implements
      this dataset. The class must be defined in the global namespace of this module.

    - has_multiturn_chat_support (bool): Whether this dataset can generate or handle
      multiturn conversational scenarios. This affects which benchmarking modes
      and evaluation patterns can be applied to the dataset.

Adding New Datasets:
    To register a new dataset:

    1. Implement a new BenchmarkDataset subclass with required methods:
       - _fetch_dataset_from_hf() for HuggingFace Hub integration
       - sample_requests() for generating benchmark requests

    2. Add an entry to this registry:
       "my_dataset": DatasetRegistryEntry(
           class_name="MyDatasetBenchmarkDataset",
           has_multiturn_chat_support=True,  # or False
       )

Notes:
    - The registry is loaded at module import time and should not be modified
      at runtime unless you understand the implications for ongoing benchmarks
    - Multiturn support affects memory usage and complexity of benchmark scenarios
    - Some datasets may support multiturn technically but set it to False due
      to domain-specific constraints or intended usage patterns
"""
# TODO: Add BenchmarkDataset modes for HF, local path only, and local path with HF.
# These modes are hacked around to account for these dataset states via the
# _fetch_dataset_from_hf() method.
DATASET_REGISTRY: Mapping[str, DatasetRegistryEntry] = {
    "arxiv-summarization": DatasetRegistryEntry(
        class_name="ArxivSummarizationBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "axolotl": DatasetRegistryEntry(
        class_name="AxolotlBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "code_debug": DatasetRegistryEntry(
        class_name="CodeDebugBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "random": DatasetRegistryEntry(
        class_name="RandomBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "sharegpt": DatasetRegistryEntry(
        class_name="ShareGPTBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "sonnet": DatasetRegistryEntry(
        class_name="SonnetBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "vision-arena": DatasetRegistryEntry(
        class_name="VisionArenaBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
}


# TODO: Enforce a sample_requests @abstractmethod interface.
class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    This class provides a common interface for working with different types of
    benchmark datasets, whether they are fetched from HuggingFace Hub or loaded
    from local files. It handles automatic dataset fetching, validation, and
    provides a standardized interface for sampling requests.

    Attributes:
        dataset_name (Optional[str]): Name of the dataset to fetch from HuggingFace Hub.
            If provided without dataset_path, the dataset will be automatically
            downloaded during initialization.
        dataset_path (Optional[str]): Local path to the dataset file. Takes precedence
            over dataset_name if both are provided. This allows for local datasets
            to be used for benchmarking without having to query / download from HuggingFace Hub.
        has_multiturn_chat_support (bool): Whether this dataset supports multiturn
            chat scenarios.

    Usage:
        Subclasses must implement _fetch_dataset_from_hf() to specify how to download
        and sample their specific datasets. Subclasses must also implement sample_requests()
        to specify how to sample requests from the dataset.

        Example initialization patterns:

        # Auto-fetch from HuggingFace Hub
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

        # Use local file
        dataset = BenchmarkDataset.from_flags(dataset_path="/path/to/local/dataset.json")

        # Sample requests
        requests = dataset.sample_requests(
            num_requests=100,
            tokenizer=tokenizer,
            input_len=1024,
            output_len=512
        )

    Subclass Requirements:
        - Must implement _fetch_dataset_from_hf(dataset_name: str) -> None
        - Must implement sample_requests(**kwargs) -> list[SampledRequest]
        - Should raise ValueError for unsupported dataset names
        - May raise NotImplementedError if HuggingFace fetching is not supported
        - Should handle only dataset types relevant to their domain

    Raises:
        ValueError: If neither dataset_name nor dataset_path is provided during initialization
    """

    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    has_multiturn_chat_support: bool = False

    @classmethod
    def from_flags(
        cls,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ) -> BenchmarkDataset:
        """Factory method to create the appropriate dataset subclass based on dataset name.

        This factory method automatically selects and instantiates the correct
        BenchmarkDataset subclass based on the provided dataset_name. This eliminates
        the need for callers to know which specific subclass to instantiate.

        Args:
            dataset_name (Optional[str]): Name of the dataset. Used to determine
                which subclass to instantiate. If None, dataset_path must be provided.
            dataset_path (Optional[str]): Local path to the dataset file. If provided,
                takes precedence over automatic fetching.

        Returns:
            BenchmarkDataset: An instance of the appropriate subclass

        Raises:
            ValueError: If dataset_name is not recognized or if both dataset_name
                and dataset_path are None

        Example:
            # Creates ShareGPTBenchmarkDataset instance
            dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

            # Creates CodeDebugBenchmarkDataset instance
            dataset = BenchmarkDataset.from_flags(dataset_name="code_debug")

            # Creates appropriate subclass with local file
            dataset = BenchmarkDataset.from_flags(
                dataset_name="sharegpt",
                dataset_path="/local/file.json"
            )
        """
        if not dataset_name and not dataset_path:
            raise ValueError(
                "Either dataset_name or dataset_path must be provided"
            )
        elif dataset_path is not None and not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")

        # If we have a dataset_path but no dataset_name, we can't determine the subclass
        if not dataset_name:
            raise ValueError(
                "dataset_name is required to determine the appropriate dataset subclass. "
                "Cannot infer subclass from dataset_path alone."
            )

        # Get the dataset class based on dataset_name
        dataset_class = cls._get_dataset_class(dataset_name)

        instance = dataset_class()
        instance.dataset_name = dataset_name
        instance.dataset_path = dataset_path
        instance.has_multiturn_chat_support = DATASET_REGISTRY[
            dataset_name
        ].has_multiturn_chat_support
        # TODO(PAQ-1075): This is a temporary interface limitation that's worth revisiting.
        # We should also add support for non-HF datasets here via a new API.
        if instance.dataset_name and instance.dataset_path is None:
            instance._fetch_dataset_from_hf(instance.dataset_name)

        return instance

    @classmethod
    def _get_dataset_class(cls, dataset_name: str) -> type[BenchmarkDataset]:
        """Get the appropriate dataset class for the given dataset name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            The appropriate BenchmarkDataset subclass

        Raises:
            ValueError: If dataset_name is not recognized
        """

        if dataset_name not in DATASET_REGISTRY:
            available_datasets = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {available_datasets}"
            )

        # Dynamically resolve the class at runtime
        dataset_entry = DATASET_REGISTRY[dataset_name]
        class_name = dataset_entry.class_name
        dataset_class = globals().get(class_name)

        if dataset_class is None:
            raise ValueError(f"Dataset class {class_name} not found")

        return dataset_class

    def __str__(self) -> str:
        """Return a user-friendly string representation of the dataset.

        Returns:
            String representation showing dataset path (if local) or name and class type
        """
        class_name = self.__class__.__name__
        if self.dataset_path:
            return f"local_dataset_at_{self.dataset_path} ({class_name})"
        elif self.dataset_name:
            return f"{self.dataset_name} ({class_name})"
        else:
            return f"uninitialized_dataset ({class_name})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        Returns:
            Detailed string representation with all attributes
        """
        return (
            f"{self.__class__.__name__}("
            f"dataset_name='{self.dataset_name}', "
            f"dataset_path='{self.dataset_path}', "
            f"has_multiturn_chat_support={self.has_multiturn_chat_support})"
        )

    @abstractmethod
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        """Fetch dataset from HuggingFace Hub and set dataset_path.

        Args:
            dataset_name: Name of the dataset to fetch

        Raises:
            ValueError: If the dataset is unknown or not supported
        """
        pass


class CodeDebugBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        if dataset_name == "code_debug":
            self.dataset_path = hf_hub_download(
                repo_id="xinrongzhang2022/InfiniteBench",
                filename="code_debug.jsonl",
                repo_type="dataset",
            )
        else:
            raise ValueError(
                f"Unknown dataset for CodeDebugBenchmarkDataset: {dataset_name}"
            )

    def gen_twoturn_longcontext_requests(
        self,
        num_chat_sessions: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Sequence[ChatSession]:
        # Expand code_debug dataset to 2-turn chats with a pre-defined followup question
        DUMMY_OUTPUT = "A"
        CODE_DEBUG_FOLLOWUP_QUESTION = "Explain your reasoning?"
        input_requests = self.sample_requests(
            num_requests=num_chat_sessions,
            tokenizer=tokenizer,
        )

        sessions: list[ChatSession] = []
        for session_id, input_request in enumerate(input_requests):
            messages = [
                build_chat_message(
                    "user", input_request.prompt_formatted, tokenizer
                ),
                # TODO, put correct answers for verification
                # NOTE: Specific single letter answer (2-token)
                build_chat_message("assistant", DUMMY_OUTPUT, tokenizer, 2),
                build_chat_message(
                    "user", CODE_DEBUG_FOLLOWUP_QUESTION, tokenizer
                ),
                build_chat_message("assistant", DUMMY_OUTPUT, tokenizer),
            ]
            sessions.append(ChatSession(session_id, messages))

        return sessions

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
    ) -> Sequence[SampledRequest]:
        """
        The Long-Context dataset workload is based on InfiniteBench Code.debug
        """
        assert self.dataset_path is not None, (
            "dataset_path must be provided for CodeDebugBenchmarkDataset"
        )
        with open(self.dataset_path) as jsonl_file:
            decoded_lines = [
                msgspec.json.decode(json_line, type=CodeDebugLine)
                for json_line in jsonl_file
            ]

        # format context/options/answer -> template of (prompt, completion)
        dataset = [
            (
                self.format_code_debug_context(data),
                self.get_code_debug_answer(data),
            )
            for data in decoded_lines
        ]
        # Filter out the task with LICENSE
        dataset = [data for data in dataset if "LICENSE" not in data[0]]

        # Shuffle the dataset.
        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        filtered_dataset: list[SampledRequest] = []
        model_max_length = tokenizer.model_max_length
        for i in range(len(dataset)):
            if len(filtered_dataset) == num_requests:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion = dataset[i][1]
            completion_token_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if output_lengths is None
                else output_lengths[i]
            )
            assert output_len is not None, "Unexpected null output length"
            if (
                prompt_len > model_max_length
                or prompt_len + output_len > model_max_length
            ):
                # Prune too long sequences.
                print(
                    f"Skip too long sequences ({prompt_len} > {model_max_length})..."
                )
                continue
            filtered_dataset.append(
                SampledRequest(prompt, prompt_len, output_len, None)
            )

        if __debug__:
            from statistics import mean

            list_prompt_len = [data.prompt_len for data in filtered_dataset]
            print(
                f"INFO: Sampled {len(filtered_dataset)} Long-Context Requests: "
                f"Input Tokens(Average: {mean(list_prompt_len)}, "
                f"Min: {min(list_prompt_len)}, Max: {max(list_prompt_len)})"
            )

        return filtered_dataset

    @staticmethod
    def format_code_debug_context(request_features: CodeDebugLine) -> str:
        prompt = CODE_DEBUG_TEMPLATE.format(
            context=request_features.context,
            OPTION_A=request_features.options[0],
            OPTION_B=request_features.options[1],
            OPTION_C=request_features.options[2],
            OPTION_D=request_features.options[3],
        )
        return prompt

    @staticmethod
    def get_code_debug_answer(request_features: CodeDebugLine) -> str:
        if len(request_features.answer) != 1:
            raise ValueError("More than 1 answers")
        OPTIONS = "ABCD"
        return OPTIONS[
            request_features.options.index(request_features.answer[0])
        ]


class ShareGPTBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        if dataset_name == "sharegpt":
            self.dataset_path = hf_hub_download(
                repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
                filename="ShareGPT_V3_unfiltered_cleaned_split.json",
                repo_type="dataset",
            )
        else:
            raise ValueError(
                f"Unknown dataset for ShareGPTBenchmarkDataset: {dataset_name}"
            )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None,
        shuffle: bool,
    ) -> Sequence[SampledRequest]:
        """Sample requests from ShareGPT dataset."""
        assert self.dataset_path is not None, (
            "dataset_path must be provided for ShareGPTBenchmarkDataset"
        )
        # Load the dataset.
        with open(self.dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]

        # Shuffle the dataset.
        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        filtered_dataset: list[SampledRequest] = []
        for i in range(len(dataset)):
            if len(filtered_dataset) == num_requests:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion = dataset[i][1]
            completion_token_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if output_lengths is None
                else output_lengths[len(filtered_dataset)]
            )
            assert output_len is not None, "Unexpected null output length"
            if prompt_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            # If we're given explicit output lengths, then run with whatever
            # we're given. Otherwise, filter requests with super short responses.
            if output_lengths is None and output_len < 4:
                continue
            filtered_dataset.append(
                SampledRequest(prompt, prompt_len, output_len, None)
            )

        return filtered_dataset


class RandomBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        # Random datasets are typically generated synthetically, not fetched from HF
        if dataset_name == "random":
            # No pre-fetching needed - dataset is loaded directly in sample_requests
            pass
        else:
            raise ValueError(
                f"Unknown dataset for RandomBenchmarkDataset: {dataset_name}"
            )

    def gen_multiturn_random_requests(
        self,
        input_len: int,
        output_len: int,
        num_chat_sessions: int,
        num_turns: int,
        coefficient_of_variation: str,
        tokenizer: PreTrainedTokenizerBase,
        sys_prompt_ratio: float,
        max_num_unique_sys_prompt: int,
        distribution_type: str,
        min_input_len: int = 4,
        min_output_len: int = 1,
        first_turn_ratio: float = 1.0,
    ) -> Sequence[ChatSession]:
        first_turns = self.sample_requests(
            int(input_len * first_turn_ratio),
            output_len,
            num_chat_sessions,
            coefficient_of_variation,
            tokenizer,
            sys_prompt_ratio,
            max_num_unique_sys_prompt,
            distribution_type,
            min_input_len,
            min_output_len,
        )

        follow_up_turns = self.sample_requests(
            input_len,
            output_len,
            num_chat_sessions * (num_turns - 1),
            coefficient_of_variation,
            tokenizer,
            0,
            1,
            distribution_type,
            min_input_len,
            min_output_len,
        )

        sessions: list[ChatSession] = []
        for session_id, first_turn in enumerate(first_turns):
            messages = [
                build_chat_message(
                    "user", first_turn.prompt_formatted, tokenizer
                ),
                build_chat_message(
                    "assistant", "", tokenizer, first_turn.output_len
                ),
            ]

            num_turns_this_session = np.random.randint(
                low=int(num_turns / 2), high=num_turns + 1
            )

            for i in range(num_turns_this_session - 1):
                follow_up_turn = follow_up_turns[
                    session_id * (num_turns - 1) + i
                ]
                messages.append(
                    build_chat_message(
                        "user", follow_up_turn.prompt_formatted, tokenizer
                    )
                )
                messages.append(
                    build_chat_message(
                        "assistant", "", tokenizer, follow_up_turn.output_len
                    )
                )

            sessions.append(ChatSession(session_id, messages))

        return sessions

    def sample_requests(
        self,
        input_len: int,
        output_len: int,
        num_prompts: int,
        coefficient_of_variation: str,
        tokenizer: PreTrainedTokenizerBase,
        sys_prompt_ratio: float,
        max_num_unique_sys_prompt: int,
        distribution_type: str,  # TODO: Make distribution_type an enum
        min_input_len: int = 4,
        min_output_len: int = 1,
        image_size: str = "",
    ) -> Sequence[SampledRequest]:
        logger.info(f"Random samples in {distribution_type} distribution")

        if len(coefficient_of_variation.split(",")) == 2:
            input_ratio, output_ratio = map(
                float, coefficient_of_variation.split(",")
            )
            input_scale = input_len * input_ratio
            output_scale = output_len * output_ratio
        else:
            inout_ratio = float(coefficient_of_variation)
            input_scale = input_len * inout_ratio
            output_scale = output_len * inout_ratio

        image_width, image_height = None, None
        if image_size:
            if len(image_size.split(",")) == 2:
                image_width, image_height = map(int, image_size.split(","))
            else:
                raise ValueError(
                    f"Expected image size to be 2 ints separated by a comma, instead got: {image_size}"
                )

        if distribution_type == "normal":
            input_lens = np.random.normal(
                loc=input_len, scale=input_scale, size=num_prompts
            ).tolist()
            input_lens = np.round(input_lens).astype(int).tolist()
            input_lens = [
                max(input_len, min_input_len) for input_len in input_lens
            ]
            output_lens = np.random.normal(
                loc=output_len, scale=output_scale, size=num_prompts
            ).tolist()
            output_lens = np.round(output_lens).astype(int).tolist()
            output_lens = [
                max(output_len, min_output_len) for output_len in output_lens
            ]
        elif distribution_type == "uniform":
            input_scale = min(input_scale, input_len)  # full length cap
            output_scale = min(output_scale, output_len)  # full length cap
            input_lens = np.random.randint(
                max(int(input_scale), min_input_len),
                input_len + 1,
                size=num_prompts,
            )
            output_lens = np.random.randint(
                max(int(output_scale), min_output_len),
                output_len + 1,
                size=num_prompts,
            )
        else:
            raise ValueError(
                f"Unknown probability distribution type: {distribution_type}"
            )

        vocab_size = tokenizer.vocab_size

        sys_prompt_len = np.floor(input_len * sys_prompt_ratio).astype(int)
        sys_prompts = []
        for i in range(max_num_unique_sys_prompt):  # noqa: B007
            sys_prompt = np.random.randint(0, vocab_size, size=sys_prompt_len)
            sys_prompts.append(sys_prompt.tolist())

        input_requests = []
        for i in range(num_prompts):
            sys_prompt_id = np.random.randint(0, max_num_unique_sys_prompt)
            user_prompt_offset = np.random.randint(0, vocab_size)
            user_prompt_len = input_lens[i] - sys_prompt_len
            prompt_ids = sys_prompts[sys_prompt_id] + [
                (user_prompt_offset + i + j) % vocab_size
                for j in range(user_prompt_len)
            ]

            # Remove special tokens from the prompt.
            special_ids = set(tokenizer.all_special_ids)
            replacement = tokenizer.encode(" ", add_special_tokens=False)[0]
            prompt_ids = [
                (replacement if (id in special_ids) else id)
                for id in prompt_ids
            ]
            prompt = tokenizer.decode(prompt_ids)

            image = None
            image_token_len = 0
            if image_size:
                assert image_height is not None
                assert image_width is not None
                raw_image = self._generate_random_image(
                    image_height, image_width
                )
                image = encode_image(raw_image)
                # TODO: figure out how to account for image tokens and chat prompts in this length.
                # For now, just hardcoding to the internvl 512x512 image token count.
                image_token_len = 256

            # We change to use the tokenizer to count the actual number of
            # input tokens encoded on the serving backends instead of looking at
            # int(input_lens[i]) that we randomly generated since multiple
            # input tokens may be bundled together in one pass
            input_len_actual = (
                len(tokenizer(prompt, add_special_tokens=False).input_ids)
                + image_token_len
            )
            input_requests.append(
                SampledRequest(
                    prompt, input_len_actual, int(output_lens[i]), image
                )
            )

        return input_requests

    def _generate_random_image(self, height: int, width: int) -> Image.Image:
        # Truly random images end up being too large and incompressible.
        # Instead create a much more limited block based random image with limited color palette.
        block_size = 16
        colors = np.array([0, 64, 128, 192, 255], dtype=np.uint8)

        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size

        # Generate colors for all blocks
        block_colors = np.random.choice(
            len(colors), size=(blocks_h, blocks_w, 3)
        )
        block_array = colors[block_colors]

        # repeat blocks to create image
        array = np.repeat(
            np.repeat(block_array, block_size, axis=0), block_size, axis=1
        )

        # crop
        array = array[:height, :width]

        return Image.fromarray(array)


class SonnetBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        # Sonnet dataset typically uses local files, not HuggingFace
        raise NotImplementedError(
            "SonnetBenchmarkDataset does not support fetching from HuggingFace Hub. "
            "Sonnet datasets are typically provided as local files."
        )

    def sample_requests(
        self,
        num_requests: int,
        input_len: int,
        output_lengths: Sequence[int] | None,
        prefix_len: int,
        apply_chat_template: bool,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Sequence[SampledRequest]:
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
                SampledRequest(prompt_out, prompt_len, output_len, None)
            )

        return sampled_requests


class AxolotlBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        # Axolotl dataset typically uses local files, not HuggingFace
        raise NotImplementedError(
            "AxolotlBenchmarkDataset does not support fetching from HuggingFace Hub. "
            "Axolotl datasets are typically provided as local files."
        )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None,
        shuffle: bool = True,
    ) -> Sequence[SampledRequest]:
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
            List of SampledRequest objects
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
                SampledRequest(prompt, prompt_len, output_len, None)
            )
        return sampled_requests


class VisionArenaBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        # Vision arena loads dataset directly in sample_requests, not as a separate fetch step
        if dataset_name == "vision-arena":
            # No pre-fetching needed - dataset is loaded directly in sample_requests
            pass
        else:
            raise ValueError(
                f"Unknown dataset for VisionArenaBenchmarkDataset: {dataset_name}"
            )

    def sample_requests(
        self,
        num_requests: int,
        output_lengths: Sequence[int] | None,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Sequence[SampledRequest]:
        dataset = load_dataset(
            "lmarena-ai/vision-arena-bench-v0.1", split="train"
        )
        sampled_requests: list[SampledRequest] = []
        for i in range(num_requests):
            # TODO: Figure out what type to 'assert isinstance' on dataset s.t.
            # MyPy is OK with this (ignored error: Value of type
            # "Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]"
            # is not indexable)
            item = dataset[len(sampled_requests)]  # type: ignore[index]
            prompt = item["turns"][0][0]["content"]
            encoded_img = encode_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            output_len = None if output_lengths is None else output_lengths[i]
            sampled_requests.append(
                SampledRequest(prompt, prompt_len, output_len, encoded_img)
            )
        return sampled_requests


class ArxivSummarizationBenchmarkDataset(BenchmarkDataset):
    def _fetch_dataset_from_hf(self, dataset_name: str) -> None:
        # Arxiv summarization loads dataset directly in sample_requests, not as a separate fetch step
        if dataset_name == "arxiv-summarization":
            # No pre-fetching needed - dataset is loaded directly in sample_requests
            pass
        else:
            raise ValueError(
                f"Unknown dataset for ArxivSummarizationBenchmarkDataset: {dataset_name}"
            )

    def sample_requests(
        self,
        num_requests: int,
        input_len: int,
        max_output_len: int | None,
        shuffle: bool,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Sequence[SampledRequest]:
        """Sample requests from the arxiv-summarization dataset.

        Args:
            num_requests: Number of requests to sample
            input_len: Maximum input length in tokens
            output_len: Target output length in tokens
            tokenizer: Tokenizer for processing text

        Returns:
            Sequence of SampledRequest objects
        """
        # Load the dataset with train split
        dataset = load_dataset("ccdv/arxiv-summarization", split="train")

        # Shuffle the dataset indices
        indices = list(range(len(dataset)))  # type: ignore[arg-type]
        if shuffle:
            random.shuffle(indices)

        # Create a summarization prompt
        prompt_prefix = "Summarize the following research paper:\n\n"
        prompt_suffix = "\n\nSummary:"

        # Calculate tokens for prefix and suffix
        prefix_tokens = tokenizer(
            prompt_prefix, add_special_tokens=False
        ).input_ids
        suffix_tokens = tokenizer(
            prompt_suffix, add_special_tokens=False
        ).input_ids

        # Reserve space for prefix and suffix
        max_article_len = input_len - len(prefix_tokens) - len(suffix_tokens)

        sampled_requests: list[SampledRequest] = []
        for idx in indices:
            if len(sampled_requests) >= num_requests:
                break

            # TODO: Figure out what type to 'assert isinstance' on dataset s.t.
            # MyPy is OK with this (ignored error: Value of type
            # "Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]"
            # is not indexable)
            item = dataset[idx]  # type: ignore[index]
            article = item["article"]

            # Tokenize the article to check length
            article_tokens = tokenizer(
                article, add_special_tokens=False
            ).input_ids

            # Truncate article if necessary
            if len(article_tokens) > max_article_len:
                article_tokens = article_tokens[:max_article_len]
                article = tokenizer.decode(
                    article_tokens, skip_special_tokens=True
                )

            # Create the full prompt
            prompt_formatted = f"{prompt_prefix}{article}{prompt_suffix}"

            # Re-tokenize and get the actual prompt length.
            # Note that the the final prompt size usually does not match
            # len(prefix)+len(suffix)+len(article_tokens) exactly because most
            # tokenizers are not entirely stateless; i.e. adding the prefix
            # changes the behavior. This means the result may be slightly larger
            # than the given input_len (by up to ~10 tokens) despite the
            # truncation logic above. The prompt could of course also be shorter
            # than the given input_len, if the downloaded paper happens to be a
            # small one.
            prompt_len = len(
                tokenizer(prompt_formatted, add_special_tokens=False).input_ids
            )

            # Tokenize the abtsract to get output length.
            abstract = item["abstract"]
            abstract_tokens = tokenizer(
                abstract, add_special_tokens=False
            ).input_ids
            output_len = len(abstract_tokens)

            # Skip outputs that are too large.
            if max_output_len and output_len > max_output_len:
                continue

            sampled_requests.append(
                SampledRequest(prompt_formatted, prompt_len, output_len, None)
            )

        return sampled_requests
