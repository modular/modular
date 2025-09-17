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

import base64
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict

import msgspec
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class OpenAIImageURL(TypedDict):
    url: str


class OpenAIImage(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURL


@dataclass
class SampledRequest:
    prompt_formatted: str
    prompt_len: int
    output_len: int | None
    encoded_img: OpenAIImage | None


MessageSource = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    source: MessageSource
    content: str
    num_tokens: int


@dataclass
class ChatSession:
    id: int | None
    messages: Sequence[ChatMessage]


# -----------------------------------------------------------------------------
# Longcontext Dataset (code_debug)
# -----------------------------------------------------------------------------

CODE_DEBUG_TEMPLATE = """\
There is ONLY ONE function in the large project that is deliberately made to \
include an obvious error. Please find the function that contains the most \
obvious errors. I will give you four options to narrow your scope. You can \
inspect the options and think. Eventually, tell me the answer using one \
single letter (A, B, C, or D).

{context}

Which function has deliberate error?
A. {OPTION_A}
B. {OPTION_B}
C. {OPTION_C}
D. {OPTION_D}

You should first find the functions in the options. Repeat their content, \
inspect through code, and at last give me your answer for the function that \
has the deliberate and obvious error in A, B, C, or D.\
"""


class CodeDebugLine(msgspec.Struct):
    id: int
    context: str
    input: str
    answer: Sequence[str]
    options: Sequence[str]


class ObfuscatedConversationsLine(msgspec.Struct):
    timestamp: str
    conversation_id: str
    messages: str


def encode_image(img: Image.Image) -> OpenAIImage:
    """
    Convert the given PIL.Image.Image to JPEG and encode in base64.
    Returns an openai API image_url content entry with the encoded string.
    """
    img_buffer = BytesIO()
    # Drop alpha channel and convert to jpeg
    img.convert("RGB").save(img_buffer, format="JPEG")
    # Encode in base64 and convert to str
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    # return openai-api dict
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
    }


# -----------------------------------------------------------------------------
# Multi-turn chat
# -----------------------------------------------------------------------------


def estimate_num_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_chat_message(
    source: MessageSource,
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int | None = None,
) -> ChatMessage:
    return ChatMessage(
        source,
        prompt,
        num_tokens or estimate_num_tokens(tokenizer, prompt),
    )


logger = logging.getLogger("benchmark_datasets")

LOCALLY_PACKAGED_DATASETS_DIR = Path(__file__).parent.resolve() / "datasets"


class DatasetMode(str, Enum):
    """Enumeration of supported dataset loading modes.

    This enum defines the different ways datasets can be loaded:
    - LOCAL: Load from a local file path (from environment variable or --dataset-path)
    - HUGGINGFACE: Load from HuggingFace Hub (default behavior)
    """

    LOCAL = "local"
    HUGGINGFACE = "huggingface"


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
       - fetch() for both local and HuggingFace Hub integration
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
    "obfuscated-conversations": DatasetRegistryEntry(
        class_name="ObfuscatedConversationsBenchmarkDataset",
        has_multiturn_chat_support=False,
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


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    This class provides a common interface for working with different types of
    benchmark datasets, whether they are fetched from HuggingFace Hub or loaded
    from local files. It handles automatic dataset fetching, validation, and
    provides a standardized interface for sampling requests.

    Attributes:
        dataset_name (str | None): Name of the dataset to fetch from HuggingFace Hub.
            If provided without dataset_path, the dataset will be automatically
            downloaded during initialization.
        dataset_path (str | None): Local path to the dataset file. Takes precedence
            over dataset_name if both are provided. This allows for local datasets
            to be used for benchmarking without having to query / download from HuggingFace Hub.
        has_multiturn_chat_support (bool): Whether this dataset supports multiturn
            chat scenarios.

    Usage:
        Subclasses must implement fetch() to specify how to download
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
        - Must implement fetch() -> None to handle both local and HuggingFace dataset loading
        - Must implement sample_requests(**kwargs) -> list[SampledRequest]
        - Should raise ValueError for unsupported dataset names or invalid configurations
        - May raise NotImplementedError if the requested mode is not supported
        - Should handle only dataset types relevant to their domain

    Raises:
        ValueError: If neither dataset_name nor dataset_path is provided during initialization
    """

    dataset_name: str | None = None
    dataset_path: str | None = None
    has_multiturn_chat_support: bool = False

    @classmethod
    def from_flags(
        cls,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> BenchmarkDataset:
        """Factory method to create the appropriate dataset subclass based on dataset name.

        This factory method automatically selects and instantiates the correct
        BenchmarkDataset subclass based on the provided dataset_name. This eliminates
        the need for callers to know which specific subclass to instantiate.

        Args:
            dataset_name (str | None): Name of the dataset. Used to determine
                which subclass to instantiate. If None, dataset_path must be provided.
            dataset_path (str | None): Local path to the dataset file. If provided,
                takes precedence over automatic fetching.

        Returns:
            BenchmarkDataset: An instance of the appropriate subclass

        Raises:
            ValueError: If dataset_name is not recognized or if both dataset_name
                and dataset_path are None

        Example:
            # Creates ShareGPTBenchmarkDataset instance from HuggingFace
            dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

            # Creates appropriate subclass with local file
            dataset = BenchmarkDataset.from_flags(
                dataset_name="sharegpt",
                dataset_path="/local/file.json",
            )

            # Creates dataset using environment variable for local path
            dataset = BenchmarkDataset.from_flags(
                dataset_name="sharegpt",
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

        instance.fetch()

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
    def fetch(self) -> None:
        """Fetch dataset based on the current dataset_mode and set dataset_path.

        This method handles both local and HuggingFace dataset loading:
        - For LOCAL mode: Uses dataset_path (from constructor or environment variable)
        - For HUGGINGFACE mode: Downloads dataset from HuggingFace Hub

        Raises:
            ValueError: If the dataset is unknown, not supported, or if both modes are specified
            NotImplementedError: If the dataset doesn't support the requested mode
        """
        pass

    @abstractmethod
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
        """Sample requests from the dataset.

        This is the standardized interface that all dataset implementations must follow.
        Additional dataset-specific parameters can be passed via **kwargs.

        Args:
            num_requests: Number of requests to sample
            tokenizer: Tokenizer for computing token lengths
            output_lengths: Optional sequence of output lengths for each request.
                If None, uses the actual completion lengths from the dataset.
                If provided, must have length equal to num_requests.
            shuffle: Whether to shuffle the dataset before sampling. Default is True.
            **kwargs: Additional dataset-specific parameters

        Returns:
            Sequence of SampledRequest objects

        Raises:
            ValueError: If the dataset cannot be loaded or parameters are invalid
            NotImplementedError: If required parameters are missing for this dataset type
        """
        pass


class LocalBenchmarkDataset(BenchmarkDataset):
    """Abstract base class for local benchmark datasets.

    This class provides a common interface for working with local benchmark datasets.
    It handles automatic dataset fetching and provides a standardized interface for sampling requests.
    """

    dataset_mode: DatasetMode = DatasetMode.LOCAL

    def fetch(self) -> None:
        # For local mode, dataset_path should already be set and validated
        if self.dataset_path is None:
            raise ValueError("For LOCAL mode, dataset_path must be provided")
        if not os.path.exists(self.dataset_path):
            raise ValueError(
                f"Local dataset path {self.dataset_path} does not exist"
            )


class HuggingFaceBenchmarkDataset(BenchmarkDataset):
    """Abstract base class for HuggingFace benchmark datasets.

    This class provides a common interface for working with HuggingFace benchmark datasets.
    It handles automatic dataset fetching and provides a standardized interface for sampling requests.
    """

    dataset_mode: DatasetMode = DatasetMode.HUGGINGFACE

    def fetch(self) -> None:
        pass


class CodeDebugBenchmarkDataset(HuggingFaceBenchmarkDataset):
    def fetch(self) -> None:
        self.dataset_path = hf_hub_download(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename="code_debug.jsonl",
            repo_type="dataset",
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
        **kwargs,
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
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_img=None,
                )
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


class ShareGPTBenchmarkDataset(HuggingFaceBenchmarkDataset):
    def fetch(self) -> None:
        self.dataset_path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
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
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_img=None,
                )
            )

        return filtered_dataset


class RandomBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Random dataset.

        Random datasets are generated synthetically and don't require file fetching.
        """
        pass

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
            num_requests=num_chat_sessions,
            tokenizer=tokenizer,
            input_len=int(input_len * first_turn_ratio),
            output_len=output_len,
            coefficient_of_variation=coefficient_of_variation,
            sys_prompt_ratio=sys_prompt_ratio,
            max_num_unique_sys_prompt=max_num_unique_sys_prompt,
            distribution_type=distribution_type,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
        )

        follow_up_turns = self.sample_requests(
            num_requests=num_chat_sessions * (num_turns - 1),
            tokenizer=tokenizer,
            input_len=input_len,
            output_len=output_len,
            coefficient_of_variation=coefficient_of_variation,
            sys_prompt_ratio=0,
            max_num_unique_sys_prompt=1,
            distribution_type=distribution_type,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
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
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
        # Extract required parameters from kwargs
        input_len = kwargs.get("input_len")
        output_len = kwargs.get("output_len")
        coefficient_of_variation = kwargs.get("coefficient_of_variation")
        sys_prompt_ratio = kwargs.get("sys_prompt_ratio", 0.0)
        max_num_unique_sys_prompt = kwargs.get("max_num_unique_sys_prompt", 1)
        distribution_type = kwargs.get("distribution_type", "uniform")
        min_input_len = kwargs.get("min_input_len", 4)
        min_output_len = kwargs.get("min_output_len", 1)
        image_size = kwargs.get("image_size", "")

        # Validate required parameters
        if input_len is None:
            raise ValueError("input_len is required for RandomBenchmarkDataset")
        if output_len is None:
            raise ValueError(
                "output_len is required for RandomBenchmarkDataset"
            )
        if coefficient_of_variation is None:
            raise ValueError(
                "coefficient_of_variation is required for RandomBenchmarkDataset"
            )

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
                loc=input_len, scale=input_scale, size=num_requests
            ).tolist()
            input_lens = np.round(input_lens).astype(int).tolist()
            input_lens = [
                max(input_len, min_input_len) for input_len in input_lens
            ]
            output_lens = np.random.normal(
                loc=output_len, scale=output_scale, size=num_requests
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
                size=num_requests,
            )
            output_lens = np.random.randint(
                max(int(output_scale), min_output_len),
                output_len + 1,
                size=num_requests,
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
        for i in range(num_requests):
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
                    prompt_formatted=prompt,
                    prompt_len=input_len_actual,
                    output_len=int(output_lens[i]),
                    encoded_img=image,
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


class SonnetBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Sonnet dataset from local file."""
        # Set default dataset path if not provided
        if self.dataset_path is None:
            self.dataset_path = str(
                LOCALLY_PACKAGED_DATASETS_DIR / "sonnet_4x.txt"
            )

        # Call parent fetch method to validate path exists
        super().fetch()

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
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
                    encoded_img=None,
                )
            )

        return sampled_requests


class AxolotlBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Axolotl dataset from local file."""
        # Set default dataset path if not provided
        if self.dataset_path is None:
            self.dataset_path = str(
                LOCALLY_PACKAGED_DATASETS_DIR / "axolotl_dummy.json"
            )

        # Call parent fetch method to validate path exists
        super().fetch()

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
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
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_img=None,
                )
            )
        return sampled_requests


class VisionArenaBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch VisionArena dataset based on the current dataset_mode.

        VisionArena datasets are loaded directly in sample_requests, not as a separate fetch step.
        """
        pass

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
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
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_img=encoded_img,
                )
            )
        return sampled_requests


class ArxivSummarizationBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch ArxivSummarization dataset based on the current dataset_mode.

        ArxivSummarization datasets are loaded directly in sample_requests, not as a separate fetch step.
        """
        pass

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
        # Extract required parameters from kwargs
        input_len = kwargs.get("input_len")
        max_output_len = kwargs.get("max_output_len")

        # Validate required parameters
        if input_len is None:
            raise ValueError(
                "input_len is required for ArxivSummarizationBenchmarkDataset"
            )

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
                SampledRequest(
                    prompt_formatted=prompt_formatted,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_img=None,
                )
            )

        return sampled_requests


class ObfuscatedConversationsBenchmarkDataset(LocalBenchmarkDataset):
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
        # Extract required parameters from kwargs
        seed = kwargs.get("seed")
        if seed is None:
            raise ValueError(
                "seed is required for ObfuscatedConversationsBenchmarkDataset"
            )

        # Validate required parameters
        if output_lengths is None:
            raise ValueError(
                "output_lengths is required for ObfuscatedConversationsBenchmarkDataset"
            )

        assert self.dataset_path is not None, (
            "dataset_path must be provided for ObfuscatedConversationsBenchmarkDataset"
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
                f"Dataset has {len(decoded_lines)} conversations but {num_requests} were requested"
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
                    encoded_img=None,
                )
            )
        return sampled_requests
