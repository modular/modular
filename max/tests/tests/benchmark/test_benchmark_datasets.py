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

"""Unit tests for benchmark_shared.datasets module."""

import json
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from max.benchmark.benchmark_shared.datasets import (
    DATASET_REGISTRY,
    ArxivSummarizationBenchmarkDataset,
    AxolotlBenchmarkDataset,
    BenchmarkDataset,
    CodeDebugBenchmarkDataset,
    DatasetRegistryEntry,
    HuggingFaceBenchmarkDataset,
    InstructCoderBenchmarkDataset,
    LocalBenchmarkDataset,
    LocalImageBenchmarkDataset,
    ObfuscatedConversationsBenchmarkDataset,
    PixelGenerationSampledRequest,
    RandomBenchmarkDataset,
    SampledRequest,
    SharedContext,
    ShareGPTBenchmarkDataset,
    SonnetBenchmarkDataset,
    SyntheticPixelBenchmarkDataset,
    VisionArenaBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
)

# Import the module under test
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    resolve_constant_delay_ms,
)
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class _FakeTokenizer:
    """Picklable stand-in for `PreTrainedTokenizerBase` used by unit tests.

    `_fake_loader` constructs one per spawn worker; the parent uses an
    instance of this class directly. Behavior is deterministic so tests
    can assert on outputs (no Mock call-count plumbing).
    """

    name_or_path = "_fake_"
    vocab_size = 1000
    unk_token_id = None
    all_special_ids: frozenset[int] = frozenset({0, 1, 2})

    def __init__(self, model_max_length: int = 4096) -> None:
        self.model_max_length = model_max_length

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: object
    ) -> list[int]:
        # Length-aware so multiturn-fit tests that scale messages to a
        # target token count round-trip through encode→decode→encode and
        # see a meaningful encoded length.
        return list(range(max(4, len(text))))

    def decode(
        self, ids: list[int], skip_special_tokens: bool = False, **_: object
    ) -> str:
        return "Z" * len(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 223


def _fake_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> _FakeTokenizer:
    return _FakeTokenizer(model_max_length=model_max_length or 4096)


def test_dataset_registry_structure() -> None:
    """Test that the registry has the expected structure."""
    assert isinstance(DATASET_REGISTRY, dict)
    assert len(DATASET_REGISTRY) > 0

    for dataset_name, dataset_info in DATASET_REGISTRY.items():
        assert isinstance(dataset_name, str)
        assert isinstance(dataset_info, DatasetRegistryEntry)
        assert dataset_info.class_name is not None
        assert dataset_info.has_multiturn_chat_support is not None


def test_dataset_registry_contents() -> None:
    """Test that the registry contains expected datasets."""
    expected_datasets = {
        "agentic-code",
        "arxiv-summarization",
        "chat-judge",
        "instruct-coder",
        "sharegpt",
        "code_debug",
        "random",
        "synthetic",
        "synthetic-pixel",
        "sonnet",
        "vision-arena",
        "axolotl",
        "batch-job",
        "obfuscated-conversations",
        "local-image",
    }
    assert set(DATASET_REGISTRY.keys()) == expected_datasets


def test_dataset_registry_multiturn_support_mapping() -> None:
    """Test that multiturn support is correctly configured."""
    expected_multiturn = {
        "agentic-code": True,
        "arxiv-summarization": False,
        "code_debug": True,
        "instruct-coder": True,
        "random": True,
        "synthetic": True,
        "sharegpt": False,
        "sonnet": False,
        "vision-arena": False,
        "axolotl": False,
        "batch-job": False,
        "synthetic-pixel": False,
        "local-image": False,
    }

    for dataset_name, expected_support in expected_multiturn.items():
        actual_support = DATASET_REGISTRY[
            dataset_name
        ].has_multiturn_chat_support
        assert actual_support == expected_support


def test_dataset_registry_class_names_exist() -> None:
    """Test that all referenced class names exist in globals."""
    from max.benchmark.benchmark_shared import datasets

    for _, dataset_info in DATASET_REGISTRY.items():
        class_name = dataset_info.class_name
        assert hasattr(datasets, class_name), (
            f"Class {class_name} not found in benchmark_shared.datasets"
        )


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_with_dataset_name(
    mock_exists: Mock,
) -> None:
    """Test creating dataset instances using dataset_name."""
    mock_exists.return_value = True  # Mock any downloaded paths exist

    with patch.object(ShareGPTBenchmarkDataset, "fetch"):
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

        assert isinstance(dataset, ShareGPTBenchmarkDataset)
        assert dataset.dataset_name == "sharegpt"
        assert not dataset.has_multiturn_chat_support


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_with_dataset_path(
    mock_exists: Mock,
) -> None:
    """Test creating dataset instances using dataset_path."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )

    assert isinstance(dataset, RandomBenchmarkDataset)
    assert dataset.dataset_name == "random"
    assert dataset.dataset_path == "/path/to/dataset.json"
    assert dataset.has_multiturn_chat_support
    mock_exists.assert_called_once_with("/path/to/dataset.json")


def test_benchmark_dataset_from_flags_no_parameters() -> None:
    """Test that providing neither dataset_name nor dataset_path raises error."""
    with pytest.raises(
        ValueError, match="Either dataset_name or dataset_path must be provided"
    ):
        BenchmarkDataset.from_flags()


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_no_dataset_name(
    mock_exists: Mock,
) -> None:
    """Test that providing only dataset_path without dataset_name raises error."""
    mock_exists.return_value = True  # Mock path exists

    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(dataset_path="/path/to/dataset.json")

    mock_exists.assert_called_once_with("/path/to/dataset.json")
    assert "dataset_name is required" in str(exc_info.value)


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_nonexistent_path(
    mock_exists: Mock,
) -> None:
    """Test that providing a nonexistent dataset_path raises error."""
    mock_exists.return_value = False  # Mock path doesn't exist

    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(
            dataset_name="random", dataset_path="/nonexistent/path.json"
        )

    assert "Dataset path /nonexistent/path.json does not exist" in str(
        exc_info.value
    )
    mock_exists.assert_called_once_with("/nonexistent/path.json")


def test_benchmark_dataset_from_flags_unknown_dataset() -> None:
    """Test that unknown dataset names raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(dataset_name="unknown_dataset")

    assert "Unknown dataset: unknown_dataset" in str(exc_info.value)
    assert "Available datasets:" in str(exc_info.value)


def test_benchmark_dataset_get_dataset_class() -> None:
    """Test the _get_dataset_class method."""
    dataset_class = BenchmarkDataset._get_dataset_class("sharegpt")
    assert dataset_class == ShareGPTBenchmarkDataset

    dataset_class = BenchmarkDataset._get_dataset_class("code_debug")
    assert dataset_class == CodeDebugBenchmarkDataset


def test_benchmark_dataset_get_dataset_class_unknown() -> None:
    """Test _get_dataset_class with unknown dataset name."""
    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset._get_dataset_class("unknown_dataset")

    assert "Unknown dataset: unknown_dataset" in str(exc_info.value)


@patch("os.path.exists")
def test_benchmark_dataset_str_with_dataset_name(mock_exists: Mock) -> None:
    """Test __str__ method with dataset_name."""
    mock_exists.return_value = True  # Mock any downloaded paths exist

    with patch.object(ShareGPTBenchmarkDataset, "fetch"):
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")
        str_repr = str(dataset)
        assert "sharegpt" in str_repr
        assert "ShareGPTBenchmarkDataset" in str_repr


@patch("os.path.exists")
def test_benchmark_dataset_str_with_dataset_path(mock_exists: Mock) -> None:
    """Test __str__ method with dataset_path."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )
    str_repr = str(dataset)
    assert "local_dataset_at_/path/to/dataset.json" in str_repr
    assert "RandomBenchmarkDataset" in str_repr


@patch("os.path.exists")
def test_benchmark_dataset_repr(mock_exists: Mock) -> None:
    """Test __repr__ method."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )
    repr_str = repr(dataset)
    assert "RandomBenchmarkDataset" in repr_str
    assert "dataset_name='random'" in repr_str
    assert "dataset_path='/path/to/dataset.json'" in repr_str
    assert "has_multiturn_chat_support=True" in repr_str


@patch("os.path.exists")
@patch("max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download")
def test_code_debug_from_flags(mock_download: Mock, mock_exists: Mock) -> None:
    """Test fetching code_debug dataset from HuggingFace Hub."""
    mock_download.return_value = "/path/to/downloaded/file.jsonl"
    mock_exists.return_value = True  # Mock downloaded file exists

    dataset = BenchmarkDataset.from_flags(dataset_name="code_debug")

    assert dataset.dataset_path == "/path/to/downloaded/file.jsonl"
    mock_download.assert_called_once_with(
        repo_id="xinrongzhang2022/InfiniteBench",
        filename="code_debug.jsonl",
        repo_type="dataset",
    )


@patch("max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download")
def test_code_debug_from_flags_unknown(mock_download: Mock) -> None:
    """Test fetching unknown dataset raises ValueError."""
    mock_download.return_value = "/tmp/fake_code_debug.jsonl"
    dataset = BenchmarkDataset.from_flags(dataset_name="code_debug")
    # CodeDebugBenchmarkDataset now inherits from HuggingFaceBenchmarkDataset
    # and will fetch from HF by default, so we test the fetch method directly
    with patch.object(CodeDebugBenchmarkDataset, "fetch") as mock_fetch:
        mock_fetch.side_effect = ValueError(
            "Unknown dataset for CodeDebugBenchmarkDataset: unknown_dataset"
        )
        with pytest.raises(ValueError, match="Unknown dataset"):
            dataset.fetch()


@patch("os.path.exists")
@patch("max.benchmark.benchmark_shared.datasets.sharegpt.hf_hub_download")
def test_sharegpt_from_flags(mock_download: Mock, mock_exists: Mock) -> None:
    """Test fetching ShareGPT dataset from HuggingFace Hub."""
    mock_download.return_value = "/path/to/downloaded/file.json"
    mock_exists.return_value = True  # Mock downloaded file exists

    dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

    assert dataset.dataset_path == "/path/to/downloaded/file.json"
    mock_download.assert_called_once_with(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )


def test_random_from_flags() -> None:
    """Test that RandomBenchmarkDataset HF fetching being essentially a no-op."""
    # This shouldn't raise an error because random dataset is not fetched from HF.
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)
    assert dataset.dataset_name == "random"
    assert dataset.dataset_path is None


def test_random_sample_requests() -> None:
    """Test sampling random requests."""
    tok = _FakeTokenizer(model_max_length=50)
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=2,
            tokenizer=tok,
            pool=pool,
            input_len="N(50, 5)",
            output_len="N(20, 2)",
            sys_prompt_ratio=0.1,
            max_num_unique_sys_prompt=1,
        )

    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, SampledRequest)


def test_shared_contexts_empty_when_no_sys_prompt() -> None:
    """shared_contexts is empty when sys_prompt_ratio is 0."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=5,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
        )

    assert samples.shared_contexts == []


def test_shared_contexts_one_entry_per_unique_idx() -> None:
    """shared_contexts has exactly one SharedContext per unique sys_prompt_idx."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=10,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.3,
            max_num_unique_sys_prompt=1,
        )

    assert len(samples.shared_contexts) == 1
    assert isinstance(samples.shared_contexts[0], SharedContext)


def test_shared_contexts_at_most_max_unique() -> None:
    """shared_contexts has at most max_num_unique_sys_prompt entries."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    max_unique = 3
    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=30,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.3,
            max_num_unique_sys_prompt=max_unique,
        )

    assert len(samples.shared_contexts) <= max_unique
    for entry in samples.shared_contexts:
        assert isinstance(entry, SharedContext)
        assert entry.num_tokens > 0


@patch("os.path.exists")
def test_sonnet_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that SonnetBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="sonnet", dataset_path="/path/to/sonnet.txt"
    )

    assert isinstance(dataset, SonnetBenchmarkDataset)
    assert dataset.dataset_name == "sonnet"
    assert dataset.dataset_path == "/path/to/sonnet.txt"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_vision_arena_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that VisionArenaBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="vision-arena", dataset_path="/path/to/vision_arena.json"
    )

    assert isinstance(dataset, VisionArenaBenchmarkDataset)
    assert dataset.dataset_name == "vision-arena"
    assert dataset.dataset_path == "/path/to/vision_arena.json"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_axolotl_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that AxolotlBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="axolotl", dataset_path="/path/to/axolotl.json"
    )

    assert isinstance(dataset, AxolotlBenchmarkDataset)
    assert dataset.dataset_name == "axolotl"
    assert dataset.dataset_path == "/path/to/axolotl.json"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_arxiv_summarization_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that ArxivSummarizationBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="arxiv-summarization", dataset_path="/path/to/arxiv.json"
    )

    assert isinstance(dataset, ArxivSummarizationBenchmarkDataset)
    assert dataset.dataset_name == "arxiv-summarization"
    assert dataset.dataset_path == "/path/to/arxiv.json"
    assert dataset.dataset_mode == "local"


def test_vision_arena_from_flags_unknown() -> None:
    """Test fetching unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        BenchmarkDataset.from_flags(dataset_name="vision-arena-unknown")


@patch("os.path.exists")
def test_obfuscated_conversations_from_flags_local_mode(
    mock_exists: Mock,
) -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="obfuscated-conversations",
        dataset_path="/path/to/obfuscated.jsonl",
    )

    assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)
    assert dataset.dataset_name == "obfuscated-conversations"
    assert dataset.dataset_path == "/path/to/obfuscated.jsonl"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_obfuscated_conversations_missing_seed(mock_exists: Mock) -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset requires seed parameter."""
    mock_exists.return_value = True

    # Mock tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.vocab_size = 1000
    mock_tokenizer.all_special_ids = {0, 1, 2}
    mock_tokenizer.encode.return_value = [100]
    mock_tokenizer.decode.return_value = "random text"
    mock_tokenizer.return_value.input_ids = [100, 101, 102]

    dataset = BenchmarkDataset.from_flags(
        dataset_name="obfuscated-conversations",
        dataset_path="/path/to/obfuscated.jsonl",
    )
    assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)

    # Test that missing seed parameter raises ValueError
    with pytest.raises(
        ValueError,
        match="seed is required for ObfuscatedConversationsBenchmarkDataset",
    ):
        dataset.sample_requests(
            num_requests=2,
            tokenizer=mock_tokenizer,
            output_lengths=[10, 20],
        )


def test_obfuscated_conversations_with_seed() -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset works with seed parameter."""
    # Mock tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.vocab_size = 1000
    mock_tokenizer.all_special_ids = {0, 1, 2}
    mock_tokenizer.encode.return_value = [100]
    mock_tokenizer.decode.return_value = "random text"
    mock_tokenizer.return_value.input_ids = [100, 101, 102]

    # Test that providing seed parameter works
    with (
        patch(
            "builtins.open",
            mock_open(
                read_data='{"timestamp": "2023-01-01", "conversation_id": "conv1", "messages": "test message"}\n{"timestamp": "2023-01-02", "conversation_id": "conv2", "messages": "test message 2"}\n'
            ),
        ),
        patch("os.path.exists", return_value=True),
    ):
        dataset = BenchmarkDataset.from_flags(
            dataset_name="obfuscated-conversations",
            dataset_path="/path/to/test/dataset.jsonl",
        )
        assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)
        samples = dataset.sample_requests(
            num_requests=2,
            tokenizer=mock_tokenizer,
            output_lengths=[10, 20],
            seed=42,
        )
        assert len(samples.requests) == 2
        for request in samples.requests:
            assert isinstance(request, SampledRequest)


# Tests for new base classes
def test_local_benchmark_dataset_base_class() -> None:
    """Test LocalBenchmarkDataset base class behavior."""
    # Test that LocalBenchmarkDataset has LOCAL mode by default
    assert LocalBenchmarkDataset.dataset_mode == "local"

    # Test that it sets default dataset_path and validates it exists
    dataset = AxolotlBenchmarkDataset()
    dataset.dataset_name = "test"

    # Test that it sets a default path (may or may not exist in test environment)
    dataset.fetch()
    assert dataset.dataset_path is not None
    assert "axolotl_dummy.json" in dataset.dataset_path

    # Test with explicit valid dataset_path
    with patch("os.path.exists", return_value=True):
        dataset.dataset_path = "/path/to/dataset.json"
        dataset.fetch()  # Should not raise


@patch("max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download")
def test_huggingface_benchmark_dataset_base_class(mock_download: Mock) -> None:
    """Test HuggingFaceBenchmarkDataset base class behavior."""
    mock_download.return_value = "/tmp/fake_code_debug.jsonl"
    # Test that HuggingFaceBenchmarkDataset has HUGGINGFACE mode by default
    assert HuggingFaceBenchmarkDataset.dataset_mode == "huggingface"

    # Test that fetch is a no-op by default using a concrete implementation
    dataset = CodeDebugBenchmarkDataset()
    dataset.fetch()  # Should not raise


def test_dataset_inheritance_hierarchy() -> None:
    """Test that datasets inherit from the correct base classes."""
    # Test local datasets
    assert issubclass(AxolotlBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(SonnetBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(RandomBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(VisionArenaBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(ArxivSummarizationBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(
        ObfuscatedConversationsBenchmarkDataset, LocalBenchmarkDataset
    )

    # Test HuggingFace datasets
    assert issubclass(CodeDebugBenchmarkDataset, HuggingFaceBenchmarkDataset)
    assert issubclass(ShareGPTBenchmarkDataset, HuggingFaceBenchmarkDataset)

    # Test that all datasets inherit from BenchmarkDataset
    assert issubclass(LocalBenchmarkDataset, BenchmarkDataset)
    assert issubclass(HuggingFaceBenchmarkDataset, BenchmarkDataset)
    assert issubclass(AxolotlBenchmarkDataset, BenchmarkDataset)
    assert issubclass(CodeDebugBenchmarkDataset, BenchmarkDataset)


def test_local_datasets_require_dataset_path() -> None:
    """Test that local datasets set default dataset_path and validate it exists."""
    # Test AxolotlBenchmarkDataset
    axolotl_dataset = AxolotlBenchmarkDataset()
    axolotl_dataset.dataset_name = "axolotl"

    # Test that it sets a default path
    axolotl_dataset.fetch()
    assert axolotl_dataset.dataset_path is not None
    assert "axolotl_dummy.json" in axolotl_dataset.dataset_path

    # Test SonnetBenchmarkDataset
    sonnet_dataset = SonnetBenchmarkDataset()
    sonnet_dataset.dataset_name = "sonnet"

    # Test that it sets a default path
    sonnet_dataset.fetch()
    assert sonnet_dataset.dataset_path is not None
    assert "sonnet_4x.txt" in sonnet_dataset.dataset_path


def test_huggingface_datasets_fetch_behavior() -> None:
    """Test that HuggingFace datasets fetch from HF when used directly."""
    # Test CodeDebugBenchmarkDataset
    with patch(
        "max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download"
    ) as mock_download:
        mock_download.return_value = "/path/to/downloaded/file.jsonl"

        code_debug_dataset = CodeDebugBenchmarkDataset()
        code_debug_dataset.dataset_name = "code_debug"
        code_debug_dataset.fetch()

        assert (
            code_debug_dataset.dataset_path == "/path/to/downloaded/file.jsonl"
        )
        mock_download.assert_called_once_with(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename="code_debug.jsonl",
            repo_type="dataset",
        )

    # Test ShareGPTBenchmarkDataset
    with patch(
        "max.benchmark.benchmark_shared.datasets.sharegpt.hf_hub_download"
    ) as mock_download:
        mock_download.return_value = "/path/to/downloaded/file.json"

        sharegpt_dataset = ShareGPTBenchmarkDataset()
        sharegpt_dataset.dataset_name = "sharegpt"
        sharegpt_dataset.fetch()

        assert sharegpt_dataset.dataset_path == "/path/to/downloaded/file.json"
        mock_download.assert_called_once_with(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )


def test_local_datasets_with_default_files() -> None:
    """Test that local datasets work with default files when using from_flags."""
    # Test Axolotl with default file
    with patch("os.path.exists", return_value=True):
        dataset = BenchmarkDataset.from_flags(dataset_name="axolotl")
        assert isinstance(dataset, AxolotlBenchmarkDataset)
        assert dataset.dataset_name == "axolotl"
        assert dataset.dataset_mode == "local"
        # Should have set the default path
        assert dataset.dataset_path is not None
        assert "axolotl_dummy.json" in dataset.dataset_path

    # Test Sonnet with default file
    with patch("os.path.exists", return_value=True):
        dataset = BenchmarkDataset.from_flags(dataset_name="sonnet")
        assert isinstance(dataset, SonnetBenchmarkDataset)
        assert dataset.dataset_name == "sonnet"
        assert dataset.dataset_mode == "local"
        # Should have set the default path
        assert dataset.dataset_path is not None
        assert "sonnet_4x.txt" in dataset.dataset_path


def _write_test_image(image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="red").save(image_path)


def test_image_edit_dataset_sample_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    _write_test_image(image_path)
    dataset_path = tmp_path / "image_edit.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Turn this into watercolor",
                "image_path": "images/sample.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = BenchmarkDataset.from_flags(
        dataset_name="local-image",
        dataset_path=str(dataset_path),
    )
    assert isinstance(dataset, LocalImageBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=1,
        tokenizer=None,
        image_width=1024,
        image_steps=28,
        image_guidance_scale=3.5,
    )
    request = samples.requests[0]
    assert isinstance(request, PixelGenerationSampledRequest)
    assert request.prompt_formatted == "Turn this into watercolor"
    assert request.input_image_paths == [str(image_path.resolve())]
    assert request.image_options is not None
    assert request.image_options.width == 1024
    assert request.image_options.steps == 28
    assert request.image_options.guidance_scale == 3.5


def test_image_edit_dataset_invalid_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text("{not-json}\n", encoding="utf-8")
    dataset = BenchmarkDataset.from_flags(
        dataset_name="local-image",
        dataset_path=str(dataset_path),
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        dataset.sample_requests(num_requests=1, tokenizer=None)


def test_synthetic_pixel_dataset_sample_requests_for_image_to_image() -> None:
    dataset = BenchmarkDataset.from_flags(dataset_name="synthetic-pixel")
    assert isinstance(dataset, SyntheticPixelBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=None,
        benchmark_task="image-to-image",
        image_width=640,
        image_height=832,
        image_steps=18,
        image_guidance_scale=3.0,
    )

    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, PixelGenerationSampledRequest)
        assert request.prompt_formatted.startswith("Random prompt")
        assert len(request.input_image_paths) == 1
        assert Path(request.input_image_paths[0]).exists()
        assert request.image_options is not None
        assert request.image_options.width == 640
        assert request.image_options.height == 832
        assert request.image_options.steps == 18
        assert request.image_options.guidance_scale == 3.0


def test_random_multiturn_emits_zero_prefix_turns() -> None:
    """gen_multiturn_random_requests always emits prefix_turns=0; the runner
    owns warmup prefix-turn assignment via _pick_warmup_population."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)
    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.gen_multiturn_random_requests(
            input_len=32,
            output_len=16,
            num_chat_sessions=20,
            num_turns=3,
            delay_between_chat_turns=500,
            pool=pool,
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
        )

    assert all(s.prefix_turns == 0 for s in samples.chat_sessions)


def test_resolve_constant_delay_ms() -> None:
    assert resolve_constant_delay_ms(None) is None
    assert resolve_constant_delay_ms(12) == 12.0
    assert resolve_constant_delay_ms("0") == 0.0


@patch.object(InstructCoderBenchmarkDataset, "_load_pairs")
def test_instruct_coder_multiturn_fit_distributions(
    mock_load_pairs: Mock,
) -> None:
    """``fit_length_distributions`` honors turn count, lengths, and delay."""
    body = "hello world " * 80
    mock_load_pairs.return_value = [(body, "line\n" * 60)] * 400

    tok = _FakeTokenizer(model_max_length=50_000)
    dataset = InstructCoderBenchmarkDataset()
    dataset.dataset_path = "/tmp/instruct_coder_mock.json"

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.gen_multiturn_sessions(
            num_sessions=4,
            tokenizer=tok,
            pool=pool,
            shuffle=False,
            fit_length_distributions=True,
            num_turns="DU(3,3)",
            input_len="80",
            output_len="20",
            delay_between_turns_dist="100",
            sys_prompt_ratio=0.0,
        )

    assert len(samples.chat_sessions) == 4
    for session in samples.chat_sessions:
        assert len(session.messages) == 6
        for assistant in session.messages[1::2]:
            assert assistant.source == "assistant"
            assert assistant.num_tokens == 20
            assert assistant.delay_until_next_message == 100.0
        for user in session.messages[0::2]:
            assert user.source == "user"
            assert user.num_tokens == 80
