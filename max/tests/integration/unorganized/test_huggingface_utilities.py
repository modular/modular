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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import errors as hf_hub_errors
from max.graph.weights import WeightsFormat
from max.pipelines.lib import HuggingFaceRepo
from max.pipelines.lib.hf_utils import validate_hf_repo_access


def test_huggingface_repo__local_path() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_path = MagicMock()
        mock_path.is_dir.return_value = True
        mock_path.glob.return_value = [Path(temp_dir) / "model.safetensors"]
        with patch("pathlib.Path", return_value=mock_path):
            # Test with local path
            hf_repo = HuggingFaceRepo(repo_id=temp_dir)

            # Verify it's treated as a local repo
            assert hf_repo.repo_type == "local"


def test_huggingface_repo__file_exists(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert len(files[WeightsFormat.safetensors]) == 4
    assert sorted(files[WeightsFormat.safetensors]) == [
        Path("model-00001-of-00004.safetensors"),
        Path("model-00002-of-00004.safetensors"),
        Path("model-00003-of-00004.safetensors"),
        Path("model-00004-of-00004.safetensors"),
    ]


def test_huggingface_repo__formats_available(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
) -> None:
    # Test a GGUF repo
    hf_repo = HuggingFaceRepo(
        repo_id=llama_3_1_8b_instruct_local_path,
    )

    assert WeightsFormat.safetensors in hf_repo.formats_available
    assert WeightsFormat.gguf not in hf_repo.formats_available

    # Test a Safetensors repo
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)

    assert WeightsFormat.safetensors in hf_repo.formats_available
    assert WeightsFormat.gguf not in hf_repo.formats_available


def test_huggingface_repo__encodings_supported(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
) -> None:
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    assert "bfloat16" in hf_repo.supported_encodings

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    assert "q4_k" not in hf_repo.supported_encodings
    assert "bfloat16" in hf_repo.supported_encodings


def test_huggingface_repo__get_files_for_encoding(
    tiny_llama_1_1b_chat_v1_0_local_path: str,
    qwen_32b_preview_local_path: str,
    mistral_nemo_instruct_2407_local_path: str,
) -> None:
    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 1
    assert files[WeightsFormat.safetensors][0] == Path("model.safetensors")

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=qwen_32b_preview_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 17
    assert sorted(files[WeightsFormat.safetensors]) == [
        Path("model-00001-of-00017.safetensors"),
        Path("model-00002-of-00017.safetensors"),
        Path("model-00003-of-00017.safetensors"),
        Path("model-00004-of-00017.safetensors"),
        Path("model-00005-of-00017.safetensors"),
        Path("model-00006-of-00017.safetensors"),
        Path("model-00007-of-00017.safetensors"),
        Path("model-00008-of-00017.safetensors"),
        Path("model-00009-of-00017.safetensors"),
        Path("model-00010-of-00017.safetensors"),
        Path("model-00011-of-00017.safetensors"),
        Path("model-00012-of-00017.safetensors"),
        Path("model-00013-of-00017.safetensors"),
        Path("model-00014-of-00017.safetensors"),
        Path("model-00015-of-00017.safetensors"),
        Path("model-00016-of-00017.safetensors"),
        Path("model-00017-of-00017.safetensors"),
    ]

    # Test a Safetensors repo, with both shared files and consolidated safetensors
    hf_repo = HuggingFaceRepo(repo_id=mistral_nemo_instruct_2407_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert len(files[WeightsFormat.safetensors]) == 5
    assert Path("consolidated.safetensors") not in files

    # Test a Safetensors repo, with the wrong encoding requested.
    hf_repo = HuggingFaceRepo(repo_id=qwen_32b_preview_local_path)
    files = hf_repo.files_for_encoding("float32")
    assert len(files) == 0


def test_huggingface_repo__encoding_for_file(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
    qwen_32b_preview_local_path: str,
) -> None:
    # This repo, has one safetensors file, and its a bf16 file.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    model_encoding = hf_repo.encoding_for_file("model.safetensors")
    assert model_encoding == "bfloat16"

    # This repo, has many safetensors file, and they are bf16.
    hf_repo = HuggingFaceRepo(repo_id=qwen_32b_preview_local_path)
    model_encoding = hf_repo.encoding_for_file(
        "model-00014-of-00017.safetensors"
    )
    assert model_encoding == "bfloat16"

    # This repo, has a few GGUF files, and they are a variety of encodings.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    model_encoding = hf_repo.encoding_for_file("llama-3.1-8b-instruct-f32.gguf")
    assert model_encoding == "float32"

    model_encoding = hf_repo.encoding_for_file(
        "llama-3.1-8b-instruct-q4_k_m.gguf"
    )
    assert model_encoding == "q4_k"


class TestValidateHfRepoAccess:
    """Test cases for validate_hf_repo_access function."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test to ensure test isolation."""
        validate_hf_repo_access.cache_clear()

    def test_valid_repo_access_success(self) -> None:
        """Test that valid repository access doesn't raise any exception."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = True

            # Should not raise any exception
            validate_hf_repo_access("valid/repo", "main")

            mock_exists.assert_called_once_with(
                repo_id="valid/repo", revision="main"
            )

    def test_repo_not_exists_raises_value_error(self) -> None:
        """Test that non-existent repository raises ValueError with appropriate message."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("nonexistent/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'nonexistent/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'main' exists" in error_msg

    def test_gated_repo_error_raises_value_error_with_auth_message(
        self,
    ) -> None:
        """Test that GatedRepoError raises ValueError with authentication guidance."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.GatedRepoError("Repository is gated")
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("gated/repo", "main")

            error_msg = str(exc_info.value)
            assert (
                "Repository 'gated/repo' exists but requires authentication"
                in error_msg
            )
            assert (
                "This is a gated/private repository that requires an access token"
                in error_msg
            )
            assert (
                "1. A valid Hugging Face access token with appropriate permissions"
                in error_msg
            )
            assert "2. The token is properly configured" in error_msg
            assert "3. You have been granted access to this model" in error_msg
            assert "Original error: Repository is gated" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_repository_not_found_error_raises_value_error(self) -> None:
        """Test that RepositoryNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.RepositoryNotFoundError(
                "Repository not found"
            )
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("missing/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'missing/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'main' exists" in error_msg
            assert "Original error: Repository not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_revision_not_found_error_raises_value_error(self) -> None:
        """Test that RevisionNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.RevisionNotFoundError(
                "Revision not found"
            )
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("valid/repo", "nonexistent-branch")

            error_msg = str(exc_info.value)
            assert "Repository 'valid/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'nonexistent-branch' exists" in error_msg
            assert "Original error: Revision not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_generic_exception_raises_value_error_with_fallback_message(
        self,
    ) -> None:
        """Test that unexpected exceptions raise ValueError with fallback message."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = ConnectionError("Network timeout")
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("some/repo", "main")

            error_msg = str(exc_info.value)
            assert "Failed to access repository 'some/repo'" in error_msg
            assert (
                "This could be due to network issues, invalid repository, or authentication problems"
                in error_msg
            )
            assert "Original error: Network timeout" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_entry_not_found_error_raises_value_error(self) -> None:
        """Test that EntryNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.EntryNotFoundError("Entry not found")
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("valid/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'valid/repo' not found" in error_msg
            assert "Original error: Entry not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_function_calls_repo_exists_with_correct_parameters(self) -> None:
        """Test that the function calls _repo_exists_with_retry with correct parameters."""
        with patch(
            "max.pipelines.lib.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = True

            validate_hf_repo_access("test/repo", "v1.0")

            mock_exists.assert_called_once_with(
                repo_id="test/repo", revision="v1.0"
            )

    def test_error_message_contains_repo_and_revision_info(self) -> None:
        """Test that error messages contain the specific repo_id and revision being validated."""
        test_cases = [
            ("my-org/my-model", "main"),
            ("another/repo", "v2.1"),
            ("user/special-chars", "feature/branch-name"),
        ]

        for repo_id, revision in test_cases:
            with patch(
                "max.pipelines.lib.hf_utils._repo_exists_with_retry"
            ) as mock_exists:
                mock_exists.return_value = False

                with pytest.raises(ValueError) as exc_info:
                    validate_hf_repo_access(repo_id, revision)

                error_msg = str(exc_info.value)
                assert f"Repository '{repo_id}' not found" in error_msg
                assert f"revision '{revision}' exists" in error_msg
