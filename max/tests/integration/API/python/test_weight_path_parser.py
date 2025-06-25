# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from unittest.mock import patch

import pytest
from max.pipelines.lib import WeightPathParser


class TestWeightPathParser:
    """Unit tests for WeightPathParser class."""

    def test_parse_single_string_path(self) -> None:
        """Test parsing a single string path."""
        model_path = "org1/model1"
        weight_path = "weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert isinstance(result_paths[0], Path)
        assert str(result_paths[0]) == "weights.safetensors"
        assert repo_id is None

    def test_parse_single_path_object(self) -> None:
        """Test parsing a single Path object."""
        model_path = "org1/model1"
        weight_path = Path("weights.safetensors")

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert isinstance(result_paths[0], Path)
        assert result_paths[0] == Path("weights.safetensors")
        assert repo_id is None

    def test_parse_list_of_paths(self) -> None:
        """Test parsing a list of paths."""
        model_path = "org1/model1"
        weight_path = [
            Path("weight1.safetensors"),
            Path("weight2.safetensors"),
        ]

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 2
        assert all(isinstance(p, Path) for p in result_paths)
        assert str(result_paths[0]) == "weight1.safetensors"
        assert str(result_paths[1]) == "weight2.safetensors"
        assert repo_id is None

    def test_parse_tuple_of_paths(self) -> None:
        """Test parsing a tuple of paths (should convert to list)."""
        model_path = "org1/model1"
        weight_path = ("weight1.safetensors", "weight2.safetensors")

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 2
        assert all(isinstance(p, Path) for p in result_paths)
        assert str(result_paths[0]) == "weight1.safetensors"
        assert str(result_paths[1]) == "weight2.safetensors"
        assert repo_id is None

    @patch("pathlib.Path.is_file")
    def test_parse_existing_file_skips_hf_parsing(self, mock_is_file) -> None:
        """Test that existing files skip HF repo parsing."""
        mock_is_file.return_value = True

        model_path = "org1/model1"
        weight_path = "existing_weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert isinstance(result_paths[0], Path)
        assert repo_id is None

    @patch("huggingface_hub.file_exists")
    def test_parse_hf_repo_path_same_as_model_path(
        self, mock_file_exists
    ) -> None:
        """Test parsing HF repo path that matches model_path."""
        mock_file_exists.return_value = True

        model_path = "organization/model-name"
        weight_path = "organization/model-name/weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert str(result_paths[0]) == "weights.safetensors"
        assert repo_id is None

    @patch("huggingface_hub.file_exists")
    def test_parse_hf_repo_path_different_from_model_path(
        self, mock_file_exists
    ) -> None:
        """Test parsing HF repo path that differs from model_path."""
        mock_file_exists.return_value = True

        model_path = "org1/model1"
        weight_path = "org2/model2/weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert str(result_paths[0]) == "weights.safetensors"
        assert repo_id == "org2/model2"

    @patch("huggingface_hub.file_exists")
    def test_parse_hf_repo_path_file_not_exists(self, mock_file_exists) -> None:
        """Test parsing HF repo path when file doesn't exist."""
        mock_file_exists.return_value = False

        model_path = "org1/model1"
        weight_path = "org2/model2/weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert str(result_paths[0]) == "org2/model2/weights.safetensors"
        assert repo_id is None

    def test_parse_insufficient_path_components_empty_model_path_raises_error(
        self,
    ) -> None:
        """Test that insufficient path components with empty model_path raises error."""
        model_path = ""
        weight_path = "just/filename.safetensors"  # Only 2 components

        with pytest.raises(
            ValueError,
            match="Unable to derive model_path from weight_path, "
            "please provide a valid Hugging Face repository id.",
        ):
            WeightPathParser.parse(model_path, weight_path)

    def test_parse_insufficient_path_components_with_model_path(self) -> None:
        """Test that insufficient path components with valid model_path works."""
        model_path = "valid/model"
        weight_path = "just/filename.safetensors"  # Only 2 components

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert str(result_paths[0]) == "just/filename.safetensors"
        assert repo_id is None

    @patch("huggingface_hub.file_exists")
    def test_parse_complex_hf_path_with_nested_structure(
        self, mock_file_exists
    ) -> None:
        """Test parsing complex HF paths with nested directory structure."""
        mock_file_exists.return_value = True

        model_path = "org1/model1"
        weight_path = "org2/model2/subfolder/nested/weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert str(result_paths[0]) == "subfolder/nested/weights.safetensors"
        assert repo_id == "org2/model2"

    @patch("huggingface_hub.file_exists")
    def test_parse_complex_hf_path_with_nested_structure_no_hf_prefix_no_model_path(
        self, mock_file_exists
    ) -> None:
        """Test parsing complex HF paths with very nested directory structure.
        In this case the weight path doesn't have a HF prefix and model_path is empty,
        which means file_exists will return False so treat the whole path (without
        trimming a potential repo id prefix) as a local path. But since model_path
        is empty, it should raise an error."""
        mock_file_exists.return_value = False

        model_path = ""
        weight_path = "very/nested/subfolder/another_nested/weights.safetensors"

        with pytest.raises(
            ValueError,
            match="Unable to derive model_path from weight_path, "
            "please provide a valid Hugging Face repository id.",
        ):
            WeightPathParser.parse(model_path, weight_path)

    @patch("huggingface_hub.file_exists")
    def test_parse_complex_hf_path_with_nested_structure_no_hf_prefix(
        self, mock_file_exists
    ) -> None:
        """Test parsing complex HF paths with very nested directory structure.
        In this case the weight path doesn't have a HF prefix, which means file_exists
        will return False so treat the whole path (without trimming a potential
        repo id prefix) as a local path."""
        mock_file_exists.return_value = False

        model_path = "org1/model1"
        weight_path = "very/nested/subfolder/another_nested/weights.safetensors"

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 1
        assert (
            str(result_paths[0])
            == "very/nested/subfolder/another_nested/weights.safetensors"
        )
        assert repo_id is None

    @patch("pathlib.Path.is_file")
    @patch("huggingface_hub.file_exists")
    def test_parse_mixed_existing_and_hf_paths(
        self, mock_file_exists, mock_is_file
    ) -> None:
        """Test parsing a mix of existing files and HF repo paths."""
        # First path exists locally, second is from HF
        mock_is_file.side_effect = lambda: str(self).endswith(
            "local_weights.safetensors"
        )
        mock_file_exists.return_value = True

        model_path = "org1/model1"
        weight_path = [
            "local_weights.safetensors",
            "org2/model2/remote_weights.safetensors",
        ]

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 2
        assert str(result_paths[0]) == "local_weights.safetensors"
        assert str(result_paths[1]) == "remote_weights.safetensors"
        assert repo_id == "org2/model2"

    def test_parse_empty_weight_path_list(self) -> None:
        """Test parsing empty weight path list."""
        model_path = "test/model"
        weight_path: list[Path] = []

        result_paths, repo_id = WeightPathParser.parse(model_path, weight_path)

        assert len(result_paths) == 0
        assert repo_id is None

    def test_parse_hf_repo_path_parsing_private_method_direct(self) -> None:
        """Test the private _parse_huggingface_repo_path method directly."""
        model_path = "org1/model1"
        path = Path("org2/model2/weights.safetensors")

        with patch("huggingface_hub.file_exists", return_value=True):
            result_path, repo_id = (
                WeightPathParser._parse_huggingface_repo_path(model_path, path)
            )

            assert str(result_path) == "weights.safetensors"
            assert repo_id == "org2/model2"

    def test_parse_hf_repo_path_parsing_no_repo_match(self) -> None:
        """Test HF repo path parsing when no repo matches."""
        model_path = "org1/model1"
        path = Path("org2/model2/weights.safetensors")

        with patch("huggingface_hub.file_exists", return_value=False):
            result_path, repo_id = (
                WeightPathParser._parse_huggingface_repo_path(model_path, path)
            )

            assert result_path == path  # Should return original path
            assert repo_id is None
