# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from max.pipelines.lib import HuggingFaceRepo


def test_huggingface_repo__local_path():
    temp_dir = tempfile.mkdtemp()
    mock_path = MagicMock()
    mock_path.is_dir.return_value = True
    mock_path.glob.return_value = [Path(temp_dir) / "model.safetensors"]
    with patch("pathlib.Path", return_value=mock_path):
        # Test with local path
        hf_repo = HuggingFaceRepo(repo_id=temp_dir)

        # Verify it's treated as a local repo
        assert hf_repo.repo_type == "local"
