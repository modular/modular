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

"""Tests for MAXModelConfig subfolder support."""

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from max.graph.weights import WeightsFormat
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.hf_utils import HuggingFaceRepo
from test_common.mocks import (
    mock_pipeline_config_resolve,
)


class TestMAXModelConfigSubfolder:
    """Test suite for MAXModelConfig subfolder field."""

    @mock_pipeline_config_resolve
    def test_subfolder_default_is_none(self) -> None:
        """Test that subfolder defaults to None."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))
        assert config.model.subfolder is None

    @mock_pipeline_config_resolve
    def test_subfolder_can_be_set(self) -> None:
        """Test that subfolder can be set to a string value."""
        config = PipelineConfig(
            model=MAXModelConfig(model_path="test/model", subfolder="vae")
        )
        assert config.model.subfolder == "vae"

    @mock_pipeline_config_resolve
    def test_subfolder_passed_to_huggingface_weight_repo(self) -> None:
        """Test that subfolder is propagated to the HuggingFaceRepo for weights."""
        model_config = MAXModelConfig(
            model_path="/tmp/fake-local-model",
            subfolder="text_encoder",
        )
        # Patch model_path to be a local path so HuggingFaceRepo doesn't
        # attempt network calls.
        with patch("os.path.exists", return_value=True):
            repo = model_config.huggingface_weight_repo
            assert repo.subfolder == "text_encoder"

    @mock_pipeline_config_resolve
    def test_subfolder_none_does_not_set_on_repo(self) -> None:
        """Test that subfolder=None results in None on HuggingFaceRepo."""
        model_config = MAXModelConfig(model_path="/tmp/fake-local-model")
        with patch("os.path.exists", return_value=True):
            repo = model_config.huggingface_weight_repo
            assert repo.subfolder is None

    @mock_pipeline_config_resolve
    def test_subfolder_passed_to_huggingface_config_loading(self) -> None:
        """Test that subfolder is on the repo passed to AutoConfig loading."""
        mock_auto_config = Mock()
        model_config = MAXModelConfig(
            model_path="test/model",
            subfolder="vae",
        )
        with (
            patch("os.path.exists", return_value=True),
            patch.object(
                PIPELINE_REGISTRY,
                "get_active_huggingface_config",
                return_value=mock_auto_config,
            ) as mock_get_config,
            patch(
                "max.pipelines.lib.hf_utils.validate_hf_repo_access",
            ),
        ):
            _ = model_config.huggingface_config
            mock_get_config.assert_called_once()
            repo_arg = mock_get_config.call_args.kwargs["huggingface_repo"]
            assert repo_arg.subfolder == "vae"

    @mock_pipeline_config_resolve
    def test_subfolder_none_passed_to_huggingface_config_loading(self) -> None:
        """Test that subfolder=None is on the repo passed to AutoConfig loading."""
        mock_auto_config = Mock()
        model_config = MAXModelConfig(model_path="test/model")
        with (
            patch("os.path.exists", return_value=True),
            patch.object(
                PIPELINE_REGISTRY,
                "get_active_huggingface_config",
                return_value=mock_auto_config,
            ) as mock_get_config,
            patch(
                "max.pipelines.lib.hf_utils.validate_hf_repo_access",
            ),
        ):
            _ = model_config.huggingface_config
            mock_get_config.assert_called_once()
            repo_arg = mock_get_config.call_args.kwargs["huggingface_repo"]
            assert repo_arg.subfolder is None


class TestHuggingFaceRepoSubfolderWeightDiscovery:
    """Test that HuggingFaceRepo.weight_files respects subfolder scoping."""

    def test_weight_files_scoped_to_subfolder(self) -> None:
        """Test that only weights inside the subfolder are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a weight at the repo root (should be excluded).
            root_weight = os.path.join(tmpdir, "model.safetensors")
            open(root_weight, "w").close()

            # Create a weight inside a subfolder (should be included).
            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(
                vae_dir, "diffusion_pytorch_model.safetensors"
            )
            open(subfolder_weight, "w").close()

            repo = HuggingFaceRepo(repo_id=tmpdir, subfolder="vae")
            wf = repo.weight_files

            assert WeightsFormat.safetensors in wf
            paths = wf[WeightsFormat.safetensors]
            assert paths == ["vae/diffusion_pytorch_model.safetensors"]

    def test_weight_files_without_subfolder_returns_all(self) -> None:
        """Test that all weights are returned when no subfolder is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_weight = os.path.join(tmpdir, "model.safetensors")
            open(root_weight, "w").close()

            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(
                vae_dir, "diffusion_pytorch_model.safetensors"
            )
            open(subfolder_weight, "w").close()

            repo = HuggingFaceRepo(repo_id=tmpdir)
            wf = repo.weight_files

            assert WeightsFormat.safetensors in wf
            paths = sorted(wf[WeightsFormat.safetensors])
            assert paths == sorted(
                [
                    "model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                ]
            )

    def test_supported_encodings_scoped_to_subfolder_local(self) -> None:
        """Test that supported_encodings reads from subfolder-scoped files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a BF16 safetensors file at root.
            root_weight = os.path.join(tmpdir, "model.safetensors")
            _write_fake_safetensors(root_weight, dtype="BF16")

            # Create an F32 safetensors file inside subfolder.
            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(vae_dir, "model.safetensors")
            _write_fake_safetensors(subfolder_weight, dtype="F32")

            # Without subfolder: reads first file found (local repos assume
            # one encoding per repo).
            repo_all = HuggingFaceRepo(repo_id=tmpdir)
            assert len(repo_all.supported_encodings) >= 1

            # With subfolder="vae": should only see F32 from the subfolder.
            repo_vae = HuggingFaceRepo(repo_id=tmpdir, subfolder="vae")
            assert repo_vae.supported_encodings == ["float32"]


class TestMAXModelConfigSubfolderWeightPathPrefixing:
    """Test that resolve() prepends subfolder to user-provided weight paths."""

    @mock_pipeline_config_resolve
    def test_subfolder_prepended_to_weight_path(self) -> None:
        """Test that user-provided weight_path gets subfolder prefix."""
        config = MAXModelConfig(
            model_path="org/model",
            subfolder="vae",
            weight_path=[Path("model.safetensors")],
        )
        # Call resolve with mocked dependencies.
        with (
            patch(
                "max.pipelines.lib.config.model_config.WeightPathParser.parse",
                return_value=([Path("model.safetensors")], None),
            ),
            patch(
                "max.pipelines.lib.config.model_config.devices_exist",
                return_value=True,
            ),
            patch(
                "max.pipelines.lib.hf_utils.validate_hf_repo_access",
            ),
        ):
            config.resolve()
            assert config.weight_path == [Path("vae/model.safetensors")]

    @mock_pipeline_config_resolve
    def test_subfolder_not_double_prepended(self) -> None:
        """Test that paths already containing subfolder are not double-prefixed."""
        config = MAXModelConfig(
            model_path="org/model",
            subfolder="vae",
            weight_path=[Path("vae/model.safetensors")],
        )
        with (
            patch(
                "max.pipelines.lib.config.model_config.WeightPathParser.parse",
                return_value=([Path("vae/model.safetensors")], None),
            ),
            patch(
                "max.pipelines.lib.config.model_config.devices_exist",
                return_value=True,
            ),
            patch(
                "max.pipelines.lib.hf_utils.validate_hf_repo_access",
            ),
        ):
            config.resolve()
            assert config.weight_path == [Path("vae/model.safetensors")]

    @mock_pipeline_config_resolve
    def test_subfolder_skips_absolute_paths(self) -> None:
        """Test that absolute paths are not prefixed with subfolder."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tmp:
            abs_path = Path(tmp.name)
            config = MAXModelConfig(
                model_path="org/model",
                subfolder="vae",
                weight_path=[abs_path],
            )
            with (
                patch(
                    "max.pipelines.lib.config.model_config.WeightPathParser.parse",
                    return_value=([abs_path], None),
                ),
                patch(
                    "max.pipelines.lib.config.model_config.devices_exist",
                    return_value=True,
                ),
            ):
                config.resolve()
                assert config.weight_path == [abs_path]

    @mock_pipeline_config_resolve
    def test_no_subfolder_leaves_weight_path_unchanged(self) -> None:
        """Test that without subfolder, weight_path is not modified."""
        config = MAXModelConfig(
            model_path="org/model",
            weight_path=[Path("model.safetensors")],
        )
        with (
            patch(
                "max.pipelines.lib.config.model_config.WeightPathParser.parse",
                return_value=([Path("model.safetensors")], None),
            ),
            patch(
                "max.pipelines.lib.config.model_config.devices_exist",
                return_value=True,
            ),
            patch(
                "max.pipelines.lib.hf_utils.validate_hf_repo_access",
            ),
        ):
            config.resolve()
            assert config.weight_path == [Path("model.safetensors")]


def _write_fake_safetensors(path: str, dtype: str = "BF16") -> None:
    """Write a minimal safetensors file with a single tensor of the given dtype."""
    import json

    header = {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00\x00")  # 2 bytes of fake tensor data
