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

"""Tests for ModelManifest."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from max.pipelines.lib.config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest

# All unit tests patch _load_model_index and validate_hf_repo_access to
# avoid network calls.  We also force HF_HUB_OFFLINE=False so that
# HuggingFaceRepo.__post_init__ takes the "online" path (where
# validate_hf_repo_access is mocked) instead of trying to resolve from a
# non-existent local cache.
LOAD_INDEX_TARGET = (
    "max.pipelines.lib.model_manifest.ModelManifest._load_model_index"
)
VALIDATE_HF_ACCESS_TARGET = "max.pipelines.lib.hf_utils.validate_hf_repo_access"
HF_OFFLINE_TARGET = "huggingface_hub.constants.HF_HUB_OFFLINE"


def _make_config(
    model_path: str = "test/model",
    weight_path: list[Path] | None = None,
    quantization_encoding: str | None = None,
) -> MAXModelConfig:
    """Create a MAXModelConfig without validation or network access."""
    kwargs: dict[str, Any] = {"model_path": model_path, "device_specs": []}
    if weight_path is not None:
        kwargs["weight_path"] = weight_path
    if quantization_encoding is not None:
        kwargs["quantization_encoding"] = quantization_encoding
    return MAXModelConfig.model_construct(**kwargs)


@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestFromModelPath:
    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_primary(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert registry["primary"].model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_contains_primary(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert "primary" in registry

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_does_not_contain_other(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert "draft" not in registry

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_items(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        items = list(registry.items())
        assert len(items) == 1
        role, cfg = items[0]
        assert role == "primary"
        assert cfg.model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_len(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert len(registry) == 1


class TestFromComponents:
    def test_get_by_role(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest.from_components({"vae": vae, "unet": unet})
        assert registry["vae"] is vae
        assert registry["unet"] is unet

    def test_contains(self) -> None:
        registry = ModelManifest.from_components(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert "vae" in registry
        assert "unet" in registry
        assert "primary" not in registry

    def test_items(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest.from_components({"vae": vae, "unet": unet})
        items = dict(registry.items())
        assert items == {"vae": vae, "unet": unet}

    def test_len(self) -> None:
        registry = ModelManifest.from_components(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert len(registry) == 2

    def test_does_not_mutate_input(self) -> None:
        components: dict[str, MAXModelConfig] = {"vae": _make_config()}
        registry = ModelManifest.from_components(components)
        components["extra"] = _make_config()
        assert "extra" not in registry


class TestSpeculativeDecoding:
    """Test a spec-decoding scenario with primary + draft models."""

    def test_primary_and_draft(self) -> None:
        primary = _make_config("primary-model")
        draft = _make_config("draft-model")
        registry = ModelManifest(
            models={"primary": primary, "draft": draft},
        )
        assert registry["primary"] is primary
        assert registry["draft"] is draft
        assert len(registry) == 2


@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestErrorMessages:
    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_missing_role(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        with pytest.raises(KeyError, match="draft"):
            registry["draft"]

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_missing_role_lists_available(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        with pytest.raises(KeyError, match="primary"):
            registry["draft"]


@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestDiffusersAutoExpansion:
    """Tests for from_model_path auto-expanding diffusers repos."""

    @staticmethod
    def _fake_model_index() -> dict[str, object]:
        return {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }

    def test_expands_diffusers_model(self, _mock_validate: Any) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        assert "transformer" in registry
        assert "vae" in registry
        assert "text_encoder" in registry
        assert "scheduler" in registry

    def test_each_component_inherits_model_path(
        self, _mock_validate: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        for _role, component_cfg in registry.items():
            assert component_cfg.model_path == "org/diffusion-model"

    def test_each_component_has_subfolder(self, _mock_validate: Any) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        for role, component_cfg in registry.items():
            assert component_cfg.subfolder == role

    def test_non_diffusers_stays_primary(self, _mock_validate: Any) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=None):
            registry = ModelManifest.from_model_path(
                "org/llm-model", device_specs=[]
            )

        assert registry["primary"].model_path == "org/llm-model"
        assert len(registry) == 1

    def test_rejects_kwargs_for_composite_model(
        self, _mock_validate: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            with pytest.raises(ValueError, match="from_components"):
                ModelManifest.from_model_path(
                    "org/diffusion-model",
                    quantization_encoding="float32",
                )

    def test_skips_private_keys(self, _mock_validate: Any) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path("org/model")

        assert "transformer" in registry
        assert "_class_name" not in registry
        assert "_diffusers_version" not in registry


@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestRevisionPropagation:
    """Verify that the revision parameter propagates to MAXModelConfig."""

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_revision_propagates_to_primary(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path(
            "test-model", revision="abc123", device_specs=[]
        )
        assert registry["primary"].huggingface_model_revision == "abc123"

    def test_revision_propagates_to_components(
        self, _mock_validate: Any
    ) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path(
                "org/diffusion-model", revision="def456"
            )

        for _role, cfg in registry.items():
            assert cfg.huggingface_model_revision == "def456"
