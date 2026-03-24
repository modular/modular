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


class TestWithOverride:
    """Tests for with_override."""

    @staticmethod
    def _flux2_manifest() -> ModelManifest:
        base = "black-forest-labs/FLUX.2-dev"
        return ModelManifest.from_components(
            {
                "transformer": _make_config(
                    base, quantization_encoding="bfloat16"
                ),
                "vae": _make_config(base, quantization_encoding="bfloat16"),
                "text_encoder": _make_config(
                    base, quantization_encoding="bfloat16"
                ),
            }
        )

    def test_partial_field_override(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4-repo/weights.safetensors")
        ]
        # Other components unchanged.
        assert updated["vae"].weight_path == manifest["vae"].weight_path
        assert (
            updated["text_encoder"].weight_path
            == manifest["text_encoder"].weight_path
        )

    def test_multiple_field_overrides(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
            huggingface_weight_revision="abc123",
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4-repo/weights.safetensors")
        ]
        assert updated["transformer"].quantization_encoding == "float4_e2m1fnx2"
        assert updated["transformer"].huggingface_weight_revision == "abc123"

    def test_encoding_preserved_when_not_overridden(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/other-repo/weights.safetensors")],
        )
        assert updated["transformer"].quantization_encoding == "bfloat16"

    def test_original_not_mutated(self) -> None:
        manifest = self._flux2_manifest()
        original_cfg = manifest["transformer"]
        _updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
        )
        # Original manifest's config is unchanged.
        assert manifest["transformer"] is original_cfg
        assert manifest["transformer"].quantization_encoding == "bfloat16"

    def test_getitem_works_after_override(self) -> None:
        cfg = _make_config("test/model", weight_path=[Path("old.safetensors")])
        manifest = ModelManifest(models={"primary": cfg})
        updated = manifest.with_override(
            "primary", weight_path=[Path("new.safetensors")]
        )
        assert updated["primary"].weight_path == [Path("new.safetensors")]

    def test_partial_update_missing_role_raises(self) -> None:
        manifest = self._flux2_manifest()
        with pytest.raises(ValueError, match="unet"):
            manifest.with_override(
                "unet", weight_path=[Path("some/path.safetensors")]
            )

    def test_no_config_or_overrides_raises(self) -> None:
        manifest = self._flux2_manifest()
        with pytest.raises(ValueError, match="requires either"):
            manifest.with_override("transformer")

    def test_chained_overrides(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4/transformer.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
        ).with_override(
            "vae",
            weight_path=[Path("org/custom-vae/vae.safetensors")],
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4/transformer.safetensors")
        ]
        assert updated["transformer"].quantization_encoding == "float4_e2m1fnx2"
        assert updated["vae"].weight_path == [
            Path("org/custom-vae/vae.safetensors")
        ]
        assert updated["vae"].quantization_encoding == "bfloat16"

    def test_full_config_replacement(self) -> None:
        manifest = self._flux2_manifest()
        new_cfg = _make_config(
            "org/new-transformer", quantization_encoding="float32"
        )
        updated = manifest.with_override("transformer", config=new_cfg)
        assert updated["transformer"] is new_cfg
        assert updated["transformer"].model_path == "org/new-transformer"

    def test_add_new_component_with_config(self) -> None:
        manifest = self._flux2_manifest()
        draft = _make_config("org/draft-model")
        updated = manifest.with_override("draft", config=draft)
        assert updated["draft"] is draft
        assert len(updated) == 4

    def test_config_with_field_overrides(self) -> None:
        manifest = self._flux2_manifest()
        base_cfg = _make_config("org/draft-model")
        updated = manifest.with_override(
            "draft", config=base_cfg, quantization_encoding="q4_0"
        )
        assert updated["draft"].model_path == "org/draft-model"
        assert updated["draft"].quantization_encoding == "q4_0"

    def test_spec_decoding_override_draft(self) -> None:
        primary = _make_config("org/primary-model")
        draft = _make_config("org/draft-model")
        manifest = ModelManifest(
            models={"primary": primary, "draft": draft},
        )
        updated = manifest.with_override(
            "draft",
            weight_path=[Path("org/draft-quantized/weights.gguf")],
            quantization_encoding="q4_0",
        )
        assert updated["draft"].weight_path == [
            Path("org/draft-quantized/weights.gguf")
        ]
        assert updated["draft"].quantization_encoding == "q4_0"
        assert updated["primary"] is primary  # primary unchanged


class TestSerialization:
    """Tests for ModelManifest serialization via msgpack.

    When ``ModelManifest`` is embedded as a field on a Pydantic model
    (e.g. ``PipelineConfig``), the serving layer serialises it with
    ``msgspec.msgpack``.  These tests verify that a ``ModelManifest``
    round-trips through the same encode → decode path used in production.
    """

    def test_single_model_msgpack_round_trip(self) -> None:
        """A single-model manifest round-trips through msgpack."""
        cfg = _make_config("org/llm-model", quantization_encoding="bfloat16")
        manifest = ModelManifest(models={"primary": cfg})

        restored = _msgpack_round_trip(manifest)

        assert list(restored.models.keys()) == ["primary"]
        assert restored["primary"].model_path == "org/llm-model"
        assert restored["primary"].quantization_encoding == "bfloat16"

    def test_multi_component_msgpack_round_trip(self) -> None:
        """A multi-component manifest round-trips."""
        manifest = ModelManifest.from_components(
            {
                "vae": _make_config(
                    "org/model", quantization_encoding="bfloat16"
                ),
                "unet": _make_config(
                    "org/model", quantization_encoding="float32"
                ),
            }
        )

        restored = _msgpack_round_trip(manifest)

        assert set(restored.models.keys()) == {"vae", "unet"}
        assert restored["vae"].quantization_encoding == "bfloat16"
        assert restored["unet"].quantization_encoding == "float32"

    def test_speculative_decoding_msgpack_round_trip(self) -> None:
        """A primary + draft manifest round-trips through msgpack."""
        manifest = ModelManifest(
            models={
                "primary": _make_config("org/target"),
                "draft": _make_config("org/draft"),
            },
        )

        restored = _msgpack_round_trip(manifest)

        assert restored["primary"].model_path == "org/target"
        assert restored["draft"].model_path == "org/draft"

    def test_weight_path_survives_msgpack_round_trip(self) -> None:
        """Weight paths (list[Path]) survive msgpack serialization."""
        cfg = _make_config("org/model")
        cfg = cfg.model_copy(
            update={
                "weight_path": [
                    Path("shard-0.safetensors"),
                    Path("shard-1.safetensors"),
                ]
            }
        )
        manifest = ModelManifest(models={"primary": cfg})

        restored = _msgpack_round_trip(manifest)

        primary = restored["primary"]
        assert len(primary.weight_path) == 2
        # After msgpack round-trip, Path objects are serialized as strings.
        # Coerce back to Path for comparison — this mirrors what Pydantic's
        # model_validate (as opposed to model_construct) would do.
        assert Path(primary.weight_path[0]) == Path("shard-0.safetensors")
        assert Path(primary.weight_path[1]) == Path("shard-1.safetensors")

    def test_empty_manifest_msgpack_round_trip(self) -> None:
        """An empty manifest round-trips through msgpack."""
        manifest = ModelManifest(models={})

        restored = _msgpack_round_trip(manifest)

        assert len(restored) == 0

    def test_subfolder_survives_msgpack_round_trip(self) -> None:
        """Subfolder field survives msgpack serialization."""
        cfg = _make_config("org/diffusion-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})
        manifest = ModelManifest.from_components({"transformer": cfg})

        restored = _msgpack_round_trip(manifest)

        assert restored["transformer"].subfolder == "transformer"


# -- Computed fields on MAXModelConfig that trigger network access and must
# -- be excluded when dumping in a sandboxed test environment.
_COMPUTED_FIELDS = {
    "huggingface_weight_repo_id",
    "huggingface_weight_repo",
    "huggingface_model_repo",
    "huggingface_config",
    "diffusers_config",
    "model_name",
    "graph_quantization_encoding",
    "generation_config",
    "sampling_params_defaults",
}


def _msgpack_round_trip(manifest: ModelManifest) -> ModelManifest:
    """Serialize a ``ModelManifest`` through msgpack and back.

    Mirrors the production path: each ``MAXModelConfig`` is dumped via
    Pydantic's ``model_dump(mode="json")`` (matching the ``enc_hook``
    used by the serving layer's ``MsgpackNumpyEncoder``), packed with
    ``msgspec.msgpack``, then unpacked and reconstructed.
    """
    import msgspec

    payload = {
        "models": {
            role: cfg.model_dump(mode="json", exclude=_COMPUTED_FIELDS)
            for role, cfg in manifest.models.items()
        },
    }
    packed = msgspec.msgpack.encode(payload)
    unpacked = msgspec.msgpack.decode(packed)

    models = {
        role: MAXModelConfig.model_construct(**cfg_data)
        for role, cfg_data in unpacked["models"].items()
    }
    return ModelManifest(models=models)
