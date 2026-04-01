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
    def test_get_main(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert registry["main"].model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_contains_main(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert "main" in registry

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
        assert role == "main"
        assert cfg.model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_len(self, _mock_load: Any, _mock_validate: Any) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert len(registry) == 1


class TestDirectConstruction:
    def test_get_by_role(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest({"vae": vae, "unet": unet})
        assert registry["vae"] is vae
        assert registry["unet"] is unet

    def test_contains(self) -> None:
        registry = ModelManifest(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert "vae" in registry
        assert "unet" in registry
        assert "main" not in registry

    def test_items(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest({"vae": vae, "unet": unet})
        items = dict(registry.items())
        assert items == {"vae": vae, "unet": unet}

    def test_len(self) -> None:
        registry = ModelManifest(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert len(registry) == 2

    def test_does_not_mutate_input(self) -> None:
        components: dict[str, MAXModelConfig] = {"vae": _make_config()}
        registry = ModelManifest(components)
        components["extra"] = _make_config()
        assert "extra" not in registry


class TestSpeculativeDecoding:
    """Test a spec-decoding scenario with main + draft models."""

    def test_main_and_draft(self) -> None:
        main_model = _make_config("main-model")
        draft = _make_config("draft-model")
        registry = ModelManifest(
            {"main": main_model, "draft": draft},
        )
        assert registry["main"] is main_model
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
        with pytest.raises(KeyError, match="main"):
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

    def test_non_diffusers_stays_main(self, _mock_validate: Any) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=None):
            registry = ModelManifest.from_model_path(
                "org/llm-model", device_specs=[]
            )

        assert registry["main"].model_path == "org/llm-model"
        assert len(registry) == 1

    def test_rejects_kwargs_for_composite_model(
        self, _mock_validate: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            with pytest.raises(ValueError, match="ModelManifest directly"):
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

    def test_metadata_populated(self, _mock_validate: Any) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path("org/model")

        assert registry.metadata == {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
        }


@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestRevisionPropagation:
    """Verify that the revision parameter propagates to MAXModelConfig."""

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_revision_propagates_to_main(
        self, _mock_load: Any, _mock_validate: Any
    ) -> None:
        registry = ModelManifest.from_model_path(
            "test-model", revision="abc123", device_specs=[]
        )
        assert registry["main"].huggingface_model_revision == "abc123"

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
        return ModelManifest(
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
        manifest = ModelManifest({"main": cfg})
        updated = manifest.with_override(
            "main", weight_path=[Path("new.safetensors")]
        )
        assert updated["main"].weight_path == [Path("new.safetensors")]

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
        main_model = _make_config("org/main-model")
        draft = _make_config("org/draft-model")
        manifest = ModelManifest(
            {"main": main_model, "draft": draft},
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
        assert updated["main"] is main_model  # main unchanged


DEVICES_EXIST_TARGET = "max.pipelines.lib.config.model_config.devices_exist"
WEIGHT_PARSE_TARGET = (
    "max.pipelines.lib.config.model_config.WeightPathParser.parse"
)


class TestResolve:
    """Tests for ModelManifest.resolve()."""

    def test_resolve_calls_each_config(self) -> None:
        """resolve() delegates to MAXModelConfig.resolve() for every component."""
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        manifest = ModelManifest({"vae": vae, "unet": unet})

        with patch.object(MAXModelConfig, "resolve") as mock_resolve:
            manifest.resolve()

        assert mock_resolve.call_count == 2

    def test_resolve_empty_manifest(self) -> None:
        """resolve() on an empty manifest is a no-op."""
        manifest = ModelManifest({})
        manifest.resolve()  # should not raise

    def test_resolve_single_main(self) -> None:
        """resolve() works for a single-model manifest."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        with patch.object(MAXModelConfig, "resolve") as mock_resolve:
            manifest.resolve()

        mock_resolve.assert_called_once()

    @patch(DEVICES_EXIST_TARGET, return_value=True)
    @patch("max.pipelines.lib.config.model_config.validate_hf_repo_access")
    def test_resolve_flux2_with_overrides(
        self, _mock_validate: Any, _mock_devices: Any
    ) -> None:
        """Resolve a FLUX.2-dev manifest with transformer and VAE overrides.

        Simulates:
          - main repo: black-forest-labs/FLUX.2-dev (diffusers pipeline)
          - transformer weights from black-forest-labs/FLUX.2-dev-NVFP4
            with float4_e2m1fnx2 quantization
          - VAE replaced by fal/FLUX.2-Tiny-AutoEncoder
        """
        # Build the base diffusers manifest via direct construction
        # (avoids network calls that from_model_path would make).
        base_repo = "black-forest-labs/FLUX.2-dev"
        manifest = ModelManifest(
            {
                "transformer": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "vae": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "text_encoder": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "scheduler": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
            }
        )

        # Apply overrides: NVFP4 transformer weights + tiny VAE.
        manifest = manifest.with_override(
            "transformer",
            weight_path=[
                Path("black-forest-labs/FLUX.2-dev-NVFP4/weights.safetensors")
            ],
            quantization_encoding="float4_e2m1fnx2",
        ).with_override(
            "vae",
            config=_make_config(
                "fal/FLUX.2-Tiny-AutoEncoder",
                quantization_encoding="bfloat16",
            ),
        )

        # Verify pre-resolve state.
        assert (
            manifest["transformer"].quantization_encoding == "float4_e2m1fnx2"
        )
        assert manifest["transformer"].model_path == base_repo
        assert manifest["vae"].model_path == "fal/FLUX.2-Tiny-AutoEncoder"
        assert manifest["text_encoder"].model_path == base_repo
        assert manifest["scheduler"].model_path == base_repo

        # Mock WeightPathParser.parse to return weight_path unchanged
        # (avoids filesystem/network access).
        def fake_parse(
            model_path: str, weight_path: list[Path]
        ) -> tuple[list[Path], str | None]:
            return (weight_path, None)

        with patch(WEIGHT_PARSE_TARGET, side_effect=fake_parse):
            manifest.resolve()

        # Print resolved manifest for manual inspection.
        import logging

        logging.basicConfig(level=logging.INFO, force=True)
        manifest.log_model_info()

        # Post-resolve: configs should retain their values.
        assert (
            manifest["transformer"].quantization_encoding == "float4_e2m1fnx2"
        )
        assert manifest["transformer"].weight_path == [
            Path("black-forest-labs/FLUX.2-dev-NVFP4/weights.safetensors")
        ]
        assert manifest["vae"].model_path == "fal/FLUX.2-Tiny-AutoEncoder"
        assert manifest["vae"].quantization_encoding == "bfloat16"
        # Other components untouched.
        assert manifest["text_encoder"].model_path == base_repo
        assert manifest["scheduler"].model_path == base_repo


class TestFrozenAfterResolve:
    """Tests that ModelManifest rejects mutations after resolve()."""

    @staticmethod
    def _resolved_manifest() -> ModelManifest:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with patch.object(MAXModelConfig, "resolve"):
            manifest.resolve()
        return manifest

    def test_setitem_raises_after_resolve(self) -> None:
        manifest = self._resolved_manifest()
        with pytest.raises(TypeError, match="frozen after resolve"):
            manifest["new"] = _make_config("org/other")

    def test_delitem_raises_after_resolve(self) -> None:
        manifest = self._resolved_manifest()
        with pytest.raises(TypeError, match="frozen after resolve"):
            del manifest["main"]

    def test_update_raises_after_resolve(self) -> None:
        manifest = self._resolved_manifest()
        with pytest.raises(TypeError, match="frozen after resolve"):
            manifest.update({"new": _make_config("org/other")})

    def test_pop_raises_after_resolve(self) -> None:
        manifest = self._resolved_manifest()
        with pytest.raises(TypeError, match="frozen after resolve"):
            manifest.pop("main")

    def test_clear_raises_after_resolve(self) -> None:
        manifest = self._resolved_manifest()
        with pytest.raises(TypeError, match="frozen after resolve"):
            manifest.clear()

    def test_mutations_allowed_before_resolve(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        new_cfg = _make_config("org/other")
        manifest["draft"] = new_cfg
        assert manifest["draft"] is new_cfg


class TestTotalWeightsSize:
    """Tests for ModelManifest.total_weights_size."""

    def test_raises_before_resolve(self) -> None:
        """Accessing total_weights_size before resolve() raises RuntimeError."""
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(RuntimeError, match="must be resolved"):
            _ = manifest.total_weights_size

    def test_sums_component_weights(self) -> None:
        """total_weights_size sums weights_size() across all components."""
        manifest = ModelManifest(
            {
                "transformer": _make_config("org/model"),
                "vae": _make_config("org/model"),
            }
        )
        with (
            patch.object(MAXModelConfig, "resolve"),
            patch.object(
                MAXModelConfig,
                "weights_size",
                side_effect=[100, 200],
            ),
        ):
            manifest.resolve()
            assert manifest.total_weights_size == 300

    def test_empty_weight_path_contributes_zero(self) -> None:
        """Components with no weight_path contribute zero bytes."""
        scheduler = _make_config("org/model", weight_path=[])
        transformer = _make_config("org/model")
        manifest = ModelManifest(
            {"scheduler": scheduler, "transformer": transformer}
        )
        with (
            patch.object(MAXModelConfig, "resolve"),
            patch.object(
                MAXModelConfig,
                "weights_size",
                side_effect=[0, 500],
            ),
        ):
            manifest.resolve()
            assert manifest.total_weights_size == 500

    def test_empty_manifest_returns_zero(self) -> None:
        """An empty resolved manifest has total_weights_size == 0."""
        manifest = ModelManifest({})
        manifest.resolve()
        assert manifest.total_weights_size == 0


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
        manifest = ModelManifest({"main": cfg})

        restored = _msgpack_round_trip(manifest)

        assert list(restored.keys()) == ["main"]
        assert restored["main"].model_path == "org/llm-model"
        assert restored["main"].quantization_encoding == "bfloat16"

    def test_multi_component_msgpack_round_trip(self) -> None:
        """A multi-component manifest round-trips."""
        manifest = ModelManifest(
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

        assert set(restored.keys()) == {"vae", "unet"}
        assert restored["vae"].quantization_encoding == "bfloat16"
        assert restored["unet"].quantization_encoding == "float32"

    def test_speculative_decoding_msgpack_round_trip(self) -> None:
        """A main + draft manifest round-trips through msgpack."""
        manifest = ModelManifest(
            {
                "main": _make_config("org/target"),
                "draft": _make_config("org/draft"),
            },
        )

        restored = _msgpack_round_trip(manifest)

        assert restored["main"].model_path == "org/target"
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
        manifest = ModelManifest({"main": cfg})

        restored = _msgpack_round_trip(manifest)

        main_model = restored["main"]
        assert len(main_model.weight_path) == 2
        # After msgpack round-trip, Path objects are serialized as strings.
        # Coerce back to Path for comparison — this mirrors what Pydantic's
        # model_validate (as opposed to model_construct) would do.
        assert Path(main_model.weight_path[0]) == Path("shard-0.safetensors")
        assert Path(main_model.weight_path[1]) == Path("shard-1.safetensors")

    def test_empty_manifest_msgpack_round_trip(self) -> None:
        """An empty manifest round-trips through msgpack."""
        manifest = ModelManifest({})

        restored = _msgpack_round_trip(manifest)

        assert len(restored) == 0

    def test_subfolder_survives_msgpack_round_trip(self) -> None:
        """Subfolder field survives msgpack serialization."""
        cfg = _make_config("org/diffusion-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})
        manifest = ModelManifest({"transformer": cfg})

        restored = _msgpack_round_trip(manifest)

        assert restored["transformer"].subfolder == "transformer"

    def test_metadata_survives_msgpack_round_trip(self) -> None:
        """Metadata survives msgpack serialization."""
        meta = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
        }
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata == meta

    def test_empty_metadata_survives_msgpack_round_trip(self) -> None:
        """Empty metadata round-trips as empty dict, not None."""
        manifest = ModelManifest({"main": _make_config("org/model")})

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata == {}

    def test_metadata_with_diverse_types_survives_msgpack_round_trip(
        self,
    ) -> None:
        """Metadata with ints, floats, None, and nested structures round-trips."""
        meta: dict[str, Any] = {
            "_class_name": "StableDiffusionPipeline",
            "num_steps": 50,
            "guidance_scale": 7.5,
            "optional_field": None,
            "scheduler_config": {"beta_start": 0.0001, "beta_end": 0.02},
        }
        manifest = ModelManifest(
            {"unet": _make_config("org/model")}, metadata=meta
        )

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata["_class_name"] == "StableDiffusionPipeline"
        assert restored.metadata["num_steps"] == 50
        assert restored.metadata["guidance_scale"] == 7.5
        assert restored.metadata["optional_field"] is None
        assert restored.metadata["scheduler_config"] == {
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }


class TestMetadata:
    """Tests for the metadata property."""

    def test_empty_for_non_diffusion(self) -> None:
        """Non-diffusion manifests have empty metadata."""
        manifest = ModelManifest({"main": _make_config("org/llm-model")})
        assert manifest.metadata == {}

    def test_empty_for_direct_construction(self) -> None:
        """Direct construction without metadata kwarg gives empty dict."""
        manifest = ModelManifest(
            {
                "vae": _make_config("org/model"),
                "unet": _make_config("org/model"),
            }
        )
        assert manifest.metadata == {}

    def test_explicit_metadata(self) -> None:
        """Metadata is accessible when passed at construction."""
        meta = {"_class_name": "FluxPipeline", "is_distilled": True}
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )
        assert manifest.metadata == meta

    def test_metadata_is_defensive_copy(self) -> None:
        """Mutating the original dict does not affect the manifest."""
        meta: dict[str, Any] = {"_class_name": "FluxPipeline"}
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )
        meta["extra"] = "should not appear"
        assert "extra" not in manifest.metadata

    def test_with_override_preserves_metadata(self) -> None:
        """with_override carries metadata to the new manifest."""
        meta = {"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0"}
        manifest = ModelManifest(
            {
                "transformer": _make_config(
                    "org/model", quantization_encoding="bfloat16"
                ),
                "vae": _make_config("org/model"),
            },
            metadata=meta,
        )
        updated = manifest.with_override(
            "transformer", quantization_encoding="float4_e2m1fnx2"
        )
        assert updated.metadata == meta


class TestPrimaryArchitectureName:
    """Tests for the main_architecture_name property."""

    def test_non_diffusion_with_architectures(self) -> None:
        """Falls back to architectures[0] when _class_name is absent."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        class FakeHFConfig:
            architectures = ["LlamaForCausalLM"]

        with patch.object(
            type(cfg),
            "huggingface_config",
            new_callable=lambda: property(lambda self: FakeHFConfig()),
        ):
            assert manifest.main_architecture_name == "LlamaForCausalLM"

    def test_non_diffusion_prefers_architectures_over_class_name(
        self,
    ) -> None:
        """Prefers architectures[0] over _class_name for registry lookup."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        class FakeHFConfig:
            _class_name = "CustomModelForCausalLM"
            architectures = ["LlamaForCausalLM"]

        with patch.object(
            type(cfg),
            "huggingface_config",
            new_callable=lambda: property(lambda self: FakeHFConfig()),
        ):
            assert manifest.main_architecture_name == "LlamaForCausalLM"

    def test_non_diffusion_no_hf_config_raises(self) -> None:
        """Raises ValueError when huggingface_config is unavailable."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        with (
            patch.object(
                type(cfg),
                "huggingface_config",
                new_callable=lambda: property(lambda self: None),
            ),
            patch.object(
                type(cfg),
                "diffusers_config",
                new_callable=lambda: property(lambda self: None),
            ),
        ):
            with pytest.raises(
                ValueError, match="Cannot determine architecture name"
            ):
                _ = manifest.main_architecture_name

    @patch(HF_OFFLINE_TARGET, False)
    @patch(VALIDATE_HF_ACCESS_TARGET)
    def test_diffusion_returns_class_name(self, _mock_validate: Any) -> None:
        """Diffusion manifests return _class_name from stored metadata."""
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            manifest = ModelManifest.from_model_path("org/diffusion-model")

        # No need to re-load model_index — metadata is stored.
        assert manifest.main_architecture_name == "FluxPipeline"

    def test_diffusion_no_class_name_raises(self) -> None:
        """Raises ValueError when metadata has no _class_name."""
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")},
            metadata={"_diffusers_version": "0.30.0"},
        )
        with pytest.raises(ValueError, match="metadata has no"):
            _ = manifest.main_architecture_name

    def test_empty_manifest_raises(self) -> None:
        """Raises ValueError for an empty manifest."""
        manifest = ModelManifest({})
        with pytest.raises(ValueError, match="manifest is empty"):
            _ = manifest.main_architecture_name


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
            for role, cfg in manifest.items()
        },
        "metadata": manifest.metadata,
    }
    packed = msgspec.msgpack.encode(payload)
    unpacked = msgspec.msgpack.decode(packed)

    models = {
        role: MAXModelConfig.model_construct(**cfg_data)
        for role, cfg_data in unpacked["models"].items()
    }
    return ModelManifest(models, metadata=unpacked.get("metadata"))
