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
"""Structural tests for the OpenELM MAX pipeline."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from max.graph.weights import Weights


class TestArchitectureRegistration:
    """Verify the openelm_arch SupportedArchitecture registration."""

    def test_architecture_name_matches_hf_config(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch

        assert openelm_arch.name == "OpenELMForCausalLM", (
            f"Architecture name is '{openelm_arch.name}'. "
            "Must be 'OpenELMForCausalLM' to match OpenELM's config.json."
        )

    def test_architecture_has_pipeline_model(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch
        from max.pipelines.lib.interfaces.pipeline_model import PipelineModel

        assert openelm_arch.pipeline_model is not None
        assert issubclass(openelm_arch.pipeline_model, PipelineModel), (
            f"{openelm_arch.pipeline_model.__name__} must inherit from PipelineModel"
        )

    def test_architecture_has_tokenizer(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch

        assert openelm_arch.tokenizer is not None

    def test_architecture_has_example_repos(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch

        assert len(openelm_arch.example_repo_ids) > 0
        for repo in openelm_arch.example_repo_ids:
            assert "openelm" in repo.lower(), (
                f"Repo '{repo}' does not look like an OpenELM repo"
            )

    def test_architecture_supports_float32(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch

        assert "float32" in openelm_arch.supported_encodings

    def test_architecture_supports_bfloat16(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch

        assert "bfloat16" in openelm_arch.supported_encodings

    def test_default_weights_format_is_safetensors(self) -> None:
        from max.graph.weights import WeightsFormat
        from max.pipelines.architectures.openelm import openelm_arch

        assert (
            openelm_arch.default_weights_format == WeightsFormat.safetensors
        ), (
            f"default_weights_format is {openelm_arch.default_weights_format}, "
            "expected WeightsFormat.safetensors"
        )

    def test_architecture_has_multi_kv_cache_config(self) -> None:
        from max.pipelines.architectures.openelm import openelm_arch
        from max.pipelines.architectures.openelm.model_config import (
            OpenELMConfig,
        )

        assert openelm_arch.config is OpenELMConfig
        assert openelm_arch.memory_planner is not None
        assert openelm_arch.batching is not None


class TestLayerConfigs:
    """Verify compute_layer_configs() for both uniform and layer-wise configs."""

    def _make_uniform_config(self, num_layers: int = 4) -> SimpleNamespace:
        return SimpleNamespace(
            num_transformer_layers=num_layers,
            head_dim=64,
            model_dim=1024,
            num_query_heads=4,
            num_kv_heads=2,
            ffn_multipliers=4.0,
            ffn_dim_divisor=256,
            max_context_length=2048,
            rms_norm_eps=1e-6,
            rope_freq_constant=10000,
            vocab_size=32000,
        )

    def _make_layerwise_config(self, num_layers: int = 8) -> SimpleNamespace:
        return SimpleNamespace(
            num_transformer_layers=num_layers,
            head_dim=64,
            model_dim=1024,
            num_query_heads=list(range(4, 4 + num_layers)),
            num_kv_heads=[max(1, h // 2) for h in range(4, 4 + num_layers)],
            ffn_multipliers=[4.0 + i * 0.25 for i in range(num_layers)],
            ffn_dim_divisor=256,
            max_context_length=2048,
            rms_norm_eps=1e-6,
            rope_freq_constant=10000,
            vocab_size=32000,
        )

    def test_correct_number_of_layer_configs(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        for num_layers in [4, 16, 20, 28, 36]:
            cfg = self._make_uniform_config(num_layers)
            configs = compute_layer_configs(cfg)
            assert len(configs) == num_layers, (
                f"Expected {num_layers} configs, got {len(configs)}"
            )

    def test_uniform_config_produces_identical_layers(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        cfg = self._make_uniform_config(num_layers=4)
        configs = compute_layer_configs(cfg)
        first = configs[0]

        for i, c in enumerate(configs[1:], start=1):
            assert c.num_query_heads == first.num_query_heads, (
                f"Layer {i} query heads differ in a uniform config"
            )
            assert c.ffn_hidden_dim == first.ffn_hidden_dim, (
                f"Layer {i} FFN dim differs in a uniform config"
            )

    def test_layerwise_config_produces_increasing_heads(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        cfg = self._make_layerwise_config(num_layers=8)
        configs = compute_layer_configs(cfg)
        heads = [c.num_query_heads for c in configs]

        assert heads == sorted(heads), (
            f"Expected query heads to increase monotonically, got: {heads}"
        )

    def test_ffn_hidden_dim_is_multiple_of_256(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        for cfg in [self._make_uniform_config(), self._make_layerwise_config()]:
            configs = compute_layer_configs(cfg)
            for i, c in enumerate(configs):
                assert c.ffn_hidden_dim % 256 == 0, (
                    f"Layer {i} FFN hidden dim {c.ffn_hidden_dim} is not a multiple of 256"
                )

    def test_ffn_hidden_dim_is_positive(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        cfg = self._make_uniform_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.ffn_hidden_dim > 0, (
                f"Layer {i} FFN hidden dim is {c.ffn_hidden_dim}"
            )

    def test_head_dim_is_preserved(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        cfg = self._make_uniform_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.head_dim == cfg.head_dim, (
                f"Layer {i} head_dim is {c.head_dim}, expected {cfg.head_dim}"
            )

    def test_kv_heads_never_exceed_query_heads(self) -> None:
        from max.pipelines.architectures.openelm.model_config import (
            compute_layer_configs,
        )

        cfg = self._make_layerwise_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.num_kv_heads <= c.num_query_heads, (
                f"Layer {i}: kv_heads ({c.num_kv_heads}) > query_heads ({c.num_query_heads})"
            )


class _FakeWeights:
    """Minimal stand-in for max.graph.weights.Weights: just needs .data()."""

    def __init__(self, array: Any) -> None:
        from max.graph.weights import WeightData

        self._weight_data = WeightData.from_numpy(array, "unused")

    def data(self) -> Any:
        return self._weight_data


class TestWeightAdapter:
    """Verify convert_safetensor_state_dict's renaming and pass-through behavior."""

    def test_adapter_returns_all_weights(self) -> None:
        import numpy as np
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        mock_weights = {
            "transformer.token_embeddings.weight": _FakeWeights(
                np.zeros((32000, 1024), dtype=np.float32)
            ),
            "transformer.layers.0.attn_norm.weight": _FakeWeights(
                np.ones((1024,), dtype=np.float32)
            ),
            "transformer.layers.0.attn.qkv_proj.weight": _FakeWeights(
                np.zeros((512, 1024), dtype=np.float32)
            ),
            "transformer.norm.weight": _FakeWeights(
                np.ones((1024,), dtype=np.float32)
            ),
        }

        result = convert_safetensor_state_dict(
            cast("dict[str, Weights]", mock_weights)
        )
        assert len(result) == len(mock_weights), (
            f"Expected {len(mock_weights)} weights, got {len(result)}"
        )

    def test_adapter_renames_out_proj_to_o_proj(self) -> None:
        import numpy as np
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        mock_weights = {
            "transformer.layers.0.attn.out_proj.weight": _FakeWeights(
                np.zeros((1024, 1024), dtype=np.float32)
            ),
        }

        result = convert_safetensor_state_dict(
            cast("dict[str, Weights]", mock_weights)
        )
        assert "transformer.layers.0.attn.o_proj.weight" in result, (
            "Expected 'out_proj' to be renamed to 'o_proj' to match "
            "AttentionWithRope's attribute name"
        )
        assert "transformer.layers.0.attn.out_proj.weight" not in result

    def test_adapter_preserves_other_names(self) -> None:
        import numpy as np
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        mock_weights = {
            "transformer.token_embeddings.weight": _FakeWeights(
                np.zeros((32000, 1024), dtype=np.float32)
            ),
            "transformer.layers.0.ffn.proj_1.weight": _FakeWeights(
                np.zeros((512, 1024), dtype=np.float32)
            ),
        }

        result = convert_safetensor_state_dict(
            cast("dict[str, Weights]", mock_weights)
        )
        for name in mock_weights:
            assert name in result, (
                f"Weight '{name}' missing from adapter output"
            )

    def test_adapter_renames_transformer_norm_to_norm(self) -> None:
        import numpy as np
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_safetensor_state_dict(
            cast(
                "dict[str, Weights]",
                {"transformer.norm.weight": _FakeWeights(original)},
            )
        )

        assert "norm.weight" in result, (
            "Expected 'transformer.norm.weight' to be renamed to 'norm.weight' "
            "to match LogitsPostprocessMixin's top-level 'norm' attribute"
        )
        assert "transformer.norm.weight" not in result

        underlying = np.from_dlpack(result["norm.weight"].data)
        assert np.array_equal(original, underlying), (
            "Tensor values were modified by the adapter"
        )

    def test_adapter_preserves_tensor_values(self) -> None:
        import numpy as np
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_safetensor_state_dict(
            cast(
                "dict[str, Weights]",
                {
                    "transformer.layers.0.ffn_norm.weight": _FakeWeights(
                        original
                    )
                },
            )
        )

        underlying = np.from_dlpack(
            result["transformer.layers.0.ffn_norm.weight"].data
        )
        assert np.array_equal(original, underlying), (
            "Tensor values were modified by the adapter"
        )

    def test_adapter_handles_empty_input(self) -> None:
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )

        result = convert_safetensor_state_dict({})
        assert result == {}
