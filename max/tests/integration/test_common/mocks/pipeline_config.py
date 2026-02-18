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
from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.graph.weights import WeightsFormat
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    SupportedEncoding,
)
from transformers import AutoConfig
from typing_extensions import ParamSpec

from .memory_estimation import mock_estimate_memory_footprint

_P = ParamSpec("_P")
_R = TypeVar("_R")


class DummyMAXModelConfig(MAXModelConfig):
    def weights_size(self) -> int:
        return 1000

    def validate_and_resolve_quantization_encoding_weight_path(
        self, default_encoding: SupportedEncoding
    ) -> None:
        pass

    def validate_and_resolve_with_resolved_quantization_encoding(
        self,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        default_weights_format: WeightsFormat,
    ) -> None:
        pass


class DummyPipelineConfig(PipelineConfig):
    def __init__(
        self,
        model_path: str,
        quantization_encoding: SupportedEncoding,
        max_batch_size: int | None,
        max_length: int | None,
        pdl_level: str = "1",
        device_specs: list[DeviceSpec] | None = None,
        kv_cache_strategy: KVCacheStrategy = "model_default",
        # TODO(AITLIB-328): These values do not belong in PipelineConfig,
        # but are somehow used by MockPipelineModel in pipeline_model.py.
        eos_prob: float | None = None,
        vocab_size: int | None = None,
        eos_token: int | None = None,
    ) -> None:
        # Mirror the construction pattern used by other test fixtures:
        # - Keep PipelineConfig surface minimal
        # - Populate nested configs via `model_construct`
        # - Attach nested configs via PipelineConfig's private attrs
        #
        # This avoids invoking expensive validation / resolution logic and keeps
        # the config shape aligned with production code (e.g. `_model`, not
        # legacy `_model_config`).
        if device_specs is None:
            device_specs = []

        # Seed `self` with a real (but unvalidated) PipelineConfig instance, so
        # we keep pydantic-internal state consistent while still avoiding full
        # validation / resolution.
        base = PipelineConfig.model_construct(
            max_batch_size=max_batch_size,
            max_length=max_length,
            pdl_level=pdl_level,
        )
        self.__dict__.update(base.__dict__)
        for attr in (
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
        ):
            if hasattr(base, attr):
                object.__setattr__(self, attr, getattr(base, attr))

        model_config = DummyMAXModelConfig.model_construct(
            model_path=model_path,
            device_specs=device_specs,
            quantization_encoding=quantization_encoding,
        )
        model_config.kv_cache = KVCacheConfig(
            cache_strategy=kv_cache_strategy,
        )
        # NOTE: Using MagicMock without spec here because HuggingFace configs
        # vary by model type (LlamaConfig, Qwen2Config, etc.). Tests that need
        # strict type checking should pass a model-specific huggingface_config
        # parameter to DummyPipelineConfig or use the real AutoConfig.
        # TODO: Consider accepting huggingface_config as an optional parameter
        # to allow tests to provide model-specific spec'd mocks.
        model_config._huggingface_config = MagicMock()
        # Populate the model field so callers see the configured model.
        object.__setattr__(self, "model", model_config)

        # These values don't belong in PipelineConfig, but are used by
        # MockPipelineModel in pipeline_model.py.
        object.__setattr__(self, "eos_prob", eos_prob)
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "eos_token", eos_token)


def mock_huggingface_config(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mock HuggingFace config to return correct architectures for test models.

    NOTE: Uses MagicMock without spec because HuggingFace configs vary by model
    type and have deeply nested structure (e.g., vision_config, llm_config).
    Tests requiring strict type checking should use real AutoConfig.from_pretrained()
    with appropriate network mocking or provide a specific config class as spec.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        def mock_from_pretrained(  # noqa: ANN202
            model_name_or_path: str | os.PathLike[str], **kwargs: Any
        ):
            # Create a mock config with the correct architectures based on model.
            mock_config = MagicMock()

            # Map model name patterns to their architectures.
            # Uses substring matching to handle both direct repo IDs and
            # local cache paths like /fake/cache/{repo_id}.
            model_architectures = {
                "OpenGVLab/InternVL2-8B": ["InternVLChatModel"],
                "modularai/Llama-3.1-8B-Instruct-GGUF": ["LlamaForCausalLM"],
                "Llama-3.1-8B-Instruct": ["LlamaForCausalLM"],
                "HuggingFaceTB/SmolLM-135M": ["LlamaForCausalLM"],
                "trl-internal-testing/tiny-random-LlamaForCausalLM": [
                    "LlamaForCausalLM"
                ],
                # Add other specific mappings as needed
            }

            path_str = str(model_name_or_path)
            mock_config.architectures = []

            # Use substring matching to handle cache paths like /fake/cache/{repo_id}
            for model_pattern, architectures in model_architectures.items():
                if model_pattern in path_str:
                    mock_config.architectures = architectures
                    break

            # Provide concrete numeric attributes expected by MAX model configs
            def _populate_llama_like_cfg(cfg: Any) -> None:
                # Use small, consistent integers that satisfy head_dim divisibility
                cfg.hidden_size = 4096
                cfg.num_attention_heads = 32
                cfg.num_key_value_heads = 32
                cfg.num_hidden_layers = 2
                cfg.rope_theta = 10000.0
                cfg.max_position_embeddings = 2048
                cfg.intermediate_size = 11008
                cfg.vocab_size = 32000
                cfg.rms_norm_eps = 1e-5
                cfg.model_type = "llama"
                # Optional fields used in some paths
                cfg.rope_scaling = None
                del cfg.head_dim

            if any(
                x in path_str
                for x in [
                    "Llama-3.1-8B-Instruct",
                    "HuggingFaceTB/SmolLM-135M",
                    "trl-internal-testing/tiny-random-LlamaForCausalLM",
                ]
            ):
                _populate_llama_like_cfg(mock_config)

            if "OpenGVLab/InternVL2-8B" in path_str:
                # For InternVL, we need both llm_config and vision_config
                llm_cfg = MagicMock()
                _populate_llama_like_cfg(llm_cfg)
                mock_config.llm_config = llm_cfg

                vision_cfg = MagicMock()
                # Minimal set used by VisionConfig.generate()
                vision_cfg.hidden_size = 1024
                vision_cfg.num_attention_heads = 16
                vision_cfg.intermediate_size = 4096
                vision_cfg.image_size = 448
                vision_cfg.patch_size = 14
                vision_cfg.layer_norm_eps = 1e-6
                vision_cfg.qk_normalization = True
                vision_cfg.qkv_bias = False
                vision_cfg.num_hidden_layers = 32
                mock_config.vision_config = vision_cfg

            return mock_config

        with patch.object(
            AutoConfig, "from_pretrained", side_effect=mock_from_pretrained
        ):
            return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_repo_exists_with_retry(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    def fake_generate_local_model_path(
        repo_id: str, revision: str | None = None
    ) -> str:
        # Return a fake path that preserves the repo_id for identification
        return f"/fake/cache/{repo_id}"

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch("huggingface_hub.revision_exists", return_value=True):
            with patch(
                "max.pipelines.lib.hf_utils.generate_local_model_path",
                side_effect=fake_generate_local_model_path,
            ):
                return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_file_exists(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch("huggingface_hub.file_exists", return_value=True):
            return func(*args, **kwargs)

    return wrapper


def mock_generate_local_model_path(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Mock generate_local_model_path to return a fake path that preserves model identity.

    This allows tests to run with HF_HUB_OFFLINE=1 without requiring
    models to be in the local cache. The returned path includes the repo_id
    so that downstream mocks (like mock_huggingface_config) can still identify
    which model is being requested.
    """

    def fake_generate_local_model_path(
        repo_id: str, revision: str | None = None
    ) -> str:
        # Return a fake path that preserves the repo_id for identification
        return f"/fake/cache/{repo_id}"

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch(
            "max.pipelines.lib.hf_utils.generate_local_model_path",
            side_effect=fake_generate_local_model_path,
        ):
            return func(*args, **kwargs)

    return wrapper


def mock_pipeline_config_hf_dependencies(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Decorator that combines multiple mock decorators for pipeline testing.

    Combines:
    - mock_generate_local_model_path
    - mock_huggingface_hub_repo_exists_with_retry
    - mock_huggingface_hub_file_exists
    - mock_huggingface_config
    - mock_estimate_memory_footprint
    """
    return mock_generate_local_model_path(
        mock_huggingface_hub_repo_exists_with_retry(
            mock_huggingface_hub_file_exists(
                mock_huggingface_config(mock_estimate_memory_footprint(func))
            )
        )
    )


# This is a helper decorator to mock the PipelineConfig.resolve() method.
# In practice, it is used to skip all the other validation and resolution steps.
# We're just testing if the config fields are set correctly.
def mock_pipeline_config_resolve(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch(
            "max.pipelines.lib.config.PipelineConfig.resolve", return_value=None
        ):
            return func(*args, **kwargs)

    return wrapper
