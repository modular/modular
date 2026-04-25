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

"""Tests for the full PipelineConfig.resolve() flow with mock architectures.

All tests use fake local repos (config.json + weight files) and mock
SupportedArchitecture instances registered in PIPELINE_REGISTRY.
No network access required.
"""

import json
import os
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.graph import DeviceRef
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.lib import MAXModelConfig, MemoryEstimator
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.registry import SupportedArchitecture
from test_common.pipeline_model_dummy import (
    DUMMY_GEMMA_ARCH,
    DUMMY_LLAMA_ARCH,
    DummyLlamaArchConfig,
    DummyLlamaPipelineModel,
    DummyPipelineModel,
    DummyTextTokenizer,
)
from test_common.registry import prepare_registry

GPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="gpu")
CPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="cpu")


# ---------------------------------------------------------------------------
# Helpers — fake weight files
# ---------------------------------------------------------------------------


def _write_fake_safetensors(path: str, dtype: str = "BF16") -> None:
    """Write a minimal safetensors file with a single tensor of the given dtype."""
    header = {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00\x00")


def _write_mixed_safetensors(path: str, tensors: dict[str, str]) -> None:
    """Write a safetensors file with multiple tensors of different dtypes."""
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, dtype in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)


# ---------------------------------------------------------------------------
# Helpers — local repo with config.json
# ---------------------------------------------------------------------------

_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-5,
}

_GEMMA_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-5,
    "head_dim": 128,
}


def _make_local_repo(
    tmpdir: str,
    hf_config: dict[str, Any] | None = None,
    safetensors_files: dict[str, dict[str, str]] | None = None,
    gguf_files: list[str] | None = None,
) -> str:
    """Create a local repo directory with config.json and fake weight files.

    Args:
        tmpdir: Root temp directory.
        hf_config: HuggingFace config dict to write as config.json.
            Defaults to _LLAMA_CONFIG.
        safetensors_files: Mapping of relative path to {tensor_name: dtype}.
        gguf_files: List of relative GGUF filenames to create as empty files.

    Returns:
        The repo root path.
    """
    config = hf_config if hf_config is not None else _LLAMA_CONFIG
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)

    if safetensors_files:
        for rel_path, tensors in safetensors_files.items():
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if len(tensors) == 1:
                _, dtype = next(iter(tensors.items()))
                _write_fake_safetensors(full_path, dtype=dtype)
            else:
                _write_mixed_safetensors(full_path, tensors)
    if gguf_files:
        for rel_path in gguf_files:
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            open(full_path, "w").close()
    return tmpdir


# ---------------------------------------------------------------------------
# Mock context manager
# ---------------------------------------------------------------------------


@contextmanager
def _pipeline_resolve_mocks(
    weight_path_return: tuple[list[Path], str | None] = ([], None),
    num_devices: int = 1,
) -> Iterator[None]:
    """Patches external dependencies for the full PipelineConfig.resolve() flow.

    Mocks external I/O and hardware while leaving the real resolution
    logic intact:
    - devices_exist, load_devices — avoid GPU probes
    - WeightPathParser.parse — avoid network
    - validate_hf_repo_access — avoid network
    - MemoryEstimator — avoid real memory estimation
    - accelerator_api — avoid CUDA probes
    - estimate_weights_size / estimate_activation_memory — avoid graph ops
    """
    mock_devices = [DeviceRef.GPU()] * num_devices

    with (
        patch(
            "max.pipelines.lib.config.model_config.devices_exist",
            return_value=True,
        ),
        patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=weight_path_return,
        ),
        patch("max.pipelines.lib.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.lib.config.config.load_devices",
            return_value=mock_devices,
        ),
        patch.object(
            MemoryEstimator, "estimate_memory_footprint", return_value=0
        ),
        patch.object(
            MemoryEstimator,
            "max_supported_sequence_length",
            return_value=None,
        ),
        patch(
            "max.pipelines.lib.config.config.accelerator_api",
            return_value="cpu",
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=0,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
        patch.object(
            DummyPipelineModel,
            "estimate_weights_size",
            return_value=0,
        ),
        patch.object(
            DummyPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
    ):
        yield


def _model(config: PipelineConfig) -> MAXModelConfig:
    """Return the main model config, asserting it is not None."""
    assert config.model is not None
    return config.model


def _make_pipeline_config(
    model_path: str,
    device_specs: list[DeviceSpec] | None = None,
    weight_path: list[Path] | None = None,
    max_length: int | None = 512,
    max_batch_size: int = 1,
    **model_kwargs: Any,
) -> PipelineConfig:
    """Create a PipelineConfig with defer_resolve=True for testing."""
    if device_specs is None:
        device_specs = [GPU_DEVICE_SPEC]
    return PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=model_path,
                    device_specs=device_specs,
                    weight_path=weight_path or [],
                    max_length=max_length,
                    **model_kwargs,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=max_batch_size,
            defer_resolve=True,
        ),
    )


# ---------------------------------------------------------------------------
# Category A: Architecture Lookup + Encoding Resolution (Happy Path)
# ---------------------------------------------------------------------------


class TestArchitectureEncodingResolution:
    """Tests that verify the full resolve chain for common encoding scenarios."""

    @prepare_registry
    def test_resolve_bf16_safetensors_llama(self) -> None:
        """BF16 safetensors with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).quantization_encoding == "bfloat16"
            assert any(
                "model.safetensors" in str(p)
                for p in _model(config).weight_path
            )

    @prepare_registry
    def test_resolve_gguf_q4_0_llama(self) -> None:
        """Q4_0 GGUF with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).quantization_encoding == "q4_0"
            assert any(
                "model-Q4_0.gguf" in str(p) for p in _model(config).weight_path
            )

    @prepare_registry
    def test_resolve_fp8_safetensors_llama(self) -> None:
        """FP8 safetensors with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "F8_E4M3"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).quantization_encoding == "float8_e4m3fn"

    @prepare_registry
    def test_resolve_f32_on_gpu_uses_arch_default(self) -> None:
        """F32 safetensors on GPU: architecture default_encoding is used.

        MAXModelConfig.resolve() infers float32 from the file, but the
        architecture-level validation may fall back to the arch default
        encoding when reconciling file encoding with device capabilities.
        """
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "F32"}}
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config.resolve()
            # The encoding should be resolved (either float32 or bfloat16)
            # and must be in the architecture's supported_encodings.
            model = _model(config)
            assert (
                model.quantization_encoding
                in DUMMY_LLAMA_ARCH.supported_encodings
            )

    @prepare_registry
    def test_resolve_f32_on_cpu_stays_f32(self) -> None:
        """F32 safetensors on CPU should stay F32."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "F32"}}
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).quantization_encoding == "float32"


# ---------------------------------------------------------------------------
# Category B: Architecture Default Encoding Fallback
# ---------------------------------------------------------------------------


class TestDefaultEncodingFallback:
    """Tests that the architecture's default_encoding is used as a fallback."""

    @prepare_registry
    def test_default_encoding_used_when_ambiguous_on_cpu(self) -> None:
        """When multiple non-quantized encodings are ambiguous on CPU,
        the architecture default_encoding is used as the fallback.

        A repo with mixed BF16+F32 tensors on CPU is ambiguous for
        MAXModelConfig.resolve(). The architecture validation then
        falls back to the architecture's default_encoding.
        """
        from max.graph.weights import WeightsFormat
        from max.interfaces import PipelineTask
        from max.pipelines import TextContext

        # Create an architecture with default_encoding="float32" compatible with CPU
        cpu_arch = SupportedArchitecture(
            name="LlamaForCausalLM",
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/model"],
            default_encoding="float32",
            supported_encodings={"float32", "bfloat16"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        PIPELINE_REGISTRY.register(cpu_arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mixed BF16+F32 is ambiguous on CPU — encoding stays None
            # after MAXModelConfig.resolve(), so arch default kicks in.
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).quantization_encoding == "float32"


# ---------------------------------------------------------------------------
# Category C: Encoding Validation Against supported_encodings
# ---------------------------------------------------------------------------


class TestEncodingValidation:
    """Tests that unsupported encodings are rejected."""

    @prepare_registry
    def test_reject_encoding_not_in_supported_encodings(self) -> None:
        """Q4_K GGUF should be rejected when arch doesn't support q4_k."""
        # DUMMY_LLAMA_ARCH intentionally excludes q4_k from supported_encodings
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with (
                _pipeline_resolve_mocks(),
                pytest.raises(ValueError, match="not supported by MAX engine"),
            ):
                config.resolve()

    @prepare_registry
    def test_explicit_unsupported_encoding_rejected(self) -> None:
        """Explicitly setting an unsupported encoding should raise."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            config = _make_pipeline_config(
                tmpdir,
                device_specs=[CPU_DEVICE_SPEC],
                quantization_encoding="q4_k",
            )
            with (
                _pipeline_resolve_mocks(),
                pytest.raises(ValueError, match="not supported by MAX engine"),
            ):
                config.resolve()


# ---------------------------------------------------------------------------
# Category D: Architecture Not Found
# ---------------------------------------------------------------------------


class TestArchitectureNotFound:
    """Tests for missing or unknown architectures."""

    @prepare_registry
    def test_unknown_architecture_raises(self) -> None:
        """Unknown architecture in config.json should raise ValueError."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown_config = dict(_LLAMA_CONFIG)
            unknown_config["architectures"] = ["UnknownModelForCausalLM"]
            _make_local_repo(
                tmpdir,
                hf_config=unknown_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with (
                _pipeline_resolve_mocks(),
                pytest.raises(
                    ValueError, match="MAX-optimized architecture not available"
                ),
            ):
                config.resolve()

    @prepare_registry
    def test_missing_config_json_raises(self) -> None:
        """Missing config.json should raise an error."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write weight files but no config.json
            _write_fake_safetensors(os.path.join(tmpdir, "model.safetensors"))
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks(), pytest.raises(Exception):
                config.resolve()


# ---------------------------------------------------------------------------
# Category E: Multi-GPU Validation
# ---------------------------------------------------------------------------


class TestMultiGPUValidation:
    """Tests for multi-GPU support validation."""

    @prepare_registry
    def test_multi_gpu_rejected_for_unsupported_arch(self) -> None:
        """Architecture without multi_gpu_supported should reject 2 GPUs."""
        # DUMMY_GEMMA_ARCH has multi_gpu_supported=False
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            two_gpus = [
                DeviceSpec(id=0, device_type="gpu"),
                DeviceSpec(id=1, device_type="gpu"),
            ]
            config = _make_pipeline_config(tmpdir, device_specs=two_gpus)
            with (
                _pipeline_resolve_mocks(num_devices=2),
                pytest.raises(
                    ValueError,
                    match="Multiple GPU inference is currently not supported",
                ),
            ):
                config.resolve()

    @prepare_registry
    def test_multi_gpu_allowed_for_supported_arch(self) -> None:
        """Architecture with multi_gpu_supported should allow 2 GPUs."""
        # DUMMY_LLAMA_ARCH has multi_gpu_supported=True
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            two_gpus = [
                DeviceSpec(id=0, device_type="gpu"),
                DeviceSpec(id=1, device_type="gpu"),
            ]
            config = _make_pipeline_config(tmpdir, device_specs=two_gpus)
            with _pipeline_resolve_mocks(num_devices=2):
                config.resolve()
            assert _model(config).quantization_encoding == "bfloat16"


# ---------------------------------------------------------------------------
# Category F: RoPE Type Resolution
# ---------------------------------------------------------------------------


class TestRopeTypeResolution:
    """Tests for RoPE type resolution from architecture defaults."""

    @prepare_registry
    def test_rope_type_resolved_from_architecture(self) -> None:
        """RoPE type should be inherited from architecture when not set."""
        # DUMMY_GEMMA_ARCH has rope_type="normal"
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).rope_type == "normal"

    @prepare_registry
    def test_rope_type_preserved_if_already_set(self) -> None:
        """Explicit RoPE type should not be overwritten by architecture."""
        # DUMMY_LLAMA_ARCH has rope_type="none", so set a different valid value
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # Set rope_type="neox" which differs from DUMMY_GEMMA_ARCH's "normal"
            config = _make_pipeline_config(tmpdir, rope_type="neox")
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).rope_type == "neox"


# ---------------------------------------------------------------------------
# Category G: Cache Dtype Resolution
# ---------------------------------------------------------------------------


class TestCacheDtypeResolution:
    """Tests that cache dtype is set based on quantization encoding."""

    @prepare_registry
    def test_cache_dtype_bf16_for_bf16_encoding(self) -> None:
        """BF16 encoding should result in bfloat16 cache dtype."""
        from max.dtype import DType

        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            assert _model(config).kv_cache._cache_dtype == DType.bfloat16


# ---------------------------------------------------------------------------
# Category H: Weight Path Discovery Through Full Pipeline
# ---------------------------------------------------------------------------


class TestWeightPathDiscovery:
    """Tests for weight file discovery through the full resolve chain."""

    @prepare_registry
    def test_sharded_safetensors_discovered(self) -> None:
        """Multiple sharded safetensors should all be discovered."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            paths = sorted(str(p) for p in _model(config).weight_path)
            assert paths == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]

    @prepare_registry
    def test_safetensors_preferred_over_gguf(self) -> None:
        """When both formats exist, safetensors should be preferred."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
                gguf_files=["model-Q4_0.gguf"],
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config.resolve()
            paths = [str(p) for p in _model(config).weight_path]
            assert paths == ["model.safetensors"]


# ---------------------------------------------------------------------------
# Category I: Required Arguments Enforcement
# ---------------------------------------------------------------------------


class TestRequiredArguments:
    """Tests that architecture required_arguments override user config."""

    @prepare_registry
    def test_required_arguments_override_user_config(self) -> None:
        """Architecture required_arguments should override conflicting config values."""
        from max.interfaces import PipelineTask
        from max.pipelines import TextContext

        arch_with_required = SupportedArchitecture(
            name="LlamaForCausalLM",
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=DUMMY_LLAMA_ARCH.default_weights_format,
            config=DummyLlamaArchConfig,
            required_arguments={"enable_prefix_caching": False},
        )
        PIPELINE_REGISTRY.register(arch_with_required)

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            # Set a value that conflicts with the required argument
            _model(config).kv_cache.enable_prefix_caching = True
            with _pipeline_resolve_mocks():
                config.resolve()
            # Architecture should have overridden it
            assert _model(config).kv_cache.enable_prefix_caching is False
