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

"""Tests for MAXModelConfig.resolve() encoding inference and weight path resolution.

All tests use fake local safetensors/GGUF repos with no network access.
"""

import json
import os
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.hf_utils import HuggingFaceRepo

GPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="gpu")
CPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="cpu")


# ---------------------------------------------------------------------------
# Helpers
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
    """Write a safetensors file with multiple tensors of different dtypes.

    Args:
        path: File path to write.
        tensors: Mapping of tensor name to safetensors dtype string,
            e.g. {"model.layers.0.weight": "U8", "model.norm.weight": "BF16"}.
    """
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


def _make_local_repo(
    tmpdir: str,
    safetensors_files: dict[str, dict[str, str]] | None = None,
    gguf_files: list[str] | None = None,
) -> str:
    """Create a local repo directory with fake weight files.

    Args:
        tmpdir: Root temp directory.
        safetensors_files: Mapping of relative path to {tensor_name: dtype}.
            If the dict has one entry, uses _write_fake_safetensors for simplicity.
        gguf_files: List of relative GGUF filenames to create as empty files.

    Returns:
        The repo root path.
    """
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


@contextmanager
def _resolve_mocks(
    weight_path_return: tuple[list[Path], str | None] = ([], None),
) -> Iterator[None]:
    """Context manager that patches external dependencies for resolve().

    Args:
        weight_path_return: Return value for WeightPathParser.parse.
    """
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
    ):
        yield


def _make_config(
    model_path: str,
    device_specs: list[DeviceSpec] | None = None,
    weight_path: list[Path] | None = None,
    **kwargs,
) -> MAXModelConfig:
    """Create a MAXModelConfig for testing."""
    if device_specs is None:
        device_specs = [GPU_DEVICE_SPEC]
    return MAXModelConfig(
        model_path=model_path,
        device_specs=device_specs,
        weight_path=weight_path or [],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Category A: Single-Encoding Repos — Encoding Inference
# ---------------------------------------------------------------------------


class TestSingleEncodingInference:
    """Tests for encoding inference from repos with a single encoding."""

    def test_infer_encoding_single_bf16_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "BF16"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "bfloat16"

    def test_infer_encoding_single_f32_on_gpu_casts_to_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "bfloat16"

    def test_infer_encoding_single_f32_on_cpu_stays_f32(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "float32"

    def test_infer_encoding_single_fp8_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F8_E4M3"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "float8_e4m3fn"

    def test_infer_encoding_single_fp4_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "U8"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "float4_e2m1fnx2"

    def test_infer_encoding_gguf_q4_k_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "q4_k"


# ---------------------------------------------------------------------------
# Category B: Mixed-Encoding Safetensors — Core Stress Tests
# ---------------------------------------------------------------------------


class TestMixedEncodingInference:
    """Tests for encoding inference from repos with mixed-encoding safetensors."""

    def test_mixed_fp4_fp8_bf16_selects_fp4(self) -> None:
        """FP4 should win when all three quantized types are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "quant_weight": "U8",
                        "scale": "F8_E4M3",
                        "norm": "BF16",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "float4_e2m1fnx2"

    def test_mixed_bf16_and_f32_selects_bf16_on_gpu(self) -> None:
        """On GPU, bf16 should be preferred over f32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "bfloat16"

    def test_mixed_bf16_and_f32_ambiguous_on_cpu(self) -> None:
        """On CPU, multiple non-quantized encodings are ambiguous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding is None

    def test_sharded_fp8_with_bf16_first_shard(self) -> None:
        """FP8 must be detected even when first shard is BF16-only norms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    # Shard 3: norms/embeddings only (BF16) - may sort first
                    "model-00003-of-00003.safetensors": {
                        "model.norm.weight": "BF16",
                    },
                    # Shard 1: FP8 quantized weights
                    "model-00001-of-00003.safetensors": {
                        "model.layers.0.self_attn.q_proj.weight": "F8_E4M3",
                        "model.layers.0.input_layernorm.weight": "BF16",
                    },
                    # Shard 2: more FP8 weights
                    "model-00002-of-00003.safetensors": {
                        "model.layers.1.self_attn.q_proj.weight": "F8_E4M3",
                    },
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "float8_e4m3fn"

    def test_gptq_detected_from_local_config_json(self) -> None:
        """gptq should be detected from config.json for local repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "U8"}})
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "gptq"}}, f)
            repo = HuggingFaceRepo(repo_id=tmpdir)
            assert "gptq" in repo.supported_encodings


# ---------------------------------------------------------------------------
# Category C: Weight Path Resolution
# ---------------------------------------------------------------------------


class TestWeightPathResolution:
    """Tests for weight file discovery during resolve()."""

    def test_resolve_weight_path_sharded_safetensors(self) -> None:
        """Sharded safetensors should all be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            paths = sorted(str(p) for p in config.weight_path)
            assert paths == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]

    def test_prefers_safetensors_over_gguf(self) -> None:
        """When both formats exist, safetensors should be preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
                gguf_files=["model-Q4_K_M.gguf"],
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model.safetensors"]

    def test_falls_back_to_gguf_when_only_format(self) -> None:
        """GGUF files should be discovered when no safetensors exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model-Q4_K_M.gguf"]

    def test_dtype_cast_fallback_finds_f32_files_for_bf16(self) -> None:
        """On GPU, encoding casts to bf16 but weight files match the original f32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding == "bfloat16"
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model.safetensors"]

    def test_explicit_weight_path_skips_discovery(self) -> None:
        """Explicit weight_path should not be overwritten by discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {"w": "BF16"},
                    "other.safetensors": {"w": "BF16"},
                },
            )
            explicit = [Path("model.safetensors")]
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks(weight_path_return=(explicit, None)):
                config.resolve()
            assert config.weight_path == [Path("model.safetensors")]

    def test_consolidated_safetensors_excluded(self) -> None:
        """consolidated.safetensors should be excluded when sharded files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "consolidated.safetensors": {"w": "BF16"},
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            paths = sorted(str(p) for p in config.weight_path)
            assert "consolidated.safetensors" not in paths
            assert len(paths) == 2


# ---------------------------------------------------------------------------
# Category D: Encoding from Explicit Weight Path
# ---------------------------------------------------------------------------


class TestEncodingFromExplicitWeightPath:
    """Tests for encoding inference when weight_path is explicitly provided."""

    def test_encoding_from_gguf_filename_in_weight_path(self) -> None:
        """GGUF encoding should be inferred from the filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            explicit = [Path("model-Q4_K_M.gguf")]
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            with _resolve_mocks(weight_path_return=(explicit, None)):
                config.resolve()
            assert config.quantization_encoding == "q4_k"

    def test_encoding_from_remote_safetensors_via_repo(self) -> None:
        """For remote safetensors, encoding is inferred from the repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "BF16"}})
            # Simulate remote: weight_path points to a non-local file, so
            # _try_infer_encoding falls through to encoding_for_file.
            explicit = [Path("model.safetensors")]
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks(weight_path_return=(explicit, None)):
                config.resolve()
            assert config.quantization_encoding == "bfloat16"

    def test_encoding_from_local_safetensors_with_name_hint(self) -> None:
        """Encoding should be parsed from filename when a hint is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, "model-bf16.safetensors")
            _write_fake_safetensors(fp, dtype="BF16")
            explicit = [Path(fp)]
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks(weight_path_return=(explicit, None)):
                config.resolve()
            assert config.quantization_encoding == "bfloat16"


# ---------------------------------------------------------------------------
# Category E: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests that encoding inference and weight path resolution are deterministic.

    These run resolve() multiple times with fresh MAXModelConfig instances to
    exercise the set→list conversion in supported_encodings.
    """

    def test_deterministic_mixed_fp4_bf16(self) -> None:
        """FP4+BF16+F32 mixed file must always resolve to fp4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "data_weight": "U8",
                        "norm_weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            results = []
            for _ in range(50):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                with _resolve_mocks():
                    config.resolve()
                results.append(config.quantization_encoding)
            assert all(r == "float4_e2m1fnx2" for r in results), (
                f"Non-deterministic results: {set(results)}"
            )

    def test_deterministic_mixed_fp8_bf16_f32(self) -> None:
        """FP8+BF16+F32 mixed file must always resolve to fp8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "data_weight": "F8_E4M3",
                        "norm_weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            results = []
            for _ in range(50):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                with _resolve_mocks():
                    config.resolve()
                results.append(config.quantization_encoding)
            assert all(r == "float8_e4m3fn" for r in results), (
                f"Non-deterministic results: {set(results)}"
            )

    def test_deterministic_weight_path_sharded(self) -> None:
        """Sharded weight path resolution must be stable across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model-00001-of-00004.safetensors": {"w": "BF16"},
                    "model-00002-of-00004.safetensors": {"w": "BF16"},
                    "model-00003-of-00004.safetensors": {"w": "BF16"},
                    "model-00004-of-00004.safetensors": {"w": "BF16"},
                },
            )
            results = []
            for _ in range(10):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                with _resolve_mocks():
                    config.resolve()
                results.append(sorted(str(p) for p in config.weight_path))
            assert all(r == results[0] for r in results), (
                f"Non-deterministic weight paths: {results}"
            )


# ---------------------------------------------------------------------------
# Category F: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases in resolve()."""

    def test_empty_model_path_and_weight_path_raises(self) -> None:
        """Empty model_path with no weight_path should raise ValueError."""
        config = _make_config("", device_specs=[CPU_DEVICE_SPEC])
        with (
            _resolve_mocks(),
            pytest.raises(ValueError, match="model must be provided"),
        ):
            config.resolve()

    def test_corrupt_safetensors_suppresses_exception(self) -> None:
        """Corrupt safetensors should not crash resolve()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_path = os.path.join(tmpdir, "model.safetensors")
            with open(corrupt_path, "wb") as f:
                # Write truncated header (only 4 bytes instead of 8).
                f.write(b"\x00\x00\x00\x00")
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.quantization_encoding is None

    def test_no_weight_files_in_repo(self) -> None:
        """Empty repo should resolve without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            with _resolve_mocks():
                config.resolve()
            assert config.weight_path == []
            assert config.quantization_encoding is None

    def test_encoding_for_file_honors_preferred_encoding(self) -> None:
        """encoding_for_file should return preferred_encoding when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "U8",
                        "norm": "BF16",
                    }
                },
            )
            repo = HuggingFaceRepo(repo_id=tmpdir)
            result = repo.encoding_for_file(
                "model.safetensors", preferred_encoding="bfloat16"
            )
            assert result == "bfloat16"

    def test_encoding_for_file_without_preferred_uses_priority(self) -> None:
        """Without preferred_encoding, priority should pick fp4 over bf16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "U8",
                        "norm": "BF16",
                    }
                },
            )
            repo = HuggingFaceRepo(repo_id=tmpdir)
            result = repo.encoding_for_file("model.safetensors")
            assert result == "float4_e2m1fnx2"
