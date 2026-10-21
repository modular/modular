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

import logging
import os

import pytest
from max._entrypoints import pipelines
from max._entrypoints.cli import generate as cli_generate
from max.pipelines import PipelineArgs

logger = logging.getLogger("max.pipelines")


@pytest.fixture
def tiny_llama_local_path() -> str:
    """Return the local path to the tiny llama model.
    Note: We are not actually running tiny llama because the architecture
    is overwritten by the custom testing architecture.
    """
    model_path = os.getenv("TEST_LLAMA_REPO")
    if model_path is None:
        raise ValueError("TEST_LLAMA_REPO environment variable is not set")
    return model_path


@pytest.fixture
def custom_architecture_path() -> str:
    path = os.getenv("PIPELINES_CUSTOM_ARCHITECTURE")
    if path is None:
        raise ValueError(
            "PIPELINES_CUSTOM_ARCHITECTURE environment variable is not set"
        )
    return path


def test_pipelines_cli__smollm_float32(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--prompt",
                "Why is the sky blue?",
                "--quantization-encoding=float32",
                "--top-k=1",
                "--max-new-tokens=10",
                "--max-length=128",
            ]
        )
    captured = capsys.readouterr()
    # Output from dummy model is based on the tokenized input, which results in
    # the character chr(0).
    assert chr(0) in captured.out


def test_pipelines_cli__invalid_quantization_encoding(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
) -> None:
    with pytest.raises(Exception, match=r".*'q4_k' not supported.*"):
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--prompt",
                "Why is the sky blue?",
                "--quantization-encoding=q4_k",
                "--top-k=1",
                "--max-new-tokens=10",
                "--max-length=128",
            ]
        )


def test_pipelines_cli__model_and_model_path_conflict(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
) -> None:
    """Test that specifying both --model and --model-path raises an error."""

    with pytest.raises(
        ValueError, match="model_path and model cannot both be specified"
    ):
        pipelines.main(
            [
                "generate",
                "--model",
                tiny_llama_local_path,
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--prompt",
                "Why is the sky blue?",
                "--quantization-encoding=float32",
                "--top-k=1",
                "--max-new-tokens=10",
                "--max-length=128",
            ]
        )


def test_pipelines_cli__set_kv_cache_dtype(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that there is no issue when the KV cache datatype is overridden with parameter `--kv-cache-format`."""

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--prompt",
                "Why is the sky blue?",
                "--quantization-encoding=float32",
                "--top-k=1",
                "--max-new-tokens=1",
                "--max-length=128",
                "--kv-cache-format=float8_e4m3fn",
            ]
        )
    captured = capsys.readouterr()
    # Two 128-token pages (incl null page)
    # 2x (4 KiB of fp8 K/V data + 2 KiB of float32 scales) = 12 KiB
    assert "cache_memory" in captured.err and ": 12.00 KiB" in captured.err

    # Bfloat16 stores 2 bytes per K/V element and needs no scales.
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--prompt",
                "Why is the sky blue?",
                "--quantization-encoding=float32",
                "--top-k=1",
                "--max-new-tokens=1",
                "--max-length=128",
                "--kv-cache-format=bfloat16",
            ]
        )
    captured = capsys.readouterr()
    # Two 128-token pages (incl null page)
    # 2x (8 KiB of float32 values) = 16 KiB
    assert "cache_memory" in captured.err and ": 16.00 KiB" in captured.err


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--no-disable-vendor-blas-fallback"], False)],
)
def test_pipelines_cli__generate_turns_the_vendor_blas_fallback_off_on_amd(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    expected: bool,
) -> None:
    monkeypatch.delenv("MODULAR_DISABLE_VENDOR_FALLBACK", raising=False)
    monkeypatch.setattr(cli_generate, "accelerator_api", lambda: "hip")
    seen: list[PipelineArgs] = []

    def record(pipeline_args: PipelineArgs, **kwargs: object) -> None:
        seen.append(pipeline_args)

    monkeypatch.setattr(cli_generate, "generate_text_for_pipeline", record)
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--quantization-encoding=float32",
                "--max-length=128",
                *flags,
            ]
        )
    [pipeline_args] = seen
    runtime = cli_generate._with_debug_defaults(pipeline_args).runtime
    assert runtime.disable_vendor_blas_fallback is expected


class _StopAtPipelineArgs(Exception):
    pass


@pytest.mark.parametrize("command", ["serve", "generate"])
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], None),
        (["--no-enable-overlap-scheduler"], False),
        (["--enable-overlap-scheduler"], True),
    ],
)
def test_pipelines_cli__overlap_scheduler_unset_unless_passed(
    tiny_llama_local_path: str,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    flags: list[str],
    expected: bool | None,
) -> None:
    """Without the flag the field stays unset, so the architecture decides."""
    seen: list[PipelineArgs] = []
    from_flat_kwargs = PipelineArgs.from_flat_kwargs

    def record(**kwargs: object) -> PipelineArgs:
        seen.append(from_flat_kwargs(**kwargs))
        raise _StopAtPipelineArgs

    monkeypatch.setattr(PipelineArgs, "from_flat_kwargs", record)
    with pytest.raises(_StopAtPipelineArgs):
        pipelines.main([command, "--model-path", tiny_llama_local_path, *flags])
    [pipeline_args] = seen
    assert pipeline_args.runtime.enable_overlap_scheduler is expected
