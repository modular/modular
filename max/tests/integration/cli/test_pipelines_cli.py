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

import io
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from max import profiler
from max._entrypoints import pipelines
from max._entrypoints.cli import generate as cli_generate
from max._entrypoints.cli.config import pipeline_config_options
from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.lib import PixelGenerationPipeline, PixelGenerationTokenizer
from max.pipelines.lib.generate_images import (
    GenerateImagesResult,
    _ImageRequests,
)
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.request.provider_options import ImageProviderOptions
from PIL import Image
from test_common.fake_weights import write_fake_safetensors

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


def _diffusers_dir(root: Path, class_name: str = "Flux2KleinPipeline") -> str:
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": class_name,
                "transformer": ["diffusers", "Transformer"],
            }
        )
    )
    (root / "transformer").mkdir()
    write_fake_safetensors(
        root / "transformer" / "diffusion_pytorch_model.safetensors"
    )
    return str(root)


def _record_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(cli_generate, "generate_images_for_pipeline", record)
    monkeypatch.setattr(cli_generate, "generate_text_for_pipeline", record)
    return calls


def test_pipelines_cli__generate_routes_image_models_with_their_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_routes(monkeypatch)
    model_path = _diffusers_dir(tmp_path / "model")
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                model_path,
                "--devices=cpu",
                "--prompt",
                "a red fox",
                "--negative-prompt",
                "blurry",
                "--height=256",
                "--width=512",
                "--num-inference-steps=4",
                "--guidance-scale=2.5",
                "--true-cfg-scale=4.0",
                "--strength=0.5",
                "--num-images-per-prompt=2",
                "--seed=7",
                "--image_url=https://example.com/fox.png",
                f"--output={tmp_path / 'out'}",
            ]
        )
    [(args, kwargs)] = calls
    pipeline_args, prompt, image_urls, output = args
    assert isinstance(pipeline_args, PipelineArgs)
    assert (prompt, image_urls, output) == (
        "a red fox",
        ("https://example.com/fox.png",),
        tmp_path / "out",
    )
    assert kwargs == {
        "negative_prompt": "blurry",
        "height": 256,
        "width": 512,
        "num_inference_steps": 4,
        "guidance_scale": 2.5,
        "true_cfg_scale": 4.0,
        "strength": 0.5,
        "num_images_per_prompt": 2,
        "seed": 7,
    }


def test_pipelines_cli__generate_seeds_images_with_zero_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_routes(monkeypatch)
    model_path = _diffusers_dir(tmp_path / "model")
    with pytest.raises(SystemExit):
        pipelines.main(["generate", "--model-path", model_path])
    [(_, kwargs)] = calls
    assert kwargs["seed"] == 0


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (["--profile"], "--profile applies"),
        (
            ["--max-new-tokens=8", "--top-k=1", "--temperature=0.5"],
            "--max-new-tokens, --top-k, --temperature apply",
        ),
        (
            ["--num-warmups=1", "--no-detokenize"],
            "--detokenize/--no-detokenize, --num-warmups",
        ),
        (
            ["--max-batch-size=2", "--device-graph-capture"],
            "--device-graph-capture/--no-device-graph-capture, "
            "--max-batch-size apply",
        ),
    ],
)
def test_pipelines_cli__generate_rejects_text_flags_for_image_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    named: str,
) -> None:
    calls = _record_routes(monkeypatch)
    model_path = _diffusers_dir(tmp_path / "model")
    with pytest.raises(ValueError) as error:
        pipelines.main(["generate", "--model-path", model_path, *flags])
    message = str(error.value)
    assert named in message
    assert message.endswith(
        f"to text generation models, and {model_path} is an image "
        "generation model."
    )
    assert calls == []


def test_pipelines_cli__generate_rejects_image_flags_for_text_models(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_routes(monkeypatch)
    with pytest.raises(ValueError) as error:
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--height=256",
                "--num-images-per-prompt=1",
                "--output=fox.png",
            ]
        )
    assert str(error.value) == (
        "--height, --num-images-per-prompt, --output apply to image "
        f"generation models, and {tiny_llama_local_path} is a text generation "
        "model."
    )
    assert calls == []


def test_pipelines_cli__generate_ignores_unset_flags(
    tmp_path: Path,
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only flags the user set count, so each model runs with its defaults."""
    calls = _record_routes(monkeypatch)
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--max-new-tokens=8",
            ]
        )
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                _diffusers_dir(tmp_path / "model"),
                "--height=256",
            ]
        )
    assert len(calls) == 2


def test_pipelines_cli__image_model_detection(
    tmp_path: Path, tiny_llama_local_path: str
) -> None:
    image_args = PipelineArgs.from_flat_kwargs(
        model_path=_diffusers_dir(tmp_path / "model"), device_specs="cpu"
    )
    assert cli_generate.is_image_generation_model(image_args)
    text_args = PipelineArgs.from_flat_kwargs(
        model_path=tiny_llama_local_path, device_specs="cpu"
    )
    assert not cli_generate.is_image_generation_model(text_args)
    explicit = text_args.model_copy(
        update={"task": PipelineTask.PIXEL_GENERATION}
    )
    with pytest.raises(ValueError) as error:
        cli_generate.is_image_generation_model(explicit)
    assert str(error.value) == (
        "--task pixel_generation needs an image generation model, but "
        f"{tiny_llama_local_path} uses LlamaForCausalLM, which is not one."
    )
    assert cli_generate.is_image_generation_model(
        image_args.model_copy(update={"task": PipelineTask.PIXEL_GENERATION})
    )


@pytest.fixture
def image_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[PipelineConfig, PipelineTask]]:
    """Records each build and returns stub image components."""
    builds: list[tuple[PipelineConfig, PipelineTask]] = []

    def retrieve(
        pipeline_config: PipelineConfig, task: PipelineTask
    ) -> tuple[MagicMock, MagicMock]:
        builds.append((pipeline_config, task))
        return (
            MagicMock(spec=PixelGenerationTokenizer),
            MagicMock(spec=PixelGenerationPipeline),
        )

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)
    return builds


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[_ImageRequests]:
    """Records each run's checked requests and returns stub images."""
    prepared: list[_ImageRequests] = []
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]

    def run(
        pipeline: object, tokenizer: object, requests: _ImageRequests
    ) -> GenerateImagesResult:
        prepared.append(requests)
        count = _options(requests).num_images
        return GenerateImagesResult(images=[images[:count]])

    monkeypatch.setattr(cli_generate, "_run_requests", run)
    return prepared


def _options(prepared: _ImageRequests) -> ImageProviderOptions:
    [request] = prepared.requests
    options = request.body.provider_options.image
    assert options is not None
    return options


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 24)).save(buffer, format="PNG")
    return buffer.getvalue()


def _generate_images(
    pipeline_args: PipelineArgs,
    output: Path,
    image_urls: tuple[str, ...] = (),
    *,
    prompt: str = "a red fox",
    num_inference_steps: int | None = 4,
    num_images_per_prompt: int = 1,
    seed: int = 0,
) -> list[Path]:
    return cli_generate.generate_images_for_pipeline(
        pipeline_args,
        prompt,
        image_urls,
        output,
        negative_prompt=None,
        height=None,
        width=None,
        num_inference_steps=num_inference_steps,
        guidance_scale=None,
        true_cfg_scale=None,
        strength=None,
        num_images_per_prompt=num_images_per_prompt,
        seed=seed,
    )


def _image_args(
    tmp_path: Path, class_name: str = "Flux2KleinPipeline"
) -> PipelineArgs:
    return PipelineArgs.from_flat_kwargs(
        model_path=_diffusers_dir(tmp_path / "model", class_name),
        device_specs="cpu",
    )


def test_pipelines_cli__generate_saves_each_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_builds: list[tuple[PipelineConfig, PipelineTask]],
    runs: list[_ImageRequests],
) -> None:
    # The image path keeps serve's AMD config, without the text defaults.
    monkeypatch.delenv("MODULAR_DISABLE_VENDOR_FALLBACK", raising=False)
    monkeypatch.setattr(cli_generate, "accelerator_api", lambda: "hip")
    pipeline_args = _image_args(tmp_path)
    [path] = _generate_images(pipeline_args, tmp_path / "out" / "nested")
    assert path == tmp_path / "out" / "nested" / "image.png"
    with Image.open(path) as saved:
        assert saved.format == "PNG"

    paths = _generate_images(
        pipeline_args, tmp_path / "fox.jpg", num_images_per_prompt=2
    )
    assert paths == [tmp_path / "fox-0.jpg", tmp_path / "fox-1.jpg"]
    for path in paths:
        with Image.open(path) as saved:
            assert saved.format == "JPEG"
    assert [prepared.input_images for prepared in runs] == [[None], [None]]
    assert runs[0].requests[0].body.input == "a red fox"
    for config, task in image_builds:
        assert task == PipelineTask.PIXEL_GENERATION
        assert config.runtime.disable_vendor_blas_fallback is False


def test_pipelines_cli__generate_downloads_the_input_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_builds: list[tuple[PipelineConfig, PipelineTask]],
    runs: list[_ImageRequests],
) -> None:
    urls: list[str] = []

    def get(url: str) -> MagicMock:
        urls.append(url)
        # The download comes before the build.
        assert image_builds == []
        return MagicMock(content=_png())

    monkeypatch.setattr(cli_generate.requests, "get", get)
    _generate_images(
        _image_args(tmp_path),
        tmp_path / "fox.png",
        ("https://example.com/a.png",),
    )
    assert urls == ["https://example.com/a.png"]
    [prepared] = runs
    [image] = prepared.input_images
    assert image is not None
    assert image.size == (16, 24)


@pytest.mark.parametrize(
    ("image_urls", "output", "kwargs", "message"),
    [
        (
            ("https://example.com/a.png", "https://example.com/b.png"),
            "fox.png",
            {},
            "takes one --image_url",
        ),
        ((), "fox.xyz", {}, "names no image format PIL can write"),
        # PIL reads PSD files but cannot write them.
        ((), "fox.psd", {}, "names no image format PIL can write"),
        ((), "missing/fox.png", {}, "which does not exist"),
        ((), "fox.png", {"prompt": ""}, "Prompt 0 is empty."),
        ((), "fox.png", {"seed": 2**32}, r"seed must be in \[0, 2\*\*32\)"),
        (
            (),
            "fox.png",
            {"num_inference_steps": 0},
            "num_inference_steps must be at least 1",
        ),
    ],
)
def test_pipelines_cli__generate_checks_arguments_before_loading(
    tmp_path: Path,
    image_builds: list[tuple[PipelineConfig, PipelineTask]],
    image_urls: tuple[str, ...],
    output: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    untyped: Callable[..., object] = _generate_images
    with pytest.raises(ValueError, match=message):
        untyped(_image_args(tmp_path), tmp_path / output, image_urls, **kwargs)
    assert image_builds == []


def test_pipelines_cli__generate_checks_the_input_image_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_builds: list[tuple[PipelineConfig, PipelineTask]],
) -> None:
    monkeypatch.setattr(
        cli_generate.requests, "get", lambda url: MagicMock(content=_png())
    )
    with pytest.raises(ValueError, match="takes text only"):
        _generate_images(
            _image_args(tmp_path, "Ideogram4Pipeline"),
            tmp_path / "fox.png",
            ("https://example.com/a.png",),
        )
    assert image_builds == []


def test_pipelines_cli__generate_rejects_video_models_before_loading(
    tmp_path: Path, image_builds: list[tuple[PipelineConfig, PipelineTask]]
) -> None:
    with pytest.raises(ValueError) as error:
        _generate_images(
            _image_args(tmp_path, "WanPipeline"), tmp_path / "fox.png"
        )
    assert str(error.value).startswith(
        "max generate does not support video generation."
    )
    assert image_builds == []


@pytest.mark.parametrize(
    "flags",
    [["--num-inference-steps=0"], ["--num-images-per-prompt=0"]],
)
def test_pipelines_cli__generate_rejects_zero_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    calls = _record_routes(monkeypatch)
    with pytest.raises(SystemExit) as error:
        pipelines.main(
            ["generate", "--model-path", _diffusers_dir(tmp_path / "m"), *flags]
        )
    assert error.value.code == 2
    assert calls == []


def test_pipelines_cli__profile_on_a_text_model_passes_the_flag_check(
    tiny_llama_local_path: str,
    custom_architecture_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReExec(Exception):
        pass

    def reexec(*args: object, **kwargs: object) -> None:
        raise _ReExec

    monkeypatch.setattr(profiler, "maybe_reexec_under_nsys", reexec)
    with pytest.raises(_ReExec):
        pipelines.main(
            [
                "generate",
                "--model-path",
                tiny_llama_local_path,
                f"--custom-architectures={custom_architecture_path}",
                "--devices=cpu",
                "--profile",
                "--max-new-tokens=4",
            ]
        )


def test_pipelines_cli__every_generate_flag_has_a_kind() -> None:
    command = pipelines.main.commands["generate"]
    params = {
        param.name for param in command.get_params(click.Context(command))
    }
    config_params = {
        param.name
        for param in getattr(
            pipeline_config_options(lambda **kwargs: None),
            "__click_params__",
            [],
        )
    }
    text = set(cli_generate._TEXT_ONLY_FLAGS)
    image = set(cli_generate._IMAGE_ONLY_FLAGS)
    shared = {"prompt", "image_url", "seed", "help"}
    assert not text & image
    assert not (text | image) & shared
    assert params - config_params == (text | image | shared) - config_params
    # The config flags rejected on image models are real flags too.
    assert (text & config_params) <= params
