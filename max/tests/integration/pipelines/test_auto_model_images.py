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
"""AutoModel and AutoPipeline on an image model, without loading one.

A local diffusers directory with a fake transformer resolves to a real
image architecture, and a stub of the registry's build stands in for the
pipeline, so the tests run offline on CPU.
"""

from __future__ import annotations

import gc
import inspect
import json
import weakref
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typing_extensions
from max.pipelines import (
    PIPELINE_REGISTRY,
    AutoModel,
    AutoPipeline,
    PipelineConfig,
)
from max.pipelines.lib import (
    PixelGenerationPipeline,
    PixelGenerationTokenizer,
    auto_model,
    auto_pipeline,
)
from max.pipelines.lib.generate_images import GenerateImagesResult
from max.pipelines.modeling.types import PipelineTask
from PIL import Image
from test_common.fake_weights import write_fake_safetensors


def _diffusers_dir(root: Path, class_name: str) -> str:
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


@pytest.fixture
def image_model_path(tmp_path: Path) -> str:
    return _diffusers_dir(tmp_path, "Flux2KleinPipeline")


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[PipelineTask]:
    """Records each build's task and returns stub image components."""
    tasks: list[PipelineTask] = []

    def retrieve(
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
    ) -> tuple[MagicMock, MagicMock]:
        tasks.append(task)
        return (
            MagicMock(spec=PixelGenerationTokenizer),
            MagicMock(spec=PixelGenerationPipeline),
        )

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)
    return tasks


def _load(model_path: str) -> AutoModel:
    return AutoModel(model_path, devices="cpu")


def test_builds_the_image_pipeline_with_harmless_defaults(
    image_model_path: str, builds: list[PipelineTask]
) -> None:
    with _load(image_model_path) as model:
        config = model.pipeline_config
        assert "main" not in config.models
        assert config.runtime.max_batch_size == 1
        assert config.runtime.enable_overlap_scheduler is False
        assert config.runtime.device_graph_capture is False
    assert builds == [PipelineTask.PIXEL_GENERATION]


def test_generate_passes_every_image_argument(
    image_model_path: str,
    builds: list[PipelineTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(auto_model, "generate_images", record)
    with _load(image_model_path) as model:
        model.generate(
            prompt=["a", "b"],
            negative_prompt="c",
            height=256,
            width=512,
            num_inference_steps=4,
            guidance_scale=2.0,
            true_cfg_scale=1.5,
            num_images_per_prompt=2,
            seed=9,
        )
        pipeline = model._image_pipeline
        tokenizer = model._image_tokenizer
    assert calls == [
        (
            (pipeline, tokenizer, ["a", "b"]),
            {
                "negative_prompt": "c",
                "height": 256,
                "width": 512,
                "num_inference_steps": 4,
                "guidance_scale": 2.0,
                "true_cfg_scale": 1.5,
                "num_images_per_prompt": 2,
                "seed": 9,
                "caller": "AutoModel",
            },
        )
    ]


def test_text_arguments_are_rejected_on_an_image_model(
    image_model_path: str, builds: list[PipelineTask]
) -> None:
    loaded = f"AutoModel loaded {image_model_path}, an image generation model."
    with _load(image_model_path) as model:
        # No overload takes the mixed calls, so they go through an untyped
        # name.
        untyped: Callable[..., object] = model.generate
        with pytest.raises(ValueError) as error:
            model.generate([[1, 2]], max_new_tokens=1)
        assert str(error.value) == (
            "input_ids, max_new_tokens are for text generation models, and "
            f"{loaded} Pass the text as prompt= instead."
        )
        with pytest.raises(ValueError) as error:
            # ignore_eos=False is its default, so it does not count.
            untyped(prompt="a", temperature=0.5, ignore_eos=False, top_k=2)
        assert str(error.value) == (
            f"temperature, top_k are for text generation models, and {loaded}"
        )
        with pytest.raises(TypeError, match="needs prompt="):
            untyped(height=256)
        with pytest.raises(ValueError) as error:
            model.generate_iter([[1, 2]], max_new_tokens=1)
        assert str(error.value) == (
            f"{loaded} Use AutoModel.generate(prompt=...) for it."
        )
        with pytest.raises(ValueError):
            _ = model.pipeline


def test_rejects_graph_capture_before_building(
    image_model_path: str, builds: list[PipelineTask]
) -> None:
    with pytest.raises(ValueError) as error:
        AutoModel(image_model_path, devices="cpu", device_graph_capture=True)
    assert str(error.value) == (
        f"device_graph_capture=True does not apply to {image_model_path}, an "
        "image generation model, since its pipeline runs the whole denoising "
        "loop in one call."
    )
    assert builds == []


def test_rejects_a_batch_size_before_building(
    image_model_path: str, builds: list[PipelineTask]
) -> None:
    with pytest.raises(ValueError) as error:
        AutoModel(image_model_path, devices="cpu", max_batch_size=2)
    assert str(error.value) == (
        f"max_batch_size must be 1 for {image_model_path}, an image "
        "generation model, since its pipeline runs one prompt per call. "
        "Got 2."
    )
    assert builds == []


def test_rejects_video_models_before_building(
    tmp_path: Path, builds: list[PipelineTask]
) -> None:
    model_path = _diffusers_dir(tmp_path, "WanPipeline")
    with pytest.raises(ValueError) as error:
        _load(model_path)
    assert str(error.value).startswith(
        "AutoModel does not support video generation. The model uses "
        "WanContext, which holds video frames, and AutoModel returns images."
    )
    assert builds == []


def test_video_error_wins_over_text_only_settings(
    tmp_path: Path, builds: list[PipelineTask]
) -> None:
    model_path = _diffusers_dir(tmp_path, "WanPipeline")
    with pytest.raises(ValueError) as error:
        AutoModel(model_path, devices="cpu", max_batch_size=2)
    assert str(error.value).startswith(
        "AutoModel does not support video generation."
    )
    assert builds == []


def test_close_releases_without_a_kv_cache(
    image_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_refs: list[weakref.ref[MagicMock]] = []

    def retrieve(
        pipeline_config: PipelineConfig, task: PipelineTask
    ) -> tuple[MagicMock, MagicMock]:
        # The spec has no kv_manager, so close() raises if it reads one.
        pipeline = MagicMock(spec=PixelGenerationPipeline)
        pipeline_refs.append(weakref.ref(pipeline))
        return MagicMock(spec=PixelGenerationTokenizer), pipeline

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)
    model = _load(image_model_path)
    model.close()
    model.close()
    gc.collect()
    [pipeline_ref] = pipeline_refs
    assert pipeline_ref() is None
    with pytest.raises(RuntimeError, match="closed"):
        model.generate(prompt="a")


@pytest.fixture
def image_results(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Records each generate_images call and returns two images per prompt."""
    calls: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> GenerateImagesResult:
        prompts = args[2]
        assert isinstance(prompts, list)
        calls.append({"prompt": prompts, **kwargs})
        return GenerateImagesResult(
            images=[
                [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
                for _ in prompts
            ]
        )

    monkeypatch.setattr(auto_model, "generate_images", record)
    return calls


def test_auto_pipeline_generates_images(
    image_model_path: str,
    builds: list[PipelineTask],
    image_results: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizers: list[object] = []
    monkeypatch.setattr(
        auto_pipeline,
        "AutoTokenizer",
        lambda *args, **kwargs: tokenizers.append(args),
    )
    with AutoPipeline(image_model_path, devices="cpu") as pipe:
        one = pipe("a fox", height=256, width=512, seed=3)
        many = pipe(["a", "b"], negative_prompt="c", num_images_per_prompt=2)
        with pytest.raises(ValueError, match="has no text tokenizer"):
            _ = pipe.tokenizer
    assert tokenizers == []
    assert len(one) == 2
    assert [len(row) for row in many] == [2, 2]
    assert image_results[0]["prompt"] == ["a fox"]
    assert image_results[0]["height"] == 256
    assert image_results[0]["seed"] == 3
    assert image_results[1]["prompt"] == ["a", "b"]
    assert image_results[1]["negative_prompt"] == "c"
    assert image_results[1]["caller"] == "AutoModel"


def test_auto_pipeline_rejects_text_arguments_on_an_image_model(
    image_model_path: str,
    builds: list[PipelineTask],
    image_results: list[dict[str, object]],
) -> None:
    with AutoPipeline(image_model_path, devices="cpu") as pipe:
        untyped: Callable[..., object] = pipe
        with pytest.raises(ValueError) as error:
            untyped("a fox", max_new_tokens=8, temperature=0.5)
        assert str(error.value) == (
            "max_new_tokens, temperature are for text generation models, and "
            f"AutoModel loaded {image_model_path}, an image generation model."
        )
        with pytest.raises(ValueError, match="not conversations"):
            untyped([{"role": "user", "content": "a fox"}])
    assert image_results == []


def test_image_models_keep_the_vendor_blas_fallback_on_amd(
    image_model_path: str,
    builds: list[PipelineTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODULAR_DISABLE_VENDOR_FALLBACK", raising=False)
    monkeypatch.setattr(auto_model, "accelerator_api", lambda: "hip")
    with _load(image_model_path) as model:
        assert model.pipeline_config.runtime.disable_vendor_blas_fallback is (
            False
        )


# A value other than the default for each text-only argument.
_TEXT_ONLY_VALUES: dict[str, object] = {
    "input_ids": [[1, 2]],
    "attention_mask": [[1, 1]],
    "messages": [[{"role": "user", "content": "hi"}]],
    "images": [[b"image"]],
    "chat_template_kwargs": {"enable_thinking": False},
    "max_new_tokens": 4,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 2,
    "top_p": 0.9,
    "min_p": 0.1,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "repetition_penalty": 1.1,
    "ignore_eos": True,
    "stop_token_ids": [3],
    "output_logits": True,
    "output_logprobs": 1,
}


def _overload_parameters() -> tuple[set[str], set[str]]:
    """Returns the text and image overloads' parameter names."""
    text: set[str] = set()
    image: set[str] = set()
    for overload in typing_extensions.get_overloads(AutoModel.generate):
        names = set(inspect.signature(overload).parameters) - {"self"}
        (image if "prompt" in names else text).update(names)
    return text, image


def test_every_text_argument_is_rejected_on_an_image_model(
    image_model_path: str, builds: list[PipelineTask]
) -> None:
    text, image = _overload_parameters()
    implementation = set(inspect.signature(AutoModel.generate).parameters)
    assert text | image == implementation - {"self"}
    with _load(image_model_path) as model:
        untyped: Callable[..., object] = model.generate
        for name in sorted(text - image):
            with pytest.raises(ValueError) as error:
                untyped(prompt="a", **{name: _TEXT_ONLY_VALUES[name]})
            assert str(error.value).startswith(
                f"{name} is for text generation models"
            )
