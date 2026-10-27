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
"""Tests for AutoModel against a real small CPU model.

Tests that only check the resolved config or the calls AutoModel makes stub
the registry's build and run on accelerator configs without a GPU.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import weakref
from collections.abc import Callable, Mapping
from typing import get_origin
from unittest.mock import MagicMock

import numpy as np
import pytest
import typing_extensions
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    AutoModel,
    AutoTokenizer,
    PipelineArgs,
    PipelineConfig,
)
from max.pipelines.architectures.llama3_modulev3.model import (
    Llama3Model as ModuleV3Llama3Model,
)
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextGenerationPipeline,
    _hf_config,
    auto_model,
    generate_local_model_path,
)
from max.pipelines.lib.registry import get_pipeline_for_task
from max.pipelines.modeling.types import PipelineTask

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"
EMBEDDINGS_REPO_ID = "sentence-transformers/all-mpnet-base-v2"
NUM_NEW_TOKENS = 8
PROMPTS = [[1, 2, 3, 4], [5, 6, 7]]
MASKS = [[1, 1, 1, 1], [1, 1, 1]]
TEXT = "The capital of France is"


def _local_or_repo(repo_id: str) -> str:
    # A cached snapshot skips the Hub probe, which CI runs without network.
    try:
        return generate_local_model_path(repo_id)
    except FileNotFoundError:
        return repo_id


@pytest.fixture(scope="module")
def model_path() -> str:
    return _local_or_repo(REPO_ID)


def _load_cpu(
    model_path: str, max_batch_size: int = 1, **overrides: object
) -> AutoModel:
    return AutoModel(
        model_path,
        max_batch_size=max_batch_size,
        device_graph_capture=False,
        prefer_module_v3=False,
        devices="cpu",
        quantization_encoding="float32",
        max_length=512,
        **overrides,
    )


def _load_accelerator(
    model_path: str, device_graph_capture: bool = False, **overrides: object
) -> AutoModel:
    return AutoModel(
        model_path,
        max_batch_size=1,
        device_graph_capture=device_graph_capture,
        prefer_module_v3=False,
        devices="gpu",
        **overrides,
    )


def _greedy(model: AutoModel) -> list[list[int]]:
    return model.generate(
        PROMPTS,
        max_new_tokens=NUM_NEW_TOKENS,
        temperature=0.0,
        top_k=1,
        ignore_eos=True,
    ).sequences


def test_generate_then_reload_matches(model_path: str) -> None:
    with _load_cpu(model_path, max_batch_size=len(PROMPTS)) as first:
        sequences = _greedy(first)
        pipeline_ref = weakref.ref(first.pipeline)
    assert pipeline_ref() is None
    assert all(sequences)

    with _load_cpu(model_path, max_batch_size=len(PROMPTS)) as second:
        assert _greedy(second) == sequences


def test_generate_iter_steps_synchronously_like_generate(
    model_path: str,
) -> None:
    with _load_cpu(model_path, max_batch_size=len(PROMPTS)) as model:
        expected = _greedy(model)
        with model.generate_iter(
            PROMPTS,
            max_new_tokens=NUM_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        ) as batch:
            assert batch.synchronous is True
            steps = list(batch)
            assert len(steps) == NUM_NEW_TOKENS
            for step in steps:
                assert sorted(step) == list(range(len(PROMPTS)))
        pipeline_ref = weakref.ref(model.pipeline)
    # The closed batch the caller still holds must not keep the pipeline.
    assert pipeline_ref() is None
    assert batch.sequences == expected


def test_generate_iter_runs_messages_like_generate(model_path: str) -> None:
    messages: list[list[Mapping[str, object]]] = [
        [{"role": "user", "content": "What is 2 + 2?"}]
    ]
    with _load_cpu(model_path) as model:
        expected = model.generate(
            messages=messages, max_new_tokens=NUM_NEW_TOKENS, top_k=1
        )
        with model.generate_iter(
            messages=messages, max_new_tokens=NUM_NEW_TOKENS, top_k=1
        ) as batch:
            list(batch)
    assert batch.sequences == expected.sequences
    assert batch.prompt_lengths == expected.prompt_lengths


def test_chat_template_kwargs_reach_the_template(model_path: str) -> None:
    messages: list[list[Mapping[str, object]]] = [
        [{"role": "user", "content": "What is 2 + 2?"}]
    ]
    with _load_cpu(model_path) as model:
        default = model.generate(messages=messages, max_new_tokens=1)
        without_prompt = model.generate(
            messages=messages,
            chat_template_kwargs={"add_generation_prompt": False},
            max_new_tokens=1,
        )
    [default_length] = default.prompt_lengths
    [length] = without_prompt.prompt_lengths
    # The template's generation prompt opens the assistant turn after the
    # conversation.
    assert length < default_length
    assert without_prompt.sequences[0][:length] == default.sequences[0][:length]


def test_greedy_logits_and_log_probs_match_sequences(model_path: str) -> None:
    num_top_log_probs = 3
    with _load_cpu(model_path, max_batch_size=len(PROMPTS)) as model:
        # The sampler's penalties and min-tokens mask edit the logits that
        # log probs read, so log probs match log_softmax only with both off.
        sampling = model.pipeline_config.sampling
        assert not sampling.enable_penalties
        assert not sampling.enable_min_tokens
        result = model.generate(
            PROMPTS,
            max_new_tokens=NUM_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
            output_logits=True,
            output_logprobs=num_top_log_probs,
        )
    assert result.logits is not None
    assert result.log_probs is not None
    for prompt, sequence, logits, log_probs in zip(
        PROMPTS, result.sequences, result.logits, result.log_probs, strict=True
    ):
        new_tokens = sequence[len(prompt) :]
        assert new_tokens
        assert logits.dtype == np.float32
        assert logits.shape[0] == len(new_tokens)
        assert logits.argmax(axis=1).tolist() == new_tokens
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_softmax = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )
        assert log_probs.sampled_token_ids == new_tokens
        assert len(log_probs.token_log_probabilities) == len(new_tokens)
        assert len(log_probs.top_log_probabilities) == len(new_tokens)
        for t, (token, log_prob, top) in enumerate(
            zip(
                new_tokens,
                log_probs.token_log_probabilities,
                log_probs.top_log_probabilities,
                strict=True,
            )
        ):
            assert len(top) == num_top_log_probs
            assert top[token] == pytest.approx(log_prob)
            assert log_prob == pytest.approx(max(top.values()))
            assert log_prob == pytest.approx(log_softmax[t, token], abs=1e-3)


def test_auto_tokenizer_matches_the_pipeline_tokenizer(model_path: str) -> None:
    tokenizer = AutoTokenizer(model_path)
    with _load_cpu(model_path) as model:
        pipeline_tokenizer = model.pipeline.tokenizer
        expected_ids = [
            int(token)
            for token in asyncio.run(
                pipeline_tokenizer.encode(TEXT, add_special_tokens=True)
            )
        ]
        expected_text = asyncio.run(
            pipeline_tokenizer.decode(np.asarray(expected_ids))
        )

    inputs = tokenizer([TEXT, TEXT])
    assert set(inputs) == {"input_ids", "attention_mask"}
    assert inputs["input_ids"] == [expected_ids] * 2
    assert inputs["attention_mask"] == [[1] * len(expected_ids)] * 2
    assert tokenizer(TEXT)["input_ids"] == expected_ids
    assert tokenizer.decode(expected_ids) == expected_text
    assert tokenizer.batch_decode([expected_ids]) == [expected_text]
    assert TEXT in expected_text


def test_generate_takes_tokenizer_output(model_path: str) -> None:
    tokenizer = AutoTokenizer(model_path)
    inputs = tokenizer([TEXT, "2 + 2 ="])
    with _load_cpu(model_path, max_batch_size=2) as model:
        outputs = model.generate(
            **inputs, max_new_tokens=NUM_NEW_TOKENS, top_k=1
        )

    for sequence, prompt in zip(
        outputs.sequences, inputs["input_ids"], strict=True
    ):
        assert sequence[: len(prompt)] == prompt
        assert len(sequence) > len(prompt)
    [text, _] = tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )
    assert text.startswith(TEXT)


def test_batch_errors_name_auto_model(model_path: str) -> None:
    with _load_cpu(model_path) as model:
        for start in (model.generate, model.generate_iter):
            with pytest.raises(ValueError) as error:
                start(PROMPTS, max_new_tokens=1)
            assert str(error.value).splitlines()[0] == (
                "AutoModel does not support scheduling across batches. 2 "
                "prompts exceed the pipeline's max batch size of 1, and "
                "AutoModel never splits its prompts."
            )


def test_debug_defaults_and_overrides(model_path: str) -> None:
    with _load_cpu(model_path, max_batch_size=2) as model:
        config = model.pipeline_config
        assert config.runtime.enable_overlap_scheduler is False
        assert config.runtime.device_graph_capture is False
        assert config.runtime.enable_chunked_prefill is False
        assert config.model.kv_cache.enable_prefix_caching is False
        assert config.runtime.vision_cache_utilization == 0
        assert config.runtime.max_batch_size == 2
        pipeline = model.pipeline
        assert isinstance(pipeline, TextGenerationPipeline)
        assert pipeline.max_batch_size == 2

    with _load_cpu(model_path, enable_prefix_caching=True) as model:
        assert model.pipeline_config.model.kv_cache.enable_prefix_caching


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": {"max_batch_size": 4}},
        {"kv_cache": {"enable_prefix_caching": True}},
        {"model": {"kv_cache": {"enable_prefix_caching": True}}},
        {"config_file": "recipe.yaml"},
        {"model_override": ['main.kv_cache={"enable_prefix_caching": true}']},
        {"model_override": ('main.kv_cache={"enable_prefix_caching": true}',)},
    ],
)
def test_rejects_overrides_the_defaults_would_clobber(
    model_path: str, overrides: Mapping[str, object]
) -> None:
    with pytest.raises(TypeError, match="flat keyword arguments"):
        _load_cpu(model_path, 1, **overrides)


def test_generate_falls_back_to_server_sampling_defaults(
    model_path: str,
) -> None:
    with _load_cpu(
        model_path, max_batch_size=len(PROMPTS), temperature=0.0, top_k=1
    ) as model:
        assert model.pipeline_config.runtime.temperature == 0.0
        assert model.pipeline_config.runtime.top_k == 1
        sequences = model.generate(
            PROMPTS, max_new_tokens=NUM_NEW_TOKENS, ignore_eos=True
        ).sequences
        assert sequences == _greedy(model)


def test_prompt_over_input_token_budget_runs_whole(model_path: str) -> None:
    # Chunked prefill would size staging for the 64-token budget and
    # overflow on the longer prompt.
    with _load_cpu(model_path, max_batch_input_tokens=64) as model:
        [tokens] = model.generate(
            [list(range(1, 301))], max_new_tokens=1, top_k=1
        ).sequences
    assert len(tokens) == 301


class _StopBeforeBuild(Exception):
    pass


@pytest.fixture
def built_configs(monkeypatch: pytest.MonkeyPatch) -> list[PipelineConfig]:
    """Collects the configs AutoModel resolves, stopping before the build."""
    configs: list[PipelineConfig] = []

    def capture(pipeline_config: PipelineConfig) -> object:
        configs.append(pipeline_config)
        raise _StopBeforeBuild

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", capture)
    return configs


@pytest.mark.parametrize(
    ("device_graph_capture", "overrides", "expected_overlap"),
    [
        (False, {}, False),
        (False, {"enable_overlap_scheduler": True}, True),
        (True, {}, True),
    ],
)
def test_builds_the_config_once(
    model_path: str,
    built_configs: list[PipelineConfig],
    monkeypatch: pytest.MonkeyPatch,
    device_graph_capture: bool,
    overrides: Mapping[str, object],
    expected_overlap: bool,
) -> None:
    from_args = PipelineConfig.from_args
    builds: list[PipelineArgs] = []

    def count(args: PipelineArgs) -> PipelineConfig:
        builds.append(args)
        return from_args(args)

    monkeypatch.setattr(PipelineConfig, "from_args", count)
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(model_path, device_graph_capture, **overrides)
    [config] = built_configs
    assert len(builds) == 1
    assert config.runtime.force is False
    assert config.runtime.enable_overlap_scheduler is expected_overlap


def test_overlap_architecture_gets_a_synchronous_pipeline(
    model_path: str, built_configs: list[PipelineConfig]
) -> None:
    serve_config = PipelineConfig.from_args(
        PipelineArgs(
            model_path=model_path, device_specs=[DeviceSpec.accelerator()]
        )
    )
    assert serve_config.runtime.enable_overlap_scheduler is True
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(model_path)
    [config] = built_configs
    pipeline_class = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert get_origin(pipeline_class) is TextGenerationPipeline


def test_devices_takes_the_cli_format(
    model_path: str, built_configs: list[PipelineConfig]
) -> None:
    with pytest.raises(_StopBeforeBuild):
        AutoModel(model_path, devices="gpu:0")
    [config] = built_configs
    assert config.model.device_specs == [DeviceSpec.accelerator(0)]

    with pytest.raises(TypeError, match="devices or device_specs"):
        AutoModel(model_path, devices="cpu", device_specs=[DeviceSpec.cpu()])


def test_signature_defaults(
    model_path: str, built_configs: list[PipelineConfig]
) -> None:
    with pytest.raises(_StopBeforeBuild):
        AutoModel(model_path, device_specs=[DeviceSpec.accelerator()])
    [config] = built_configs
    assert config.runtime.max_batch_size == 1
    assert config.runtime.device_graph_capture is False
    assert config.runtime.enable_overlap_scheduler is False
    assert config.runtime.force is False
    assert config.runtime.prefer_module_v3 is False


def test_prefer_module_v3_reaches_the_config(
    model_path: str, built_configs: list[PipelineConfig]
) -> None:
    with pytest.raises(_StopBeforeBuild):
        AutoModel(model_path, devices="cpu", prefer_module_v3=True)
    [config] = built_configs
    assert config.runtime.prefer_module_v3 is True


def test_prefer_module_v3_loads_and_generates(model_path: str) -> None:
    with AutoModel(
        model_path,
        devices="cpu",
        prefer_module_v3=True,
        quantization_encoding="float32",
        max_length=512,
    ) as model:
        pipeline = model.pipeline
        assert isinstance(pipeline, TextGenerationPipeline)
        assert isinstance(pipeline._pipeline_model, ModuleV3Llama3Model)
        [sequence] = model.generate(
            PROMPTS[:1], max_new_tokens=NUM_NEW_TOKENS, ignore_eos=True
        ).sequences
    assert len(sequence) == len(PROMPTS[0]) + NUM_NEW_TOKENS


@pytest.mark.parametrize(
    ("api", "overrides", "expected"),
    [
        ("hip", {}, True),
        ("hip", {"disable_vendor_blas_fallback": False}, False),
        ("cuda", {}, False),
    ],
)
def test_vendor_blas_fallback_off_on_amd(
    model_path: str,
    built_configs: list[PipelineConfig],
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    overrides: Mapping[str, object],
    expected: bool,
) -> None:
    monkeypatch.delenv("MODULAR_DISABLE_VENDOR_FALLBACK", raising=False)
    monkeypatch.setattr(auto_model, "accelerator_api", lambda: api)
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(model_path, False, **overrides)
    [config] = built_configs
    assert config.runtime.disable_vendor_blas_fallback is expected


@pytest.fixture
def llama_arch_required(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Replaces SmolLM's architecture's required arguments for one test."""
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "LlamaForCausalLM", task=PipelineTask.TEXT_GENERATION
    )
    assert arch is not None
    required: dict[str, object] = {}
    monkeypatch.setattr(arch, "required_arguments", required)
    return required


def test_required_arguments_win_over_debug_defaults(
    model_path: str,
    built_configs: list[PipelineConfig],
    llama_arch_required: dict[str, object],
) -> None:
    # Stands in for an architecture that mandates a value the debug
    # defaults contradict.
    llama_arch_required["enable_prefix_caching"] = True
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(model_path)
    [config] = built_configs
    assert config.model.kv_cache.enable_prefix_caching is True


def test_required_arguments_that_name_no_field_are_skipped(
    model_path: str,
    built_configs: list[PipelineConfig],
    llama_arch_required: dict[str, object],
) -> None:
    llama_arch_required.update(
        {"not_a_config_field": True, "enable_prefix_caching": True}
    )
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(model_path)
    [config] = built_configs
    assert config.model.kv_cache.enable_prefix_caching is True


def test_required_arguments_win_over_model_override(
    model_path: str,
    built_configs: list[PipelineConfig],
    llama_arch_required: dict[str, object],
) -> None:
    llama_arch_required["trust_remote_code"] = True
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(
            model_path, model_override=["main.trust_remote_code=false"]
        )
    [config] = built_configs
    assert config.model.trust_remote_code is True


def _assert_points_to_serve(
    error: pytest.ExceptionInfo[ValueError], feature: str
) -> None:
    lines = str(error.value).splitlines()
    assert lines[0].startswith(f"AutoModel does not support {feature}. ")
    assert lines[1].startswith(
        "AutoModel runs the pipeline directly, outside max serve."
    )
    assert lines[2].startswith("  max serve --model-path ")


def test_loads_speculative_decoding(
    model_path: str,
    built_configs: list[PipelineConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolving a speculative config rewrites the cached Hugging Face
    # config's architecture in place, which would leak into later tests.
    monkeypatch.setattr(_hf_config, "_config_cache", {})
    with pytest.raises(_StopBeforeBuild):
        _load_accelerator(
            model_path, speculative_method="eagle", draft_model_path=model_path
        )
    [config] = built_configs
    assert config.speculative is not None


def test_rejects_lora(
    model_path: str, built_configs: list[PipelineConfig]
) -> None:
    with pytest.raises(ValueError) as error:
        _load_cpu(model_path, enable_lora=True)
    _assert_points_to_serve(error, "LoRA adapters")
    assert " --enable-lora " in str(error.value)
    assert built_configs == []


def test_rejects_an_embeddings_model(
    built_configs: list[PipelineConfig],
) -> None:
    with pytest.raises(ValueError) as error:
        AutoModel(_local_or_repo(EMBEDDINGS_REPO_ID), devices="cpu")
    message = str(error.value)
    assert "AutoModel runs text and image generation models" in message
    assert "MPNetForMaskedLM for embeddings generation" in message
    assert built_configs == []


class _Cyclic:
    def __init__(self) -> None:
        self.self = self


def test_collects_cycles_before_resolving(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycle = _Cyclic()
    cycle_ref = weakref.ref(cycle)
    del cycle
    collected_first: list[bool] = []

    def capture(args: PipelineArgs) -> PipelineConfig:
        collected_first.append(cycle_ref() is None)
        raise _StopBeforeBuild

    monkeypatch.setattr(PipelineConfig, "from_args", capture)
    gc.disable()
    try:
        with pytest.raises(_StopBeforeBuild):
            _load_cpu(model_path)
    finally:
        gc.enable()
    assert collected_first == [True]


def test_close_shuts_down_once_and_releases(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutdowns: list[None] = []
    with _load_cpu(model_path) as model:
        monkeypatch.setattr(
            model.pipeline.kv_manager,
            "shutdown",
            lambda: shutdowns.append(None),
        )
        pipeline_ref = weakref.ref(model.pipeline)
    model.close()

    assert shutdowns == [None]
    assert pipeline_ref() is None
    with pytest.raises(RuntimeError, match="closed"):
        _ = model.pipeline
    with pytest.raises(RuntimeError, match="closed"):
        model.generate(PROMPTS, max_new_tokens=1)
    with pytest.raises(RuntimeError, match="closed"):
        model.generate_iter(PROMPTS, max_new_tokens=1)
    assert model.pipeline_config.runtime.max_batch_size == 1


def test_close_releases_when_shutdown_raises(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise RuntimeError("shutdown failed")

    model = _load_cpu(model_path)
    monkeypatch.setattr(model.pipeline.kv_manager, "shutdown", fail)
    pipeline_ref = weakref.ref(model.pipeline)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        model.close()

    assert pipeline_ref() is None
    model.close()


def _stub_retrieve(
    monkeypatch: pytest.MonkeyPatch, pipeline: MagicMock, events: list[str]
) -> None:
    def retrieve(pipeline_config: PipelineConfig) -> tuple[object, MagicMock]:
        disable_overlap = pipeline_config.runtime.disable_overlap
        events.append(f"retrieve(disable_overlap={disable_overlap})")
        return pipeline.tokenizer, pipeline

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)


def _stub_overlap_pipeline(events: list[str]) -> MagicMock:
    pipeline = MagicMock(spec=OverlapTextGenerationPipeline)
    pipeline.max_batch_size = 1
    pipeline.warmup_graph_capture.side_effect = lambda: events.append("warmup")
    return pipeline


def test_generate_and_generate_iter_pass_every_argument(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MagicMock(spec=TextGenerationPipeline)
    _stub_retrieve(monkeypatch, pipeline, [])
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(auto_model, "generate", record)
    monkeypatch.setattr(auto_model, "GenerateIterator", record)
    messages: list[list[Mapping[str, object]]] = [
        [{"role": "user", "content": "hi"}]
    ]
    images = [[b"image"]]
    with _load_cpu(model_path) as model:
        model.generate(
            PROMPTS,
            MASKS,
            messages=messages,
            images=images,
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=3,
            min_new_tokens=2,
            temperature=0.5,
            top_k=4,
            top_p=0.9,
            min_p=0.1,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            ignore_eos=True,
            stop_token_ids=[9],
            seed=7,
            output_logits=True,
            output_logprobs=2,
        )
        model.generate_iter(
            PROMPTS,
            MASKS,
            messages=messages,
            images=images,
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=3,
            min_new_tokens=2,
            temperature=0.5,
            top_k=4,
            top_p=0.9,
            min_p=0.1,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            ignore_eos=True,
            stop_token_ids=[9],
            seed=7,
            output_logits=True,
            output_logprobs=2,
        )
    expected_kwargs: dict[str, object] = {
        "messages": messages,
        "images": images,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_new_tokens": 3,
        "min_new_tokens": 2,
        "temperature": 0.5,
        "top_k": 4,
        "top_p": 0.9,
        "min_p": 0.1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "repetition_penalty": 1.1,
        "ignore_eos": True,
        "stop_token_ids": [9],
        "seed": 7,
        "output_logits": True,
        "output_logprobs": 2,
        "caller": "AutoModel",
    }
    assert calls == [((pipeline, PROMPTS, MASKS), expected_kwargs)] * 2


def test_generate_forwards_text_defaults_unchanged(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MagicMock(spec=TextGenerationPipeline)
    _stub_retrieve(monkeypatch, pipeline, [])
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(auto_model, "generate", record)
    with _load_cpu(model_path) as model:
        model.generate(PROMPTS, max_new_tokens=3)
    assert calls == [
        (
            (pipeline, PROMPTS, None),
            {
                "messages": None,
                "images": None,
                "chat_template_kwargs": None,
                "max_new_tokens": 3,
                "min_new_tokens": None,
                "temperature": None,
                "top_k": None,
                "top_p": None,
                "min_p": None,
                "frequency_penalty": None,
                "presence_penalty": None,
                "repetition_penalty": None,
                "ignore_eos": False,
                "stop_token_ids": None,
                "seed": 0,
                "output_logits": False,
                "output_logprobs": 0,
                "caller": "AutoModel",
            },
        )
    ]


def test_image_arguments_are_rejected_on_a_text_model(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retrieve(monkeypatch, MagicMock(spec=TextGenerationPipeline), [])
    loaded = f"AutoModel loaded {model_path}, a text generation model."
    with _load_cpu(model_path) as model:
        # No overload takes the mixed calls, so they go through an untyped
        # name.
        untyped: Callable[..., object] = model.generate
        with pytest.raises(ValueError) as error:
            model.generate(prompt="a red fox")
        assert str(error.value) == (
            f"prompt is for image generation models, and {loaded} Pass "
            "input_ids or messages instead."
        )
        with pytest.raises(ValueError) as error:
            untyped(PROMPTS, max_new_tokens=3, height=256, width=256)
        assert str(error.value) == (
            f"height, width are for image generation models, and {loaded}"
        )
        with pytest.raises(TypeError, match="'max_new_tokens'"):
            untyped(PROMPTS)


# A value other than the default for each image-only argument.
_IMAGE_ONLY_VALUES: dict[str, object] = {
    "prompt": "a red fox",
    "negative_prompt": "blurry",
    "height": 256,
    "width": 256,
    "num_inference_steps": 4,
    "guidance_scale": 2.0,
    "true_cfg_scale": 2.0,
    "strength": 0.5,
    "num_images_per_prompt": 2,
}


def test_every_image_argument_is_rejected_on_a_text_model(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    text: set[str] = set()
    image: set[str] = set()
    for overload in typing_extensions.get_overloads(AutoModel.generate):
        names = set(inspect.signature(overload).parameters) - {"self"}
        (image if "prompt" in names else text).update(names)
    _stub_retrieve(monkeypatch, MagicMock(spec=TextGenerationPipeline), [])
    with _load_cpu(model_path) as model:
        untyped: Callable[..., object] = model.generate
        for name in sorted(image - text):
            with pytest.raises(ValueError) as error:
                untyped(
                    PROMPTS,
                    max_new_tokens=1,
                    **{name: _IMAGE_ONLY_VALUES[name]},
                )
            assert str(error.value).startswith(
                f"{name} is for image generation models"
            )


def test_rejects_a_pipeline_that_does_not_generate_text(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_refs: list[weakref.ref[_Cyclic]] = []

    def retrieve(pipeline_config: PipelineConfig) -> tuple[None, _Cyclic]:
        # A cycle, so only a collection frees it.
        pipeline = _Cyclic()
        pipeline_refs.append(weakref.ref(pipeline))
        return None, pipeline

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)
    with pytest.raises(ValueError) as error:
        _load_cpu(model_path)
    assert str(error.value) == (
        f"AutoModel runs text and image generation models, and {model_path} "
        "loads as _Cyclic."
    )
    [pipeline_ref] = pipeline_refs
    assert pipeline_ref() is None


def test_graph_capture_warms_up_once_after_a_synchronous_build(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _stub_retrieve(monkeypatch, _stub_overlap_pipeline(events), events)
    with _load_accelerator(model_path, device_graph_capture=True) as model:
        runtime = model.pipeline_config.runtime
        assert runtime.device_graph_capture is True
        assert runtime.enable_overlap_scheduler is True
        assert runtime.force is False
    assert events == ["retrieve(disable_overlap=True)", "warmup"]


def test_graph_capture_keeps_overlap_the_caller_requests(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _stub_retrieve(monkeypatch, _stub_overlap_pipeline(events), events)
    with _load_accelerator(
        model_path, device_graph_capture=True, enable_overlap_scheduler=True
    ):
        pass
    assert events == ["retrieve(disable_overlap=False)", "warmup"]


def test_no_graph_capture_never_warms_up(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _stub_retrieve(monkeypatch, _stub_overlap_pipeline(events), events)
    with _load_accelerator(model_path):
        pass
    assert events == ["retrieve(disable_overlap=False)"]


def test_graph_capture_on_a_pipeline_that_cannot_capture_tears_down(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MagicMock(spec=TextGenerationPipeline)
    _stub_retrieve(monkeypatch, pipeline, [])
    with pytest.raises(ValueError, match="OverlapTextGenerationPipeline"):
        _load_accelerator(model_path, device_graph_capture=True)
    pipeline.kv_manager.shutdown.assert_called_once_with()


def test_graph_capture_warmup_failure_tears_down(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    pipeline = _stub_overlap_pipeline(events)
    pipeline.warmup_graph_capture.side_effect = RuntimeError("capture failed")
    _stub_retrieve(monkeypatch, pipeline, events)
    with pytest.raises(RuntimeError, match="capture failed"):
        _load_accelerator(model_path, device_graph_capture=True)
    pipeline.kv_manager.shutdown.assert_called_once_with()
