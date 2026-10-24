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
"""Tests for cascade pipeline dispatch in ``all_pipelines``.

Covers the two dispatch paths ``build_pipeline`` exposes: exact-match dummy
fixtures, and architecture-driven selection via
``SupportedArchitecture.cascade_pipeline_factory``. The architecture-driven
cases stub ``_resolve_architecture`` so no Hugging Face config is downloaded,
while a dedicated test exercises the real registry wiring (a text-generation
architecture declares the common text pipeline class).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.core import worker_method
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAIRequest,
    GenAITextChunk,
    RawTextChunk,
    TextGenOptions,
)
from max.experimental.cascade.pipelines import all_pipelines
from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
    chat_parser_config,
)
from max.experimental.cascade.pipelines.dummy_imgen import DummyImageGenPipeline
from max.experimental.cascade.pipelines.dummy_textgen import (
    DummyTextGenPipeline,
)
from max.experimental.cascade.pipelines.echo_textgen import EchoTextGenPipeline
from max.experimental.cascade.workers.max_model_worker import (
    MAXModelWorker,
    make_model_worker,
)
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer
from max.experimental.cascade.workers.model_worker import (
    Int32Array,
    ModelWorker,
)
from max.experimental.cascade.workers.tokenizer_worker import (
    TokenizerWorker,
    stream_incremental_text,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    PipelineArgs,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.reasoning import get_parser_cls as reasoning_parser_cls
from max.pipelines.lib.reasoning import register as register_reasoning_parser
from max.pipelines.modeling.types import ParsedReasoningDelta, ReasoningParser


class _StubReasoningParser(ReasoningParser):
    """Never instantiated: these stubs exist to declare (or omit) delimiters."""

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        raise NotImplementedError


@register_reasoning_parser("_test_no_delimiters")
class _NoDelimiterParser(_StubReasoningParser):
    """Stands in for a parser written only for the token domain."""


@register_reasoning_parser("_test_half_declared_delimiters")
class _HalfDeclaredParser(_StubReasoningParser):
    """Declares a closing delimiter with no opening one."""

    REASONING_END: ClassVar[str | None] = "</think>"


class _StubModelWorker(ModelWorker):
    """Model worker named by ``model_worker_impl`` in these tests."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(deploy_hints=["rust"])
        self.config = config

    @worker_method()
    async def decode(
        self, options: TextGenOptions, tokens: Int32Array
    ) -> AsyncIterator[Int32Array]:
        yield tokens


_STUB_MODEL_WORKER = f"{__name__}:_StubModelWorker"


class _CodePointTokenizer(TokenizerWorker):
    """Offline tokenizer: one token per character, its Unicode code point."""

    def __init__(self, model_path: str) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.model_path = model_path

    @worker_method()
    async def encode(self, messages: list[ChatMessage]) -> Int32Array:
        text = "".join(
            part.text
            for message in messages
            for part in message.content
            if isinstance(part, GenAITextChunk)
        )
        return np.array([ord(char) for char in text], dtype=np.int32)

    @worker_method()
    async def decode(
        self, tokens: Int32Array, skip_special_tokens: bool
    ) -> str:
        return "".join(map(chr, tokens.tolist()))

    @worker_method()
    async def decode_stream(
        self, token_iter: AsyncIterable[Int32Array], skip_special_tokens: bool
    ) -> AsyncIterator[RawTextChunk]:
        async for piece in stream_incremental_text(
            token_iter, lambda ids: "".join(map(chr, ids))
        ):
            yield piece


class _FixedReplyModelWorker(ModelWorker):
    """Generates the code points of a fixed reply, ignoring the prompt."""

    REPLY: ClassVar[str] = "from the custom worker"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, options: TextGenOptions, tokens: Int32Array
    ) -> AsyncIterator[Int32Array]:
        for char in self.REPLY:
            yield np.array([ord(char)], dtype=np.int32)


def _args(
    model_path: str,
    tokenizer_impl: str | None = None,
    model_worker_impl: str | None = None,
) -> PipelineArgs:
    """Build raw pipeline args for construction-only (no-download) tests."""
    return PipelineArgs(
        model_path=model_path,
        tokenizer_impl=tokenizer_impl,
        model_worker_impl=model_worker_impl,
    )


@pytest.mark.asyncio
async def test_build_pipeline_dummy_textgen() -> None:
    pipeline = await all_pipelines.build_pipeline(_args("dummy_textgen"))
    assert isinstance(pipeline, DummyTextGenPipeline)


@pytest.mark.asyncio
async def test_build_pipeline_dummy_imgen() -> None:
    pipeline = await all_pipelines.build_pipeline(_args("dummy_imgen"))
    assert isinstance(pipeline, DummyImageGenPipeline)


@pytest.mark.asyncio
async def test_build_pipeline_echo() -> None:
    # An ``echo:`` model-path prefix skips config construction and architecture
    # resolution entirely (no network), building an echo pipeline for the
    # remaining tokenizer path.
    pipeline = await all_pipelines.build_pipeline(
        _args("echo:some-org/some-llm")
    )
    assert isinstance(pipeline, EchoTextGenPipeline)
    assert pipeline.tokenizer.model_path == "some-org/some-llm"


def _stub_llama_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route dispatch to the real Llama arch, stubbing arch resolution and
    ``retrieve_factory`` so tests never hit the network.
    """
    register_all_models()
    arch = PIPELINE_REGISTRY.retrieve_architecture("LlamaForCausalLM")
    assert arch is not None
    monkeypatch.setattr(
        all_pipelines, "_resolve_architecture", lambda config: arch
    )
    monkeypatch.setattr(
        all_pipelines.PIPELINE_REGISTRY,
        "retrieve_factory",
        lambda config: SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_ids=set()),
            factory=lambda: None,
            memory_plan=None,
        ),
    )


@pytest.mark.asyncio
async def test_build_pipeline_uses_arch_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_llama_arch(monkeypatch)
    pipeline = await all_pipelines.build_pipeline(_args("some-org/some-llm"))
    assert isinstance(pipeline, CommonTextGenPipeline)
    assert isinstance(pipeline.model, MAXModelWorker)


@pytest.mark.asyncio
async def test_build_pipeline_arch_factory_threads_tokenizer_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit tokenizer_impl must survive the registry-factory path
    # rather than being lost to the constructor default.
    _stub_llama_arch(monkeypatch)
    pipeline = await all_pipelines.build_pipeline(
        _args(
            "some-org/some-llm",
            "max.experimental.cascade.workers.max_tokenizer:MAXTokenizer",
        )
    )
    assert isinstance(pipeline, CommonTextGenPipeline)
    assert isinstance(pipeline.tokenizer, MAXTokenizer)


@pytest.mark.asyncio
async def test_build_pipeline_arch_factory_threads_model_worker_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_llama_arch(monkeypatch)
    pipeline = await all_pipelines.build_pipeline(
        _args("some-org/some-llm", model_worker_impl=_STUB_MODEL_WORKER)
    )
    assert isinstance(pipeline, CommonTextGenPipeline)
    assert isinstance(pipeline.model, _StubModelWorker)
    assert pipeline.model.config.model.model_path == "some-org/some-llm"


@pytest.mark.asyncio
async def test_build_pipeline_rejects_model_worker_impl_for_other_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the common text-generation pipeline uses ``model_worker_impl``."""
    stub_arch = SimpleNamespace(
        name="SomeForCausalLM", cascade_pipeline_factory=DummyTextGenPipeline
    )
    monkeypatch.setattr(
        all_pipelines, "_resolve_architecture", lambda config: stub_arch
    )
    with pytest.raises(ValueError, match="--model-worker-impl"):
        await all_pipelines.build_pipeline(
            _args("some-org/some-model", model_worker_impl=_STUB_MODEL_WORKER)
        )


@pytest.mark.asyncio
async def test_common_textgen_generates_through_custom_model_worker() -> None:
    """``model_worker_impl`` swaps the model worker the pipeline decodes with."""
    pipeline = CommonTextGenPipeline(
        PipelineConfig.from_args(_args("some-org/some-llm")),
        tokenizer_impl=f"{__name__}:_CodePointTokenizer",
        model_worker_impl=f"{__name__}:_FixedReplyModelWorker",
    )
    request = GenAIRequest(
        messages=[ChatMessage.text("user", "hello")],
        text=TextGenOptions(num_tokens=64),
    )
    async with LocalRuntime() as runtime:
        await pipeline.deploy(runtime)
        chunks = [chunk async for chunk in pipeline.generate(request)]

    assert all(isinstance(chunk, GenAITextChunk) for chunk in chunks)
    assert (
        "".join(
            chunk.text for chunk in chunks if isinstance(chunk, GenAITextChunk)
        )
        == _FixedReplyModelWorker.REPLY
    )


def test_make_model_worker_requires_module_and_class() -> None:
    config = PipelineConfig.from_args(_args("some-org/some-llm"))
    with pytest.raises(ValueError, match=r"module\.path:ClassName"):
        make_model_worker(config, "no_class_name")


def test_make_model_worker_requires_a_model_worker() -> None:
    config = PipelineConfig.from_args(_args("some-org/some-llm"))
    with pytest.raises(TypeError, match="expected a ModelWorker subclass"):
        make_model_worker(
            config,
            "max.experimental.cascade.workers.max_tokenizer:MAXTokenizer",
        )


@pytest.mark.asyncio
async def test_build_pipeline_arch_without_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_arch = SimpleNamespace(
        name="SomeForCausalLM", cascade_pipeline_factory=None
    )
    monkeypatch.setattr(
        all_pipelines, "_resolve_architecture", lambda config: stub_arch
    )
    with pytest.raises(NotImplementedError, match="no cascade"):
        await all_pipelines.build_pipeline(_args("some-org/some-model"))


@pytest.mark.asyncio
async def test_build_pipeline_no_models() -> None:
    with pytest.raises(ValueError, match="No models specified"):
        await all_pipelines.build_pipeline(PipelineArgs())


@pytest.fixture(autouse=True)
def _offline_hf_construction() -> Iterator[None]:
    """Keep ``MAXModelConfig`` construction offline (CI runs
    ``HF_HUB_OFFLINE=1``): ``__init__`` eagerly builds the HuggingFace repo
    handles. Real cached repos resolve normally; uncached/placeholder repos
    get a fake path.
    """

    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
    ):
        yield


def test_every_arch_reasoning_parser_declares_text_delimiters() -> None:
    # Cascade parses after detokenization, so it takes each model's thinking
    # region from the delimiters its ReasoningParser declares. A parser that
    # declares none would silently surface reasoning as ordinary content, which
    # is what a hand-maintained table of delimiters used to do by omission.
    register_all_models()
    missing = sorted(
        {
            arch.reasoning_parser
            for arch in PIPELINE_REGISTRY.all_architectures()
            if arch.reasoning_parser is not None
            and not (
                (cls := reasoning_parser_cls(arch.reasoning_parser))
                and cls.REASONING_START
                and cls.REASONING_END
            )
        }
    )
    assert not missing, (
        "Reasoning parsers named by an architecture but declaring no "
        f"REASONING_START/REASONING_END: {missing}"
    )


def _config_using_reasoning_parser(name: str) -> PipelineConfig:
    args = PipelineArgs(
        model_path="some-org/some-llm",
        runtime=PipelineRuntimeConfig(reasoning_parser=name),
    )
    return PipelineConfig.from_args(args)


def test_reasoning_parser_without_delimiters_opts_out() -> None:
    # A parser with no text form is a deliberate opt-out, not a
    # misconfiguration: reasoning stays in the assistant's content.
    parser_config = chat_parser_config(
        _config_using_reasoning_parser("_test_no_delimiters")
    )
    assert parser_config.reasoning_start is None
    assert parser_config.reasoning_end is None
    assert not parser_config.reasoning_enabled


def test_half_declared_reasoning_delimiters_are_fatal() -> None:
    # One end of a span cannot bound it, so neither honoring nor ignoring the
    # declaration is right; fail while the pipeline is being built.
    with pytest.raises(ValueError, match="declare both delimiters or neither"):
        chat_parser_config(
            _config_using_reasoning_parser("_test_half_declared_delimiters")
        )


def test_llama_arch_declares_cascade_factory() -> None:
    # End-to-end check of the integration the dispatcher depends on: a real
    # text-generation architecture declares the common text pipeline class.
    register_all_models()
    arch = PIPELINE_REGISTRY.retrieve_architecture("LlamaForCausalLM")
    assert arch is not None
    assert arch.cascade_pipeline_factory is CommonTextGenPipeline
