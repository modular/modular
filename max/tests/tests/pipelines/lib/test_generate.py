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
"""``GenerateIterator`` bookkeeping over fake pipelines.

The fakes stand in for a model because everything under test is bookkeeping
over ``execute``: which rows run each step, when a row's KV cache and
pipeline state are released, and whether a failed run leaves anything
claimed. Each prompt's first token is the step on which it emits EOS, and
the fakes update the real ``TextContext`` so EOS and max-length stops come
from the context itself, as they do in a real pipeline. The fakes also run
each context's logits processors on logits whose argmax is the token they
emit, then overwrite those logits as a graph replay would, and attach log
probabilities when a context asks for them.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import gc
import io
import logging
import weakref
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max._entrypoints.cli.generate import _with_debug_defaults, stream_context
from max.driver import Buffer, DeviceSpec
from max.pipelines.context import (
    EOSTracker,
    GenerationStatus,
    InputError,
    LogProbabilities,
    PixelContext,
    ProcessorInputs,
    PromptTooLongError,
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.lib.auto_processor import _model_inputs
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.generate import (
    GenerateIterator,
    GenerateResult,
    generate,
)
from max.pipelines.lib.interfaces import PipelineModel
from max.pipelines.lib.log_probabilities import _MAX_TOP_LOGPROBS
from max.pipelines.lib.pipeline_args import PipelineArgs
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.block_diffusion_text_generation import (
    BlockDiffusionTextGenerationPipeline,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.lib.tokenizer import open_image
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types import (
    BatchType,
    ImageContentPart,
    InputModality,
    PipelineTask,
    RequestID,
    TextContentPart,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.sampling import SamplingConfig, request_row_seed
from max.pipelines.speculative import SpeculativeConfig
from PIL import Image

EOS = 2
NEVER = 99
MODEL_PATH = "org/model"
VOCAB = 128
# Wins the argmax on every prefill position but the last.
DECOY = 1
IMAGE_TOKEN = 50
IMAGE_TOKENS = 4
TOKEN_ID_BOUND = 1000


@dataclasses.dataclass
class FakeArch:
    name: str = "TextModel"
    task: PipelineTask = PipelineTask.TEXT_GENERATION
    pipeline_cls: type | None = None
    input_modalities: set[InputModality] = dataclasses.field(
        default_factory=lambda: {InputModality.TEXT}
    )


@pytest.fixture(autouse=True)
def text_context_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves the fake config to a text-only architecture."""
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: TextContext,
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "architecture_for_config",
        lambda pipeline_config: FakeArch(),
    )


@dataclasses.dataclass
class FakeTransfer:
    """An onload that lands only once synchronized."""

    landed: bool = False

    def is_complete(self) -> bool:
        return self.landed

    def synchronize(self) -> None:
        self.landed = True


@dataclasses.dataclass
class FakeKVParams:
    page_size: int = 4


@dataclasses.dataclass
class FakeKVManager:
    """Tracks claims by request id, since a context is not hashable."""

    params: FakeKVParams = dataclasses.field(default_factory=FakeKVParams)
    claimed: dict[RequestID, int] = dataclasses.field(default_factory=dict)
    contexts: list[TextContext] = dataclasses.field(default_factory=list)
    released: list[RequestID] = dataclasses.field(default_factory=list)
    transfers: dict[RequestID, FakeTransfer] = dataclasses.field(
        default_factory=dict
    )
    fail_on_claim: int | None = None

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        if len(self.claimed) == self.fail_on_claim:
            raise RuntimeError("claim failed")
        self.claimed[ctx.request_id] = replica_idx
        self.contexts.append(ctx)

    def alloc(self, ctx: TextContext) -> FakeTransfer:
        assert ctx.request_id in self.claimed
        transfer = FakeTransfer()
        self.transfers[ctx.request_id] = transfer
        return transfer

    def release(self, ctx: TextContext) -> None:
        del self.claimed[ctx.request_id]
        self.released.append(ctx.request_id)

    def contains(self, ctx: TextContext) -> bool:
        return ctx.request_id in self.claimed


@dataclasses.dataclass
class FakeTokenizer:
    eos_token_ids: set[int] = dataclasses.field(default_factory=lambda: {EOS})
    tokenizer_vocab_size: int = 1000
    token_id_bound: int = TOKEN_ID_BOUND


@dataclasses.dataclass
class FakeVisionTokenizer(FakeTokenizer):
    """Builds vision contexts as an image-capable model's tokenizer does.

    A message's text is its token ids, space separated, and each image part
    expands to ``IMAGE_TOKENS`` placeholder tokens. Each context's max length
    is over any plan, as a tokenizer's own max length can be.
    """

    requests: list[TextGenerationRequest] = dataclasses.field(
        default_factory=list
    )
    error: Exception | None = None
    max_prompt_tokens: int | None = None

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        if self.error is not None:
            raise self.error
        self.requests.append(request)
        if request.messages:
            prompt = [
                token
                for message in request.messages
                for token in _message_tokens(message)
            ]
        else:
            assert isinstance(request.prompt, list)
            prompt = request.prompt
        if self.max_prompt_tokens is not None and (
            len(prompt) > self.max_prompt_tokens
        ):
            raise PromptTooLongError(len(prompt), self.max_prompt_tokens)
        params = request.sampling_params
        return TextAndVisionContext(
            request_id=request.request_id,
            max_length=10_000,
            tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
            eos_tracker=EOSTracker(
                eos_token_ids=set() if params.ignore_eos else {EOS}
            ),
            sampling_params=params,
            vision_token_ids=[],
        )


def _message_tokens(message: TextGenerationRequestMessage) -> list[int]:
    if isinstance(message.content, str):
        return [int(token) for token in message.content.split()]
    tokens: list[int] = []
    for part in message.content:
        if isinstance(part, TextContentPart):
            tokens += [int(token) for token in part.text.split()]
        else:
            tokens += [IMAGE_TOKEN] * IMAGE_TOKENS
    return tokens


@dataclasses.dataclass
class FakeRuntime:
    max_batch_active_tokens: int


@dataclasses.dataclass
class FakeBatchProcessor:
    runtime: FakeRuntime


@dataclasses.dataclass
class FakeLoRAManager:
    loaded: set[str]
    active: set[str]

    def is_lora(self, name: str) -> bool:
        return name in self.loaded

    def is_active_lora(self, name: str) -> bool:
        return name in self.active


@dataclasses.dataclass
class FakeModel:
    max_seq_len: int = 64
    batch_processor: FakeBatchProcessor | None = None
    planned_max_batch_total_tokens: int | None = None
    lora_manager: FakeLoRAManager | None = None


class FakePipeline:
    """Emits EOS on the step a prompt's first token names, else step ids.

    Like ``TextGenerationPipeline``, it returns each step's tokens from the
    same ``execute``. It also checks that every row's KV onload landed
    before the forward reads it.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        data_parallel_degree: int = 1,
        max_seq_len: int = 64,
        max_batch_active_tokens: int | None = None,
        max_batch_total_tokens: int | None = None,
        page_size: int = 4,
        generation_config: SamplingParamsGenerationConfigDefaults | None = None,
        runtime_temperature: float | None = None,
        runtime_top_k: int | None = None,
        speculative: SpeculativeConfig | None = None,
        draft_model_path: str | None = None,
        lora: LoRAConfig | None = None,
        device_graph_capture: bool = False,
        enable_penalties: bool = False,
        tokenizer_vocab_size: int = 1000,
        device_specs: list[DeviceSpec] | None = None,
        quantization_encoding: str | None = None,
        max_length: int | None = None,
        max_length_user_provided: bool = False,
        trust_remote_code: bool = False,
        weight_path: list[Path] | None = None,
        weights_repo_id: str = MODEL_PATH,
    ) -> None:
        self.kv_manager = FakeKVManager(FakeKVParams(page_size))
        self.tokenizer = FakeTokenizer(
            tokenizer_vocab_size=tokenizer_vocab_size
        )
        self.max_batch_size = max_batch_size
        self._pipeline_model = FakeModel(
            max_seq_len,
            None
            if max_batch_active_tokens is None
            else FakeBatchProcessor(FakeRuntime(max_batch_active_tokens)),
            max_batch_total_tokens,
        )
        config = MagicMock(spec=PipelineConfig)
        config.model.model_path = MODEL_PATH
        config.model.data_parallel_degree = data_parallel_degree
        config.model.device_specs = device_specs or [DeviceSpec.accelerator(0)]
        config.model.quantization_encoding = quantization_encoding
        config.model.max_length = max_length
        config.model.max_length_is_user_provided = max_length_user_provided
        config.model.trust_remote_code = trust_remote_code
        config.model.weight_path = weight_path or []
        config.model.huggingface_weight_repo_id = weights_repo_id
        config.model.sampling_params_defaults = (
            generation_config or SamplingParamsGenerationConfigDefaults()
        )
        config.models = {"main": config.model}
        config.runtime = MagicMock(
            spec=PipelineRuntimeConfig,
            temperature=runtime_temperature,
            top_k=runtime_top_k,
            device_graph_capture=device_graph_capture,
            # A real runtime's max_batch_active_tokens sums the per-replica
            # max_batch_input_tokens over DP replicas.
            max_batch_input_tokens=None
            if max_batch_active_tokens is None
            else max_batch_active_tokens // data_parallel_degree,
        )
        config.speculative = speculative
        config.draft_model = (
            None
            if draft_model_path is None
            else MagicMock(model_path=draft_model_path)
        )
        config.lora = lora
        config.sampling = SamplingConfig(enable_penalties=enable_penalties)
        self.pipeline_config: PipelineConfig = config
        self.fail_on_step: int | None = None
        self.fail_next_release = False
        self.run_logits_processors = True
        self.return_log_probs = True
        self.report_sampled_tokens = True
        self.steps_run = 0
        # Rows per replica on each step.
        self.batch_sizes: list[list[int]] = []
        self.batch_types: list[BatchType] = []
        self.released: list[RequestID] = []

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        self.steps_run += 1
        if self.steps_run == self.fail_on_step:
            raise RuntimeError("execute interrupted")
        self.batch_sizes.append([len(batch) for batch in inputs.batches])
        self.batch_types.append(inputs.batch_type)
        for context in inputs.flat_batch:
            assert self.kv_manager.transfers[context.request_id].landed
        return self._forward(inputs.flat_batch)

    def _forward(
        self, contexts: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for context in contexts:
            token = _next_token(context)
            if self.run_logits_processors:
                logits = Buffer.from_numpy(_logits(context, token))
                for processor in (
                    context.sampling_params.logits_processors or ()
                ):
                    processor(ProcessorInputs(logits=logits, context=context))
                # Replay reuses its output buffer on the next step.
                logits.to_numpy().fill(np.nan)
            log_probs = None
            if context.log_probabilities and self.return_log_probs:
                log_prob = -float(context.tokens.generated_length + 1)
                log_probs = LogProbabilities(
                    token_log_probabilities=[log_prob],
                    top_log_probabilities=[{token: log_prob, 0: -50.0}],
                    sampled_token_ids=[
                        token if self.report_sampled_tokens else 0
                    ],
                )
            context.update(token, log_probabilities=log_probs)
            outputs[context.request_id] = context.to_generation_output()
        return outputs

    def release(self, request_id: RequestID) -> None:
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("release failed")
        self.released.append(request_id)


class FakeOverlapDisabledPipeline(FakePipeline):
    """An overlap pipeline with overlap turned off, which runs synchronously."""

    overlap_active = False


class FakeOverlapPipeline(FakePipeline):
    """Returns the previous execute's tokens, like the overlap pipeline.

    Each forward appends a future token and is synced on the next execute,
    including rows released between executes.
    """

    overlap_active = True

    def __init__(self, device_graph_capture: bool = False) -> None:
        super().__init__(device_graph_capture=device_graph_capture)
        self._pending: list[tuple[TextContext, int]] = []

    def has_pending_outputs(self) -> bool:
        return bool(self._pending)

    def _forward(
        self, contexts: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        # The new forward is sampled from the device before the previous
        # batch syncs, so it cannot see whether that batch ended a row.
        sampled = [(context, _next_token(context)) for context in contexts]
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for context, token in self._pending:
            context.realize_future_token(token)
            outputs[context.request_id] = context.to_generation_output()
        for context in contexts:
            context.update_with_future_token()
        self._pending = sampled
        return outputs


class FakeWarmedUpPipeline(FakeOverlapDisabledPipeline):
    """A synchronous pipeline that ran ``warmup_graph_capture()``."""

    graph_capture_ready = True


class FakeWarmedUpOverlapPipeline(FakeOverlapPipeline):
    """A lagged pipeline that ran ``warmup_graph_capture()``."""

    graph_capture_ready = True


def _next_token(context: TextContext) -> int:
    step = context.tokens.generated_length + 1
    return EOS if step == int(context.tokens.prompt[0]) else 100 + step


def _logits(context: TextContext, token: int) -> npt.NDArray[np.float32]:
    """Returns a logits row per active token, as ``ReturnLogits.ALL`` would.

    Only the last row predicts ``token``, and its peak grows each step.
    """
    logits = np.zeros((context.tokens.active_length, VOCAB), dtype=np.float32)
    logits[:-1, DECOY] = 100.0
    logits[-1, token] = 5.0 + context.tokens.generated_length
    return logits


def _prompts(*eos_steps: int) -> list[list[int]]:
    return [[eos_step, 7, 8] for eos_step in eos_steps]


def _new_tokens(sequences: list[list[int]]) -> list[list[int]]:
    """Drops the 3-token prompt :func:`_prompts` starts each row with."""
    return [sequence[3:] for sequence in sequences]


def _assert_points_to_serve(
    error: pytest.ExceptionInfo[ValueError],
    feature: str,
    flags: str = "",
    caller: str = "generate()",
) -> None:
    lines = str(error.value).splitlines()
    assert lines[0].startswith(f"{caller} does not support {feature}. ")
    assert lines[1] == (
        f"{caller} runs the pipeline directly, outside max serve. "
        f"Test {feature} with this max serve command instead."
    )
    assert lines[2] == f"  max serve --model-path {MODEL_PATH}{flags}"


def _assert_nothing_held(pipeline: FakePipeline, num_rows: int) -> None:
    assert pipeline.kv_manager.claimed == {}, "a row was left claimed"
    assert len(pipeline.kv_manager.released) == num_rows
    assert len(set(pipeline.released)) == num_rows
    assert len(pipeline.released) == num_rows, "a row was released twice"


def test_finished_rows_leave_the_batch() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        assert batch.synchronous
        steps = []
        while not batch.done:
            steps.append(batch.step())

    assert steps == [{0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    assert pipeline.batch_sizes == [[2], [1], [1]]
    assert _new_tokens(batch.sequences) == [[EOS], [101, 102, EOS]]
    _assert_nothing_held(pipeline, 2)


def test_decode_steps_run_as_token_generation() -> None:
    pipeline = FakePipeline()
    generate(pipeline, _prompts(3), max_new_tokens=5)

    assert pipeline.batch_types == [BatchType.CE, BatchType.TG, BatchType.TG]


def test_rows_stop_at_max_new_tokens() -> None:
    pipeline = FakePipeline()
    output = generate(pipeline, _prompts(NEVER, 2), max_new_tokens=3)

    assert _new_tokens(output.sequences) == [[101, 102, 103], [101, EOS]]
    assert pipeline.batch_sizes == [[2], [2], [1]]
    _assert_nothing_held(pipeline, 2)


def test_sequences_start_with_the_prompt() -> None:
    output = generate(FakePipeline(), _prompts(1, 3), max_new_tokens=5)
    assert output.sequences == [[1, 7, 8, EOS], [3, 7, 8, 101, 102, EOS]]


def test_iterating_a_batch_steps_it_to_the_end() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        steps = list(batch)

    assert steps == [{0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    assert batch.done


def test_attention_mask_drops_padding() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline,
        [[0, 3, 7, 8], [1, 7, 8, 0]],
        attention_mask=[[0, 1, 1, 1], [1, 1, 1, 0]],
        max_new_tokens=5,
    )
    assert _new_tokens(output.sequences) == [[101, 102, EOS], [EOS]]


def test_attention_mask_of_another_shape_raises() -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="same shape as input_ids"):
        GenerateIterator(
            pipeline,
            _prompts(1, 3),
            attention_mask=[[1, 1, 1], [1, 1]],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


def test_ignore_eos_keeps_every_row_to_max_new_tokens() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(
        pipeline, _prompts(1, 3), max_new_tokens=5, ignore_eos=True
    ) as batch:
        for _ in batch:
            pass

    assert pipeline.batch_sizes == [[2]] * 5
    # As with serve's ignore_eos, tokens past EOS stay in the result.
    assert _new_tokens(batch.sequences) == [
        [EOS, 102, 103, 104, 105],
        [101, 102, EOS, 104, 105],
    ]
    _assert_nothing_held(pipeline, 2)


def test_ignore_eos_rejects_a_row_the_plan_would_clamp() -> None:
    pipeline = FakePipeline(max_seq_len=8)
    with pytest.raises(
        ValueError, match=r"planned max length is 8.*Raise max_length"
    ):
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=6, ignore_eos=True
        )
    assert pipeline.kv_manager.claimed == {}


def test_stop_token_ids_end_a_row() -> None:
    output = generate(
        FakePipeline(), _prompts(NEVER), max_new_tokens=5, stop_token_ids=[102]
    )
    assert _new_tokens(output.sequences) == [[101, 102]]


def test_generation_config_min_new_tokens_over_max_new_tokens_raises() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            min_new_tokens=8
        )
    )
    with pytest.raises(
        ValueError,
        match=r"min_new_tokens=8, .* exceeds max_new_tokens=4.*Raise "
        r"max_new_tokens",
    ):
        GenerateIterator(pipeline, _prompts(NEVER), max_new_tokens=4)
    with pytest.raises(ValueError, match=r"min_new_tokens=6, "):
        GenerateIterator(
            FakePipeline(), _prompts(NEVER), max_new_tokens=4, min_new_tokens=6
        )
    assert pipeline.kv_manager.claimed == {}


def test_rows_round_robin_across_replicas() -> None:
    pipeline = FakePipeline(max_batch_size=2, data_parallel_degree=2)
    with GenerateIterator(
        pipeline, _prompts(1, 2, 3, 3), max_new_tokens=5
    ) as batch:
        assert list(pipeline.kv_manager.claimed.values()) == [0, 1, 0, 1]
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[2, 2], [1, 2], [1, 1]]
    _assert_nothing_held(pipeline, 4)


def test_remove_frees_the_row_and_the_rest_finish() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(
        pipeline, _prompts(3, 3, 3), max_new_tokens=5
    ) as batch:
        batch.step()
        batch.remove(1)
        assert len(pipeline.released) == 1
        assert pipeline.kv_manager.released == pipeline.released
        assert pipeline.released[0] not in pipeline.kv_manager.claimed
        batch.remove(1)
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[3], [2], [2]]
    assert _new_tokens(batch.sequences) == [
        [101, 102, EOS],
        [101],
        [101, 102, EOS],
    ]
    _assert_nothing_held(pipeline, 3)


def test_remove_rejects_an_unknown_row() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(pipeline, _prompts(1), max_new_tokens=5) as batch:
        with pytest.raises(IndexError):
            batch.remove(1)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_interrupted_run_releases_rows_and_pipeline_is_reusable(
    error: type[BaseException],
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(error):
        with GenerateIterator(
            pipeline, _prompts(4, 4), max_new_tokens=5
        ) as batch:
            batch.step()
            raise error()
    _assert_nothing_held(pipeline, 2)

    output = generate(pipeline, _prompts(2, 2), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[101, EOS], [101, EOS]]
    _assert_nothing_held(pipeline, 4)


def test_execute_failure_releases_rows() -> None:
    pipeline = FakePipeline()
    pipeline.fail_on_step = 2
    with pytest.raises(RuntimeError, match="execute interrupted"):
        generate(pipeline, _prompts(4, 4), max_new_tokens=5)
    _assert_nothing_held(pipeline, 2)


def test_failed_claim_releases_rows_already_claimed() -> None:
    pipeline = FakePipeline()
    pipeline.kv_manager.fail_on_claim = 2
    with pytest.raises(RuntimeError, match="claim failed"):
        GenerateIterator(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_nothing_held(pipeline, 2)


def test_close_is_idempotent() -> None:
    pipeline = FakePipeline()
    batch = GenerateIterator(pipeline, _prompts(4, 4), max_new_tokens=5)
    batch.close()
    batch.close()

    assert batch.done
    _assert_nothing_held(pipeline, 2)
    with pytest.raises(RuntimeError, match="no live rows"):
        batch.step()


def test_closed_batch_drops_its_pipeline() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        batch.step()
    pipeline_ref = weakref.ref(pipeline)
    del pipeline
    gc.collect()

    assert pipeline_ref() is None
    assert _new_tokens(batch.sequences) == [[EOS], [101]]
    with pytest.raises(RuntimeError, match="no live rows"):
        batch.step()


def test_close_releases_every_row_when_one_release_fails() -> None:
    pipeline = FakePipeline()
    batch = GenerateIterator(pipeline, _prompts(4, 4, 4), max_new_tokens=5)
    pipeline.fail_next_release = True
    with pytest.raises(RuntimeError, match="release failed"):
        batch.close()

    assert pipeline.kv_manager.claimed == {}
    assert len(pipeline.released) == 2
    assert not batch.done
    batch.close()
    assert batch.done
    _assert_nothing_held(pipeline, 3)


def test_out_of_vocab_token_raises() -> None:
    pipeline = FakePipeline(tokenizer_vocab_size=102)
    with pytest.raises(RuntimeError, match="out-of-vocabulary token_id=102"):
        generate(pipeline, _prompts(NEVER), max_new_tokens=3)
    _assert_nothing_held(pipeline, 1)


@pytest.mark.parametrize("bad_id", [-1, TOKEN_ID_BOUND])
def test_out_of_range_prompt_token_raises_before_claiming(bad_id: int) -> None:
    pipeline = FakePipeline()
    with pytest.raises(
        InputError,
        match=f"^Row 1: Prompt token id {bad_id} at position 2 is out of range",
    ):
        GenerateIterator(
            pipeline, [[1, 7, 8], [1, 7, bad_id]], max_new_tokens=3
        )
    assert pipeline.kv_manager.claimed == {}
    assert pipeline.kv_manager.contexts == []


def test_highest_prompt_token_id_is_accepted() -> None:
    pipeline = FakePipeline()
    result = generate(pipeline, [[1, 7, TOKEN_ID_BOUND - 1]], max_new_tokens=3)
    assert result.sequences == [[1, 7, TOKEN_ID_BOUND - 1, EOS]]
    _assert_nothing_held(pipeline, 1)


def test_max_new_tokens_below_one_raises() -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="at least 1, was 0"):
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=0)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("max_batch_size", "data_parallel_degree", "num_rows", "max_rows"),
    [
        (2, 1, 3, "2"),
        (2, 2, 5, "4 (2 per replica x 2 replicas)"),
    ],
)
def test_oversize_batch_raises_before_claiming(
    max_batch_size: int,
    data_parallel_degree: int,
    num_rows: int,
    max_rows: str,
) -> None:
    pipeline = FakePipeline(
        max_batch_size=max_batch_size, data_parallel_degree=data_parallel_degree
    )
    with pytest.raises(ValueError, match="raise max_batch_size") as error:
        GenerateIterator(pipeline, _prompts(*[1] * num_rows), max_new_tokens=5)
    assert str(error.value).splitlines()[0] == (
        "generate() does not support scheduling across batches. "
        f"{num_rows} prompts exceed the pipeline's max batch size of "
        f"{max_rows}, and generate() never splits its prompts."
    )
    dp_flag = (
        ""
        if data_parallel_degree == 1
        else f" --data-parallel-degree {data_parallel_degree}"
    )
    _assert_points_to_serve(
        error,
        "scheduling across batches",
        f"{dp_flag} --max-batch-size {max_batch_size}",
    )
    assert pipeline.kv_manager.claimed == {}


def test_prefill_over_a_replica_token_budget_raises() -> None:
    # 12 staged tokens over 2 replicas leaves 6 per replica: two prompts.
    pipeline = FakePipeline(data_parallel_degree=2, max_batch_active_tokens=12)
    generate(pipeline, _prompts(1, 1, 1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError,
        match=r"(?s)chunked prefill\. Replica 0's prompts total 9 tokens, "
        r"over the pipeline's per-replica prefill budget of 6 tokens"
        r".*raise max_batch_input_tokens or turn chunked prefill off",
    ) as error:
        GenerateIterator(pipeline, _prompts(1, 1, 1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "chunked prefill",
        " --data-parallel-degree 2 --enable-chunked-prefill"
        " --max-batch-input-tokens 6",
    )
    _assert_nothing_held(pipeline, 4)


def test_image_generation_models_point_to_generate_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: PixelContext,
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "architecture_for_config",
        lambda pipeline_config: FakeArch(task=PipelineTask.PIXEL_GENERATION),
    )
    pipeline = FakePipeline()
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=5)
    assert str(error.value) == (
        "generate() generates text, and the model uses PixelContext. Use "
        "generate_images() for image generation models."
    )
    assert pipeline.kv_manager.claimed == {}


@dataclasses.dataclass(kw_only=True)
class FakeVideoContext(PixelContext):
    num_frames: int | None = None


def test_video_generation_models_point_to_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: FakeVideoContext,
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "architecture_for_config",
        lambda pipeline_config: FakeArch(task=PipelineTask.PIXEL_GENERATION),
    )
    pipeline = FakePipeline()
    config = pipeline.pipeline_config
    assert isinstance(config, MagicMock)
    # A diffusion pipeline has components and no main model.
    config.models = {"transformer": MagicMock(model_path=MODEL_PATH)}
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(error, "pixel generation")
    assert (
        "The model uses FakeVideoContext, and generate() generates text"
        in str(error.value)
    )
    assert pipeline.kv_manager.claimed == {}


def test_block_diffusion_models_point_to_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "architecture_for_config",
        lambda pipeline_config: FakeArch(
            name="DiffusionGemmaForBlockDiffusion",
            pipeline_cls=BlockDiffusionTextGenerationPipeline,
        ),
    )
    pipeline = FakePipeline()
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(error, "block diffusion models")
    assert "it ignores temperature, top_k and seed" in str(error.value)
    assert pipeline.kv_manager.claimed == {}


def _vision_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: TextAndVisionContext,
    )
    pipeline = FakePipeline()
    pipeline.tokenizer = FakeVisionTokenizer()
    return pipeline


def test_image_capable_model_rows_come_from_its_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    output = generate(pipeline, _prompts(1, 3), max_new_tokens=5, seed=5)

    assert _new_tokens(output.sequences) == [[EOS], [101, 102, EOS]]
    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    requests = pipeline.tokenizer.requests
    assert [request.prompt for request in requests] == _prompts(1, 3)
    contexts = pipeline.kv_manager.contexts
    assert [context.request_id for context in contexts] == [
        request.request_id for request in requests
    ]
    assert all(isinstance(c, TextAndVisionContext) for c in contexts)
    assert [c.sampling_params.seed for c in contexts] == [5, 5 + 2**32]
    assert [c.max_length for c in contexts] == [8, 8]
    _assert_nothing_held(pipeline, 2)


def test_image_capable_model_rows_build_inside_a_running_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)

    async def generate_in_loop() -> list[list[int]]:
        return generate(pipeline, _prompts(2), max_new_tokens=5).sequences

    assert _new_tokens(asyncio.run(generate_in_loop())) == [[101, EOS]]


_CALLER_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "caller_value", default="unset"
)


def test_image_capable_model_rows_see_the_callers_context_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    tokenizer = pipeline.tokenizer
    assert isinstance(tokenizer, FakeVisionTokenizer)
    seen = []
    new_context = tokenizer.new_context

    async def recording_new_context(
        request: TextGenerationRequest,
    ) -> TextAndVisionContext:
        seen.append(_CALLER_VALUE.get())
        return await new_context(request)

    monkeypatch.setattr(tokenizer, "new_context", recording_new_context)
    token = _CALLER_VALUE.set("caller")
    try:
        generate(pipeline, _prompts(2), max_new_tokens=5)
    finally:
        _CALLER_VALUE.reset(token)
    assert seen == ["caller"]


def test_image_capable_model_rows_keep_the_callers_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        generate(pipeline, _prompts(2), max_new_tokens=5)
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_tokenizer_error_inside_a_running_event_loop_claims_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    pipeline.tokenizer.error = ValueError("bad prompt")

    async def generate_in_loop() -> None:
        generate(pipeline, _prompts(2), max_new_tokens=5)

    with pytest.raises(ValueError, match=r"^Row 0: bad prompt$"):
        asyncio.run(generate_in_loop())
    assert pipeline.kv_manager.claimed == {}


def test_prompt_the_tokenizer_finds_too_long_names_its_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    pipeline.tokenizer.max_prompt_tokens = 2 * IMAGE_TOKENS
    _image_arch(monkeypatch)
    with pytest.raises(
        PromptTooLongError,
        match=rf"^Row 1: Prompt is too long: {2 * IMAGE_TOKENS + 1} tokens",
    ) as error:
        generate(
            pipeline,
            messages=[_chat("1"), _chat("1", num_images=2)],
            images=[[], [b"cat", b"dog"]],
            max_new_tokens=5,
        )
    assert error.value.num_tokens == 2 * IMAGE_TOKENS + 1
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("cap", "expected"), [(0, "0 images "), (1, "1 image ")]
)
def test_images_over_the_tokenizer_cap_are_rejected(
    monkeypatch: pytest.MonkeyPatch, cap: int, expected: str
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline.tokenizer, "max_images_per_request", cap, raising=False
    )
    _image_arch(monkeypatch)
    with pytest.raises(
        ValueError, match=rf"^Row 1 has 2 images, .* at most {expected}per row"
    ):
        generate(
            pipeline,
            messages=[_chat("1"), _chat("1", num_images=2)],
            images=[[], [b"cat", b"dog"]],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


def _image_arch(
    monkeypatch: pytest.MonkeyPatch, takes_images: bool = True
) -> None:
    arch = (
        FakeArch(
            name="VisionModel",
            input_modalities={InputModality.TEXT, InputModality.IMAGE},
        )
        if takes_images
        else FakeArch(name="TextOnlyModel")
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "architecture_for_config",
        lambda pipeline_config: arch,
    )


# The image count error with no AutoProcessor hint after it.
_NO_HINT = (
    r"^Row {row} has 1 image part but 0 images\. Pass one image per image "
    r"part, in order\.$"
)


def _chat(text: str, num_images: int = 0) -> list[dict[str, object]]:
    """One user turn holding ``num_images`` image parts, then ``text``."""
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in range(num_images)),
                {"type": "text", "text": text},
            ],
        }
    ]


def test_messages_rows_come_from_the_tokenizer() -> None:
    pipeline = FakePipeline()
    tokenizer = FakeVisionTokenizer()
    pipeline.tokenizer = tokenizer
    output = generate(
        pipeline,
        messages=[
            [{"role": "user", "content": "1 7 8"}],
            [TextGenerationRequestMessage(role="user", content="3 7")],
        ],
        max_new_tokens=5,
        seed=5,
    )

    assert output.sequences == [[1, 7, 8, EOS], [3, 7, 101, 102, EOS]]
    assert output.prompt_lengths == [3, 2]
    assert [request.prompt for request in tokenizer.requests] == [None, None]
    assert [request.messages[0].content for request in tokenizer.requests] == [
        "1 7 8",
        "3 7",
    ]
    contexts = pipeline.kv_manager.contexts
    assert [c.sampling_params.seed for c in contexts] == [5, 5 + 2**32]
    assert [c.max_length for c in contexts] == [8, 7]
    _assert_nothing_held(pipeline, 2)


def test_chat_template_kwargs_reach_every_messages_row() -> None:
    pipeline = FakePipeline()
    tokenizer = FakeVisionTokenizer()
    pipeline.tokenizer = tokenizer
    messages = [_chat("1"), _chat("3 7")]
    options = MappingProxyType({"enable_thinking": False})
    generate(pipeline, messages=messages, max_new_tokens=5)
    generate(
        pipeline,
        messages=messages,
        chat_template_kwargs=options,
        max_new_tokens=5,
    )

    requests = tokenizer.requests
    assert [r.chat_template_options for r in requests] == [
        None,
        None,
        {"enable_thinking": False},
        {"enable_thinking": False},
    ]
    # A tokenizer that edits one row's options must not reach the next row.
    assert requests[2].chat_template_options is not (
        requests[3].chat_template_options
    )


def test_image_rows_expand_their_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    output = generate(
        pipeline,
        messages=[_chat("7", num_images=2), _chat("2 7")],
        images=[[b"cat", b"dog"], []],
        max_new_tokens=5,
        output_logits=True,
        output_logprobs=1,
    )

    image_prompt = [IMAGE_TOKEN] * (2 * IMAGE_TOKENS) + [7]
    assert output.prompt_lengths == [len(image_prompt), 2]
    assert [
        sequence[:length]
        for sequence, length in zip(
            output.sequences, output.prompt_lengths, strict=True
        )
    ] == [image_prompt, [2, 7]]
    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    assert [r.images for r in pipeline.tokenizer.requests] == [
        [b"cat", b"dog"],
        [],
    ]
    new_tokens = [
        sequence[length:]
        for sequence, length in zip(
            output.sequences, output.prompt_lengths, strict=True
        )
    ]
    assert new_tokens == [[101, 102, 103, 104, 105], [101, EOS]]
    assert output.logits is not None
    assert [logits.argmax(axis=-1).tolist() for logits in output.logits] == (
        new_tokens
    )
    assert output.log_probs is not None
    assert [lp.sampled_token_ids for lp in output.log_probs] == new_tokens
    _assert_nothing_held(pipeline, 2)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pil_images_reach_the_tokenizer_without_a_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    tokenizer = pipeline.tokenizer
    assert isinstance(tokenizer, FakeVisionTokenizer)
    red = Image.new("RGB", (4, 3), "red")
    blue = _png_bytes(Image.new("RGB", (4, 3), "blue"))
    pixels: list[list[object]] = []
    new_context = tokenizer.new_context

    async def decoding_new_context(
        request: TextGenerationRequest,
    ) -> TextAndVisionContext:
        # As every vision tokenizer reads its images.
        pixels.extend(
            list(open_image(image).getdata())
            for image in request.images_for_processing()
        )
        return await new_context(request)

    monkeypatch.setattr(tokenizer, "new_context", decoding_new_context)
    opened: list[object] = []
    image_open: Callable[..., Image.Image] = Image.open

    def counting_open(*args: object, **kwargs: object) -> Image.Image:
        opened.append(args[0])
        return image_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_open)
    generate(
        pipeline,
        messages=[_chat("7", num_images=2), _chat("7", num_images=1)],
        images=[[red, blue], [blue]],
        max_new_tokens=5,
    )

    image_row, bytes_row = tokenizer.requests
    assert image_row.images == [_png_bytes(red), blue]
    assert image_row.decoded_images == [red, None]
    assert image_row.images_for_processing()[0] is red
    assert bytes_row.images == [blue]
    assert bytes_row.decoded_images == [None]
    # Only the two blue bytes entries are decoded.
    assert len(opened) == 2
    assert pixels == [
        list(red.getdata()),
        list(Image.new("RGB", (4, 3), "blue").getdata()),
        list(Image.new("RGB", (4, 3), "blue").getdata()),
    ]


@pytest.mark.parametrize(
    ("image", "mode"),
    [
        (Image.new("CMYK", (4, 3), (0, 255, 255, 0)), "RGB"),
        (Image.new("PA", (4, 3)), "RGBA"),
        (Image.new("LA", (4, 3)), "LA"),
    ],
)
def test_pil_images_encode_as_the_png_they_decode_to(
    monkeypatch: pytest.MonkeyPatch, image: Image.Image, mode: str
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    generate(
        pipeline,
        messages=[_chat("7", num_images=1)],
        images=[[image]],
        max_new_tokens=5,
    )

    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    [request] = pipeline.tokenizer.requests
    [encoded] = request.images
    [decoded] = request.decoded_images
    assert decoded is not None
    assert decoded.mode == mode
    with Image.open(io.BytesIO(encoded)) as from_bytes:
        assert from_bytes.format == "PNG"
        assert from_bytes.mode == mode
        assert list(from_bytes.getdata()) == list(decoded.getdata())
    assert list(decoded.getdata()) == list(image.convert(mode).getdata())


def test_token_rows_report_their_prompt_lengths() -> None:
    output = generate(FakePipeline(), [[1, 7, 8], [2, 7]], max_new_tokens=5)
    assert output.prompt_lengths == [3, 2]


def test_expanded_prompt_over_the_planned_max_length_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    pipeline._pipeline_model.max_seq_len = 2 * IMAGE_TOKENS
    _image_arch(monkeypatch)
    with pytest.raises(
        ValueError,
        match=rf"Row 0 needs at least {2 * IMAGE_TOKENS + 2} tokens "
        rf"\({2 * IMAGE_TOKENS + 1} prompt \+ 1 new\)",
    ):
        generate(
            pipeline,
            messages=[_chat("1", num_images=2)],
            images=[[b"cat", b"dog"]],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("start", "message"),
    [
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                _prompts(1),
                messages=[_chat("1")],
                max_new_tokens=5,
            ),
            r"^Pass input_ids or messages, not both\.$",
        ),
        (
            lambda pipeline: GenerateIterator(pipeline, max_new_tokens=5),
            r"^Pass input_ids or messages\.$",
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline, _prompts(1), images=[[b"cat"]], max_new_tokens=5
            ),
            "Pass images with messages",
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                _prompts(1),
                chat_template_kwargs={"enable_thinking": False},
                max_new_tokens=5,
            ),
            "Pass chat_template_kwargs with messages",
        ),
        *(
            (
                lambda pipeline, key=key: GenerateIterator(
                    pipeline,
                    messages=[_chat("1")],
                    chat_template_kwargs={key: None},
                    max_new_tokens=5,
                ),
                rf"^chat_template_kwargs cannot set '{key}', since the "
                "tokenizer",
            )
            for key in ("tools", "conversation")
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                attention_mask=[[1]],
                messages=[_chat("1")],
                max_new_tokens=5,
            ),
            "Pass attention_mask only with input_ids",
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                messages=[_chat("1"), _chat("1")],
                images=[[]],
                max_new_tokens=5,
            ),
            "images has 1 row, but messages has 2",
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                messages=[_chat("1"), _chat("1", num_images=2)],
                images=[[], [b"cat"]],
                max_new_tokens=5,
            ),
            "Row 1 has 2 image parts but 1 image",
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline, messages=[_chat("1", num_images=1)], max_new_tokens=5
            ),
            _NO_HINT.format(row=0),
        ),
        # A part with no image of its own gets the plain count error.
        *(
            (
                lambda pipeline, conversation=conversation: GenerateIterator(
                    pipeline, messages=[conversation], max_new_tokens=5
                ),
                _NO_HINT.format(row=0),
            )
            for conversation in (
                [{"role": "user", "content": [{"type": "image", "url": None}]}],
                [{"role": "user", "content": [ImageContentPart()]}],
                [
                    TextGenerationRequestMessage(
                        role="user", content=[ImageContentPart()]
                    )
                ],
            )
        ),
        (
            lambda pipeline: GenerateIterator(
                pipeline,
                messages=[
                    _chat("1"),
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "cat.png"},
                                }
                            ],
                        }
                    ],
                ],
                max_new_tokens=5,
            ),
            r"^Row 1 has an image_url part, which generate\(\) does not read\. "
            "Pass the conversations through AutoProcessor",
        ),
    ],
)
def test_malformed_messages_and_images_raise(
    monkeypatch: pytest.MonkeyPatch,
    start: Callable[[FakePipeline], GenerateIterator],
    message: str,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    with pytest.raises(ValueError, match=message):
        start(pipeline)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize("key", ["image", "path", "url", "base64"])
@pytest.mark.parametrize(
    "images", [None, [[], [b"dog"]]], ids=["no_images", "matching_count"]
)
def test_image_parts_holding_their_image_point_to_auto_processor(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    images: list[list[bytes]] | None,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    with pytest.raises(ValueError) as error:
        GenerateIterator(
            pipeline,
            messages=[
                _chat("1"),
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", key: "cat.png"}],
                    }
                ],
            ],
            images=images,
            max_new_tokens=5,
        )
    assert str(error.value) == (
        'Row 1 has an image part with an "image", "path", "url" or "base64" '
        "entry, which generate() does not read. Pass the conversations "
        "through AutoProcessor, which moves them into images."
    )
    assert pipeline.kv_manager.claimed == {}


class _FlatImage:
    """Stands in for a PIL image, which does not iterate."""


@pytest.mark.parametrize(
    ("messages", "images", "message"),
    [
        (_chat("1"), None, r"Wrap one conversation as \[messages\]"),
        ([_chat("1", num_images=1)], [b"cat"], "list of image lists"),
        (
            [_chat("1", num_images=2)],
            [_FlatImage(), _FlatImage()],
            "list of image lists",
        ),
    ],
)
def test_messages_or_images_of_the_wrong_shape_raise_type_error(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[list[dict[str, object]]],
    images: list[list[bytes]] | None,
    message: str,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    with pytest.raises(TypeError, match=message):
        GenerateIterator(
            pipeline, messages=messages, images=images, max_new_tokens=5
        )
    assert pipeline.kv_manager.claimed == {}


def test_messages_and_image_rows_accept_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    # The types rule generators out, but a caller without type checks can
    # pass them.
    untyped_generate: Callable[..., GenerateResult] = generate
    output = untyped_generate(
        pipeline,
        messages=(row for row in [_chat("7", num_images=1), _chat("2 7")]),
        images=(iter(row) for row in [[b"cat"], []]),
        max_new_tokens=5,
    )
    assert output.prompt_lengths == [IMAGE_TOKENS + 1, 2]
    _assert_nothing_held(pipeline, 2)


@pytest.mark.parametrize("input_ids", [_chat("1"), [_chat("1")]])
def test_conversations_passed_as_input_ids_point_to_messages(
    input_ids: list[list[int]],
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(TypeError, match="Pass conversations as messages="):
        GenerateIterator(pipeline, input_ids, max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


def test_images_on_a_text_only_architecture_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch, takes_images=False)
    with pytest.raises(
        ValueError,
        match=r"^Row 1 has 2 images, but TextOnlyModel takes text only\.$",
    ):
        GenerateIterator(
            pipeline,
            messages=[_chat("1"), _chat("1", num_images=2)],
            images=[[], [b"cat", b"dog"]],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


def test_video_parts_point_to_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    with pytest.raises(ValueError) as error:
        GenerateIterator(
            pipeline,
            messages=[
                _chat("1"),
                [{"role": "user", "content": [{"type": "video"}]}],
            ],
            max_new_tokens=5,
        )
    _assert_points_to_serve(error, "video inputs")
    assert "Row 1 has a video part" in str(error.value)
    assert pipeline.kv_manager.claimed == {}


def test_video_parts_on_a_text_only_model_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch, takes_images=False)
    with pytest.raises(
        ValueError, match=r"^Row 1 has a video part, but .* takes text only\.$"
    ):
        GenerateIterator(
            pipeline,
            messages=[
                _chat("1"),
                [{"role": "user", "content": [{"type": "video"}]}],
            ],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


def test_auto_processor_rows_build_through_the_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _vision_pipeline(monkeypatch)
    _image_arch(monkeypatch)
    inputs = _model_inputs(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b"cat"},
                        {"type": "text", "text": "7"},
                    ],
                }
            ],
            [{"role": "user", "content": "2 7"}],
        ]
    )
    with GenerateIterator(pipeline, **inputs, max_new_tokens=5) as batch:
        list(batch)

    assert [
        sequence[:length]
        for sequence, length in zip(
            batch.sequences, batch.prompt_lengths, strict=True
        )
    ] == [[IMAGE_TOKEN] * IMAGE_TOKENS + [7], [2, 7]]
    assert isinstance(pipeline.tokenizer, FakeVisionTokenizer)
    assert [r.images for r in pipeline.tokenizer.requests] == [[b"cat"], []]
    _assert_nothing_held(pipeline, 2)


def test_overlap_results_lag_one_step() -> None:
    pipeline = FakeOverlapPipeline()
    with GenerateIterator(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        assert not batch.synchronous
        steps = []
        while not batch.done:
            steps.append(batch.step())

    assert steps == [{}, {0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    # Each row runs one more forward after the one that ended it.
    assert pipeline.batch_sizes == [[2], [2], [1], [1]]
    assert _new_tokens(batch.sequences) == [[EOS], [101, 102, EOS]]
    assert pipeline.has_pending_outputs()
    _assert_nothing_held(pipeline, 2)


def test_overlap_ignore_eos_takes_one_extra_step() -> None:
    pipeline = FakeOverlapPipeline()
    output = generate(
        pipeline, _prompts(1, 3), max_new_tokens=3, ignore_eos=True
    )

    assert pipeline.batch_sizes == [[2]] * 4
    assert _new_tokens(output.sequences) == [[EOS, 102, 103], [101, 102, EOS]]
    _assert_nothing_held(pipeline, 2)


def test_overlap_pending_batch_after_close_leaves_next_batch_clean() -> None:
    pipeline = FakeOverlapPipeline()
    with GenerateIterator(pipeline, _prompts(4, 4), max_new_tokens=5) as batch:
        batch.step()
        batch.step()
    assert pipeline.has_pending_outputs()
    _assert_nothing_held(pipeline, 2)

    output = generate(pipeline, _prompts(2, 2), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[101, EOS], [101, EOS]]
    _assert_nothing_held(pipeline, 4)


@pytest.mark.parametrize("pipeline_type", [FakePipeline, FakeOverlapPipeline])
@pytest.mark.parametrize("eos_steps", [(1, 2, 3), (3, 1, 2), (2, 3, 1)])
def test_rows_finishing_in_any_order_are_released_once(
    pipeline_type: type[FakePipeline], eos_steps: tuple[int, ...]
) -> None:
    """Rows finish in any order, and late overlap outputs are dropped."""
    pipeline = pipeline_type()
    output = generate(pipeline, _prompts(*eos_steps), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [
        [*range(101, 100 + eos_step), EOS] for eos_step in eos_steps
    ]
    _assert_nothing_held(pipeline, len(eos_steps))


def _prompt_of_length(length: int) -> list[int]:
    return [NEVER] + [7] * (length - 1)


def test_batch_total_tokens_cap_charges_page_aligned_rows() -> None:
    # Pages of 4: two 3-token rows cost 4 + 3, a third pushes past 10.
    pipeline = FakePipeline(max_batch_total_tokens=10, page_size=4)
    generate(pipeline, _prompts(1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError,
        match=r"(?s)11 tokens.*caps.*10.*raise max_batch_total_tokens",
    ) as error:
        GenerateIterator(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error, "scheduling across batches", " --max-batch-total-tokens 10"
    )
    _assert_nothing_held(pipeline, 2)


def test_batch_total_tokens_cap_is_per_replica() -> None:
    pipeline = FakePipeline(
        data_parallel_degree=2, max_batch_total_tokens=10, page_size=4
    )
    generate(pipeline, _prompts(1, 1, 1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError, match="scheduling across batches\\. Replica 0's prompts"
    ):
        GenerateIterator(pipeline, _prompts(1, 1, 1, 1, 1), max_new_tokens=5)


def test_batch_total_tokens_cap_skips_a_replica_with_no_rows() -> None:
    pipeline = FakePipeline(
        data_parallel_degree=2, max_batch_total_tokens=10, page_size=4
    )
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]


def test_batch_total_tokens_cap_admits_a_lone_row_under_the_cap() -> None:
    # Page-aligned, a 9-token row would cost 12, but nothing sits after it.
    pipeline = FakePipeline(max_batch_total_tokens=10, page_size=4)
    prompt = _prompt_of_length(9)
    output = generate(pipeline, [prompt], max_new_tokens=1)
    assert output.sequences == [[*prompt, 101]]

    with pytest.raises(ValueError, match="11 tokens"):
        GenerateIterator(pipeline, [_prompt_of_length(11)], max_new_tokens=1)


def test_speculative_pipeline_runs_but_rejects_per_token_outputs() -> None:
    pipeline = FakePipeline(
        speculative=SpeculativeConfig(
            speculative_method="eagle", num_speculative_tokens=3
        ),
        draft_model_path="org/draft",
    )
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]

    with pytest.raises(ValueError, match="ignore_eos cannot keep"):
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=5, ignore_eos=True
        )
    _assert_nothing_held(pipeline, 1)

    with pytest.raises(ValueError, match="speculative decoding"):
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=5, output_logits=True
        )
    with pytest.raises(ValueError, match="speculative decoding"):
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=5, output_logprobs=1
        )
    _assert_nothing_held(pipeline, 1)


def test_lora_pipeline_points_to_serve() -> None:
    pipeline = FakePipeline(
        lora=LoRAConfig(
            enable_lora=True,
            lora_paths=["sql=/adapters/sql", "/b"],
            max_lora_rank=64,
            max_num_loras=2,
        )
    )
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "LoRA adapters",
        " --enable-lora --no-enable-prefix-caching --lora-paths"
        " sql=/adapters/sql --lora-paths /b --max-lora-rank 64"
        " --max-num-loras 2",
    )
    assert str(error.value).splitlines()[3] == (
        'Then send requests with "model": "<adapter name>". An adapter\'s '
        'name is the part of its --lora-paths entry before "=", or the '
        'whole entry if it has no "=" (names here: "sql", "/b").'
    )
    assert pipeline.kv_manager.claimed == {}


def test_lora_pipeline_without_static_adapters_points_to_load_endpoint() -> (
    None
):
    pipeline = FakePipeline(lora=LoRAConfig(enable_lora=True, lora_paths=[]))
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "LoRA adapters",
        " --enable-lora --no-enable-prefix-caching --max-lora-rank 16"
        " --max-num-loras 1",
    )
    assert str(error.value).splitlines()[3] == (
        "Load an adapter with POST /v1/load_lora_adapter, then send "
        'requests with "model": "<lora_name>".'
    )
    assert pipeline.kv_manager.claimed == {}


def _graph_capture_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "max.pipelines"
        and record.levelno == logging.WARNING
        and "warmup_graph_capture()" in record.getMessage()
    ]


def test_graph_capture_without_warmup_warns_and_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = generate(FakePipeline(), _prompts(1, 3), max_new_tokens=5)
    assert _graph_capture_warnings(caplog) == []

    pipeline = FakePipeline(device_graph_capture=True)
    output = generate(pipeline, _prompts(1, 3), max_new_tokens=5)

    assert output.sequences == expected.sequences
    assert _graph_capture_warnings(caplog) == [
        "Device graph capture is on, but warmup_graph_capture() has not "
        "run on this pipeline, so every step runs eagerly. To replay "
        "captured graphs, call pipeline.warmup_graph_capture() first, or "
        "load the model with AutoModel(..., device_graph_capture=True)."
    ]
    _assert_nothing_held(pipeline, 2)


@pytest.mark.parametrize(
    "pipeline_type", [FakeWarmedUpPipeline, FakeWarmedUpOverlapPipeline]
)
def test_warmed_up_pipeline_runs_without_warning(
    pipeline_type: type[FakeWarmedUpPipeline | FakeWarmedUpOverlapPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = pipeline_type(device_graph_capture=True)
    output = generate(pipeline, _prompts(1, 3), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[EOS], [101, 102, EOS]]
    assert _graph_capture_warnings(caplog) == []
    _assert_nothing_held(pipeline, 2)


def test_warmed_up_pipeline_with_data_parallelism_points_to_serve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakeWarmedUpPipeline(
        data_parallel_degree=2, device_graph_capture=True
    )
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "device graph capture with data parallelism",
        " --data-parallel-degree 2 --device-graph-capture",
    )
    assert str(error.value).splitlines()[3] == (
        "To run this here, load the model without device graph capture."
    )
    assert pipeline.kv_manager.contexts == []
    assert pipeline.kv_manager.claimed == {}
    assert _graph_capture_warnings(caplog) == []


def test_data_parallel_without_warmup_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakePipeline(
        max_batch_size=2, data_parallel_degree=2, device_graph_capture=True
    )
    with GenerateIterator(
        pipeline, _prompts(1, 2, 3, 3), max_new_tokens=5
    ) as batch:
        assert list(pipeline.kv_manager.claimed.values()) == [0, 1, 0, 1]
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[2, 2], [1, 2], [1, 1]]
    assert len(_graph_capture_warnings(caplog)) == 1
    _assert_nothing_held(pipeline, 4)


def test_warmed_up_with_capture_off_in_config_still_checks_dp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakeWarmedUpPipeline(data_parallel_degree=2)
    with pytest.raises(
        ValueError, match="device graph capture with data parallelism"
    ):
        GenerateIterator(pipeline, _prompts(1, 1), max_new_tokens=5)
    assert pipeline.kv_manager.contexts == []

    output = generate(FakeWarmedUpPipeline(), _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]
    assert _graph_capture_warnings(caplog) == []


@pytest.mark.parametrize(
    ("make_pipeline", "flags"),
    [
        (
            lambda: FakePipeline(
                max_batch_size=1, device_specs=[DeviceSpec.cpu()]
            ),
            " --devices cpu",
        ),
        (
            lambda: FakePipeline(
                max_batch_size=1, device_specs=[DeviceSpec.accelerator(1)]
            ),
            " --devices gpu:1",
        ),
        (
            lambda: FakePipeline(
                max_batch_size=1,
                device_specs=[
                    DeviceSpec.accelerator(0),
                    DeviceSpec.accelerator(1),
                ],
                data_parallel_degree=2,
                quantization_encoding="bfloat16",
                max_length=4096,
                max_length_user_provided=True,
            ),
            " --devices gpu:0,1 --quantization-encoding bfloat16"
            " --data-parallel-degree 2 --max-length 4096",
        ),
        # The architecture resolved this length, so serve resolves it too.
        (lambda: FakePipeline(max_batch_size=1, max_length=2048), ""),
        (
            lambda: FakePipeline(max_batch_size=1, trust_remote_code=True),
            " --trust-remote-code",
        ),
        # A bare weights repo id leaves only discovered files, so the
        # command names the repo.
        (
            lambda: FakePipeline(
                max_batch_size=1,
                weight_path=[Path("model.safetensors")],
                weights_repo_id="org/weights",
            ),
            " --weight-path org/weights",
        ),
        # Serve discovers the same default weight files.
        (
            lambda: FakePipeline(
                max_batch_size=1, weight_path=[Path("model.safetensors")]
            ),
            "",
        ),
    ],
)
def test_serve_command_repeats_the_pipeline_settings(
    make_pipeline: Callable[[], FakePipeline], flags: str
) -> None:
    pipeline = make_pipeline()
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error, "scheduling across batches", f"{flags} --max-batch-size 1"
    )


def test_serve_command_puts_feature_flags_after_pipeline_settings() -> None:
    pipeline = FakePipeline(
        device_specs=[DeviceSpec.accelerator(0), DeviceSpec.accelerator(1)],
        data_parallel_degree=2,
        quantization_encoding="float8_e4m3fn",
        max_length=8192,
        max_length_user_provided=True,
        max_batch_size=1,
    )
    with pytest.raises(ValueError) as error:
        GenerateIterator(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    assert str(error.value).splitlines()[2] == (
        "  max serve --model-path org/model --devices gpu:0,1"
        " --quantization-encoding float8_e4m3fn --data-parallel-degree 2"
        " --max-length 8192 --max-batch-size 1"
    )


@pytest.mark.parametrize(
    ("repetition", "frequency", "presence", "requested"),
    [
        (1.1, None, None, "repetition_penalty=1.1"),
        (None, 0.2, None, "frequency_penalty=0.2"),
        (None, None, 0.3, "presence_penalty=0.3"),
    ],
)
def test_penalties_without_enable_penalties_raise(
    repetition: float | None,
    frequency: float | None,
    presence: float | None,
    requested: str,
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(
        ValueError,
        match=rf"enable_penalties=False, so its sampler would ignore "
        rf"{requested}\..*sampling=SamplingConfig\(enable_penalties=True\)"
        r".*--enable-penalties",
    ) as error:
        GenerateIterator(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            repetition_penalty=repetition,
            frequency_penalty=frequency,
            presence_penalty=presence,
        )
    # GenerateIterator runs penalties on a pipeline that enables them.
    assert "does not support" not in str(error.value)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("repetition", "frequency", "message"),
    [
        (0.0, None, "repetition_penalty must be a finite"),
        (None, 5.0, r"frequency_penalty must be in \[-2.0"),
    ],
)
def test_invalid_penalty_reports_the_value_before_enable_penalties(
    repetition: float | None, frequency: float | None, message: str
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match=message):
        GenerateIterator(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            repetition_penalty=repetition,
            frequency_penalty=frequency,
        )
    assert pipeline.kv_manager.claimed == {}


def test_default_penalties_run_without_enable_penalties() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline,
        _prompts(2),
        max_new_tokens=5,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    assert _new_tokens(output.sequences) == [[101, EOS]]


@pytest.mark.parametrize(
    "input_ids", [[1, 7, 8], np.array([1, 7, 8]), [[1, 7], "ab"]]
)
def test_input_ids_that_are_not_token_lists_raise_type_error(
    input_ids: list[list[int]],
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(TypeError, match=r"Wrap one prompt as \[ids\]"):
        GenerateIterator(pipeline, input_ids, max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


def test_input_ids_accepts_a_generator() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline, (prompt for prompt in _prompts(1, 3)), max_new_tokens=5
    )
    assert _new_tokens(output.sequences) == [[EOS], [101, 102, EOS]]


def test_prompt_at_planned_max_length_raises() -> None:
    pipeline = FakePipeline(max_seq_len=3)
    with pytest.raises(
        ValueError,
        match=r"Row 0 needs at least 4 tokens.*planned max length.*"
        r"Raise max_length",
    ):
        GenerateIterator(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("max_batch_active_tokens", "max_batch_total_tokens"),
    [(8, None), (None, 8)],
)
def test_prompt_over_planned_max_length_raises_before_batch_limits(
    max_batch_active_tokens: int | None, max_batch_total_tokens: int | None
) -> None:
    # The prompt also exceeds the staged-token budget or total-token cap,
    # but only raising max_length would let it run.
    pipeline = FakePipeline(
        max_seq_len=8,
        max_batch_active_tokens=max_batch_active_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
    )
    with pytest.raises(ValueError, match="Raise max_length") as error:
        GenerateIterator(pipeline, [_prompt_of_length(9)], max_new_tokens=1)
    assert "max serve" not in str(error.value)
    assert pipeline.kv_manager.claimed == {}


def test_prompt_one_under_planned_max_length_generates_one_token() -> None:
    pipeline = FakePipeline(max_seq_len=4)
    output = generate(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101]]


def test_generation_config_min_new_tokens_counts_against_max_length() -> None:
    # 3 prompt + 1 new fits in 5, but 3 prompt + 3 required new does not.
    pipeline = FakePipeline(
        max_seq_len=5,
        generation_config=SamplingParamsGenerationConfigDefaults(
            min_new_tokens=3
        ),
    )
    with pytest.raises(
        ValueError, match=r"needs at least 6 tokens.*Raise max_length"
    ):
        GenerateIterator(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


def test_rows_are_clamped_to_the_planned_max_length() -> None:
    pipeline = FakePipeline(max_seq_len=8)
    output = generate(pipeline, _prompts(NEVER), max_new_tokens=10)

    assert _new_tokens(output.sequences) == [[101, 102, 103, 104, 105]]
    (context,) = pipeline.kv_manager.contexts
    assert context.status == GenerationStatus.MAXIMUM_LENGTH


@pytest.mark.parametrize("entry", [GenerateIterator, generate])
def test_sampling_knobs_reach_every_context(
    entry: Callable[..., object],
) -> None:
    pipeline = FakePipeline(enable_penalties=True)
    result = entry(
        pipeline,
        _prompts(1, 1),
        max_new_tokens=5,
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        min_new_tokens=2,
        stop_token_ids=[42],
    )
    if isinstance(result, GenerateIterator):
        result.close()

    assert len(pipeline.kv_manager.contexts) == 2
    for context in pipeline.kv_manager.contexts:
        params = context.sampling_params
        assert params.temperature == 0.5
        assert params.top_k == 20
        assert params.top_p == 0.9
        assert params.min_p == 0.05
        assert params.repetition_penalty == 1.1
        assert params.frequency_penalty == 0.2
        assert params.presence_penalty == 0.3
        assert params.min_new_tokens == 2
        assert params.stop_token_ids == [42]


def test_unset_sampling_knobs_fall_back_like_serve() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50, top_p=0.8, repetition_penalty=1.2
        ),
        runtime_temperature=0.0,
    )
    with GenerateIterator(pipeline, _prompts(1), max_new_tokens=5):
        pass
    with GenerateIterator(
        pipeline, _prompts(1), max_new_tokens=5, temperature=0.3
    ):
        pass

    runtime_default, caller_override = (
        context.sampling_params for context in pipeline.kv_manager.contexts
    )
    assert runtime_default.temperature == 0.0
    assert caller_override.temperature == 0.3
    assert caller_override.top_k == 50
    assert caller_override.top_p == 0.8
    assert caller_override.repetition_penalty == 1.2


def test_unset_top_k_falls_back_to_the_runtime_default() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50
        ),
        runtime_top_k=5,
    )
    with GenerateIterator(pipeline, _prompts(1), max_new_tokens=5):
        pass
    with GenerateIterator(pipeline, _prompts(1), max_new_tokens=5, top_k=20):
        pass

    runtime_default, caller_override = (
        context.sampling_params for context in pipeline.kv_manager.contexts
    )
    assert runtime_default.temperature == 0.7
    assert runtime_default.top_k == 5
    assert caller_override.top_k == 20


def test_each_row_samples_with_its_own_seed() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(
        pipeline, _prompts(1, 1, 1), max_new_tokens=5, seed=5
    ):
        pass

    seeds = [
        context.sampling_params.seed for context in pipeline.kv_manager.contexts
    ]
    assert seeds == [5, 5 + 2**32, 5 + 2 * 2**32]


def test_rows_offset_by_their_length_draw_distinct_keys() -> None:
    # 4 + 0 == 3 + 1, so seeds one apart would give both rows one key.
    pipeline = FakePipeline()
    with GenerateIterator(
        pipeline, [_prompt_of_length(4), _prompt_of_length(3)], max_new_tokens=5
    ):
        pass

    first, second = pipeline.kv_manager.contexts
    assert request_row_seed(first) != request_row_seed(second)


def test_overlap_pipeline_allows_one_open_batch() -> None:
    pipeline = FakeOverlapPipeline()
    first = GenerateIterator(pipeline, _prompts(2), max_new_tokens=5)
    with pytest.raises(RuntimeError, match="already running another batch"):
        GenerateIterator(pipeline, _prompts(2), max_new_tokens=5)
    assert len(pipeline.kv_manager.claimed) == 1

    first.close()
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]


@pytest.mark.parametrize(
    "pipeline_type", [FakePipeline, FakeOverlapDisabledPipeline]
)
def test_synchronous_pipeline_allows_concurrent_batches(
    pipeline_type: type[FakePipeline],
) -> None:
    pipeline = pipeline_type()
    with GenerateIterator(pipeline, _prompts(2), max_new_tokens=5) as first:
        with GenerateIterator(
            pipeline, _prompts(3), max_new_tokens=5
        ) as second:
            assert first.synchronous
            assert second.synchronous
            assert first.step() == {0: [101]}
            while not (first.done and second.done):
                for batch in (first, second):
                    if not batch.done:
                        batch.step()

    assert _new_tokens(first.sequences) == [[101, EOS]]
    assert _new_tokens(second.sequences) == [[101, 102, EOS]]


def test_logits_follow_each_rows_tokens_from_prefill_on() -> None:
    pipeline = FakePipeline()
    result = generate(
        pipeline, _prompts(1, 3), max_new_tokens=5, output_logits=True
    )

    assert result.log_probs is None
    assert result.logits is not None
    assert [logits.shape for logits in result.logits] == [
        (1, VOCAB),
        (3, VOCAB),
    ]
    # The prefill step's row is the last prompt position, not the decoy.
    assert [
        logits.argmax(axis=1).tolist() for logits in result.logits
    ] == _new_tokens(result.sequences)
    # Copied inside the processor, so the fake's overwrite never shows.
    assert result.logits[1].max(axis=1).tolist() == [5.0, 6.0, 7.0]
    _assert_nothing_held(pipeline, 2)


def test_logits_on_a_synchronous_overlap_pipeline() -> None:
    pipeline = FakeOverlapDisabledPipeline()
    result = generate(
        pipeline, _prompts(2), max_new_tokens=5, output_logits=True
    )

    assert result.logits is not None
    assert (
        result.logits[0].argmax(axis=1).tolist()
        == _new_tokens(result.sequences)[0]
    )


def test_log_probs_follow_each_rows_tokens() -> None:
    pipeline = FakePipeline()
    result = generate(
        pipeline, _prompts(1, 3), max_new_tokens=5, output_logprobs=1
    )

    assert result.logits is None
    assert result.log_probs is not None
    assert [lp.token_log_probabilities for lp in result.log_probs] == [
        [-1.0],
        [-1.0, -2.0, -3.0],
    ]
    assert [
        [max(top, key=top.__getitem__) for top in lp.top_log_probabilities]
        for lp in result.log_probs
    ] == _new_tokens(result.sequences)
    assert [lp.sampled_token_ids for lp in result.log_probs] == (
        _new_tokens(result.sequences)
    )
    _assert_nothing_held(pipeline, 2)


def test_ignore_eos_keeps_logits_and_log_probs_past_eos() -> None:
    pipeline = FakePipeline()
    with GenerateIterator(
        pipeline,
        _prompts(1, 3),
        max_new_tokens=5,
        ignore_eos=True,
        output_logits=True,
        output_logprobs=1,
    ) as batch:
        assert [logits.shape for logits in batch.logits] == [(0, 0)] * 2
        for _ in batch:
            pass

    new_tokens = _new_tokens(batch.sequences)
    assert pipeline.batch_sizes == [[2]] * 5
    assert new_tokens == [[EOS, 102, 103, 104, 105], [101, 102, EOS, 104, 105]]
    assert [
        logits.argmax(axis=1).tolist() for logits in batch.logits
    ] == new_tokens
    assert [lp.token_log_probabilities for lp in batch.log_probs] == [
        [-1.0, -2.0, -3.0, -4.0, -5.0]
    ] * 2
    assert [len(lp.top_log_probabilities) for lp in batch.log_probs] == [5, 5]
    assert [lp.sampled_token_ids for lp in batch.log_probs] == new_tokens


def test_removed_row_keeps_logits_and_log_probs_aligned() -> None:
    with GenerateIterator(
        FakePipeline(),
        _prompts(NEVER, NEVER),
        max_new_tokens=5,
        output_logits=True,
        output_logprobs=1,
    ) as batch:
        batch.step()
        batch.remove(1)
        while not batch.done:
            batch.step()

    new_tokens = _new_tokens(batch.sequences)
    assert [len(tokens) for tokens in new_tokens] == [5, 1]
    assert [logits.shape[0] for logits in batch.logits] == [5, 1]
    assert [lp.sampled_token_ids for lp in batch.log_probs] == new_tokens


@pytest.mark.parametrize(
    ("caller", "advice"),
    [
        ("generate()", "Use AutoModel, which turns the overlap scheduler off."),
        (
            "AutoModel",
            "Load the AutoModel without enable_overlap_scheduler=True.",
        ),
    ],
)
def test_logits_need_a_synchronous_pipeline(caller: str, advice: str) -> None:
    pipeline = FakeOverlapPipeline()
    with pytest.raises(
        ValueError, match="output_logits needs a synchronous pipeline"
    ) as error:
        GenerateIterator(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            output_logits=True,
            caller=caller,
        )
    assert str(error.value).endswith(advice)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    "pipeline_type", [FakeOverlapPipeline, FakeOverlapDisabledPipeline]
)
def test_log_probs_rejected_on_any_overlap_pipeline(
    pipeline_type: type[FakePipeline],
) -> None:
    pipeline = pipeline_type()
    with pytest.raises(ValueError) as error:
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=5, output_logprobs=1
        )
    message = str(error.value)
    assert "rejects log probabilities even with overlap off" in message
    assert "log_softmax" in message
    assert "max serve" not in message
    assert pipeline.kv_manager.claimed == {}


def test_log_probs_rejected_on_a_model_that_does_not_compute_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        FakeModel,
        "compute_log_probabilities",
        PipelineModel.compute_log_probabilities,
        raising=False,
    )
    pipeline = FakePipeline()
    with pytest.raises(
        ValueError,
        match=r"^FakeModel does not compute log probabilities\. Pass "
        r"output_logits=True",
    ):
        GenerateIterator(
            pipeline, _prompts(1), max_new_tokens=5, output_logprobs=1
        )
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize("output_logprobs", [-1, _MAX_TOP_LOGPROBS + 1])
def test_out_of_range_output_logprobs_raises(output_logprobs: int) -> None:
    pipeline = FakePipeline()
    with pytest.raises(
        ValueError,
        match=rf"in \[0, {_MAX_TOP_LOGPROBS}\], was {output_logprobs}\.",
    ):
        GenerateIterator(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            output_logprobs=output_logprobs,
        )
    assert pipeline.kv_manager.claimed == {}


def test_unrequested_logits_and_log_probs_raise() -> None:
    with GenerateIterator(
        FakePipeline(), _prompts(1), max_new_tokens=5
    ) as batch:
        with pytest.raises(RuntimeError, match="output_logits=True"):
            _ = batch.logits
        with pytest.raises(RuntimeError, match="output_logprobs > 0"):
            _ = batch.log_probs


@pytest.mark.parametrize(
    ("attribute", "output_logits", "output_logprobs", "message"),
    [
        (
            "run_logits_processors",
            True,
            0,
            "0 logits rows for 1 tokens. The pipeline did not run the "
            "logits processors",
        ),
        ("return_log_probs", False, 1, "log probabilities for 0 of 1 tokens"),
        (
            "report_sampled_tokens",
            False,
            1,
            r"log probabilities for \[0\] but generated \[101\]",
        ),
    ],
)
@pytest.mark.parametrize("image_rows", [False, True])
def test_outputs_out_of_step_with_tokens_raise(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    output_logits: bool,
    output_logprobs: int,
    message: str,
    image_rows: bool,
) -> None:
    input_ids: list[list[int]] | None = None
    messages: list[list[dict[str, object]]] | None = None
    images: list[list[bytes]] | None = None
    if image_rows:
        pipeline = _vision_pipeline(monkeypatch)
        _image_arch(monkeypatch)
        messages = [_chat("3", num_images=1)]
        images = [[b"cat"]]
    else:
        pipeline = FakePipeline()
        input_ids = _prompts(3)
    setattr(pipeline, attribute, False)
    with pytest.raises(RuntimeError, match=message):
        generate(
            pipeline,
            input_ids,
            messages=messages,
            images=images,
            max_new_tokens=5,
            output_logits=output_logits,
            output_logprobs=output_logprobs,
        )
    _assert_nothing_held(pipeline, 1)


def _built_contexts(*eos_steps: int, model_name: str = "") -> list[TextContext]:
    """Builds contexts that stop at EOS or after 4 new tokens."""
    return [
        TextContext(
            max_length=len(prompt) + 4,
            tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
            eos_tracker=EOSTracker(eos_token_ids={EOS}),
            model_name=model_name,
        )
        for prompt in _prompts(*eos_steps)
    ]


def test_from_contexts_runs_vision_contexts_to_their_own_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: TextAndVisionContext,
    )
    pipeline = FakePipeline()
    contexts = [
        TextAndVisionContext(
            max_length=len(prompt) + 4,
            tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
            eos_tracker=EOSTracker(eos_token_ids={EOS}),
            vision_token_ids=[],
        )
        for prompt in _prompts(2, NEVER)
    ]
    with GenerateIterator.from_contexts(pipeline, contexts) as batch:
        assert pipeline.kv_manager.claimed == {
            context.request_id: 0 for context in contexts
        }
        steps = []
        while not batch.done:
            steps.append(batch.step())

    assert steps == [
        {0: [101], 1: [101]},
        {0: [EOS], 1: [102]},
        {1: [103]},
        {1: [104]},
    ]
    assert _new_tokens(batch.sequences) == [[101, EOS], [101, 102, 103, 104]]
    assert [context.status for context in contexts] == [
        GenerationStatus.END_OF_SEQUENCE,
        GenerationStatus.MAXIMUM_LENGTH,
    ]
    assert pipeline.released == [context.request_id for context in contexts]
    _assert_nothing_held(pipeline, 2)


def test_from_contexts_keeps_each_contexts_sampling_params() -> None:
    pipeline = FakePipeline()
    seen: list[RequestID] = []

    def processor(inputs: ProcessorInputs) -> None:
        seen.append(inputs.context.request_id)

    # Row 1, since a per-row reseed of seed + row * 2**32 is a no-op at row 0.
    contexts = _built_contexts(1, 2)
    sampling_params = SamplingParams(seed=123, logits_processors=[processor])
    contexts[1].sampling_params = sampling_params
    with GenerateIterator.from_contexts(pipeline, contexts) as batch:
        while not batch.done:
            batch.step()

    assert contexts[1].sampling_params is sampling_params
    assert sampling_params.seed == 123
    assert sampling_params.logits_processors == [processor]
    assert seen == [contexts[1].request_id] * 2
    assert _new_tokens(batch.sequences) == [[EOS], [101, EOS]]


def test_from_contexts_fit_checks_count_context_tokens() -> None:
    pipeline = FakePipeline(max_batch_active_tokens=5)
    with pytest.raises(ValueError, match="prompts total 6 tokens"):
        GenerateIterator.from_contexts(pipeline, _built_contexts(1, 1))
    assert pipeline.kv_manager.contexts == []


def test_from_contexts_oversize_batch_raises_before_claiming() -> None:
    pipeline = FakePipeline(max_batch_size=1)
    with pytest.raises(ValueError, match="never splits"):
        GenerateIterator.from_contexts(pipeline, _built_contexts(1, 1))
    assert pipeline.kv_manager.contexts == []


def test_from_contexts_errors_name_the_caller() -> None:
    pipeline = FakePipeline(max_batch_size=1)
    with pytest.raises(ValueError) as error:
        GenerateIterator.from_contexts(
            pipeline, _built_contexts(1, 1), caller="max generate"
        )
    _assert_points_to_serve(
        error,
        "scheduling across batches",
        " --max-batch-size 1",
        caller="max generate",
    )
    assert (
        str(error.value)
        .splitlines()[0]
        .endswith(", and max generate never splits its prompts.")
    )


def test_batch_and_generate_errors_name_the_caller() -> None:
    pipeline = FakePipeline(max_batch_size=1)
    with pytest.raises(ValueError) as error:
        GenerateIterator(
            pipeline, _prompts(1, 1), max_new_tokens=5, caller="AutoPipeline"
        )
    _assert_points_to_serve(
        error,
        "scheduling across batches",
        " --max-batch-size 1",
        caller="AutoPipeline",
    )
    with pytest.raises(ValueError) as error:
        generate(
            pipeline, _prompts(1, 1), max_new_tokens=5, caller="AutoPipeline"
        )
    _assert_points_to_serve(
        error,
        "scheduling across batches",
        " --max-batch-size 1",
        caller="AutoPipeline",
    )
    assert pipeline.kv_manager.claimed == {}


def test_from_contexts_runs_a_speculative_pipeline() -> None:
    pipeline = FakePipeline(
        speculative=SpeculativeConfig(
            speculative_method="eagle", num_speculative_tokens=3
        ),
    )
    with GenerateIterator.from_contexts(pipeline, _built_contexts(2)) as batch:
        while not batch.done:
            batch.step()

    assert _new_tokens(batch.sequences) == [[101, EOS]]
    _assert_nothing_held(pipeline, 1)


def test_from_contexts_graph_capture_without_warmup_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakePipeline(device_graph_capture=True)
    with GenerateIterator.from_contexts(pipeline, _built_contexts(2)) as batch:
        while not batch.done:
            batch.step()

    assert _new_tokens(batch.sequences) == [[101, EOS]]
    assert len(_graph_capture_warnings(caplog)) == 1


def test_graph_capture_without_warmup_warns_once_per_pipeline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakePipeline(device_graph_capture=True)
    generate(pipeline, _prompts(1), max_new_tokens=5)
    with GenerateIterator.from_contexts(pipeline, _built_contexts(2)) as batch:
        while not batch.done:
            batch.step()
    assert len(_graph_capture_warnings(caplog)) == 1

    other = FakePipeline(device_graph_capture=True)
    generate(other, _prompts(1), max_new_tokens=5)
    assert len(_graph_capture_warnings(caplog)) == 2


def test_from_contexts_warmed_up_with_data_parallelism_points_to_serve() -> (
    None
):
    pipeline = FakeWarmedUpPipeline(
        data_parallel_degree=2, device_graph_capture=True
    )
    with pytest.raises(ValueError) as error:
        GenerateIterator.from_contexts(pipeline, _built_contexts(1, 1))
    _assert_points_to_serve(
        error,
        "device graph capture with data parallelism",
        " --data-parallel-degree 2 --device-graph-capture",
        caller="GenerateIterator.from_contexts()",
    )
    assert pipeline.kv_manager.contexts == []


def test_from_contexts_rejects_an_inactive_adapter() -> None:
    pipeline = FakePipeline(
        lora=LoRAConfig(enable_lora=True, lora_paths=["sql=/adapters/sql"])
    )
    pipeline._pipeline_model.lora_manager = FakeLoRAManager(
        loaded={"sql"}, active=set()
    )
    contexts = [
        *_built_contexts(1),
        *_built_contexts(1, model_name="sql"),
    ]
    with pytest.raises(ValueError) as error:
        GenerateIterator.from_contexts(pipeline, contexts)
    _assert_points_to_serve(
        error,
        "inactive LoRA adapters",
        " --enable-lora --no-enable-prefix-caching --lora-paths"
        " sql=/adapters/sql --max-lora-rank 16 --max-num-loras 1",
        caller="GenerateIterator.from_contexts()",
    )
    lines = str(error.value).splitlines()
    assert lines[0] == (
        "GenerateIterator.from_contexts() does not support inactive LoRA "
        'adapters. Row 1 names the adapter "sql", which is loaded but not '
        "active, and GenerateIterator.from_contexts() never activates an "
        "adapter."
    )
    assert lines[-1] == (
        "To run it here, activate it first with "
        "max.pipelines.lora.get_lora_manager(pipeline)"
        '.activate_adapter("sql").'
    )
    assert pipeline.kv_manager.contexts == []


def test_from_contexts_runs_an_active_adapter_and_the_base_model() -> None:
    pipeline = FakePipeline(
        lora=LoRAConfig(enable_lora=True, lora_paths=["sql=/adapters/sql"])
    )
    pipeline._pipeline_model.lora_manager = FakeLoRAManager(
        loaded={"sql"}, active={"sql"}
    )
    contexts = [
        *_built_contexts(1, model_name="base"),
        *_built_contexts(2, model_name="sql"),
    ]
    with GenerateIterator.from_contexts(pipeline, contexts) as batch:
        while not batch.done:
            batch.step()

    assert _new_tokens(batch.sequences) == [[EOS], [101, EOS]]


def test_from_contexts_shares_the_one_open_batch_limit() -> None:
    pipeline = FakeOverlapPipeline()
    first = GenerateIterator.from_contexts(pipeline, _built_contexts(2))
    with pytest.raises(RuntimeError, match="already running another batch"):
        GenerateIterator(pipeline, _prompts(2), max_new_tokens=5)
    with pytest.raises(RuntimeError, match="already running another batch"):
        GenerateIterator.from_contexts(pipeline, _built_contexts(2))
    assert len(pipeline.kv_manager.claimed) == 1

    first.close()
    with GenerateIterator(pipeline, _prompts(2), max_new_tokens=5):
        with pytest.raises(RuntimeError, match="already running another batch"):
            GenerateIterator.from_contexts(pipeline, _built_contexts(2))


def test_from_contexts_batch_records_no_logits_or_log_probs() -> None:
    contexts = _built_contexts(1)
    contexts[0].log_probabilities = 1
    with GenerateIterator.from_contexts(FakePipeline(), contexts) as batch:
        batch.step()
        with pytest.raises(
            RuntimeError, match="from_contexts records no logits"
        ):
            _ = batch.logits
        with pytest.raises(
            RuntimeError, match="from_contexts records no log probabilities"
        ):
            _ = batch.log_probs


@pytest.mark.parametrize(
    ("api", "runtime_fields", "expected"),
    [
        ("hip", {}, True),
        ("hip", {"disable_vendor_blas_fallback": False}, False),
        ("hip", {"device_graph_capture": False}, False),
        ("cuda", {}, False),
    ],
)
def test_cli_turns_the_vendor_blas_fallback_off_on_amd(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    runtime_fields: dict[str, bool],
    expected: bool,
) -> None:
    monkeypatch.delenv("MODULAR_DISABLE_VENDOR_FALLBACK", raising=False)
    monkeypatch.setattr(
        "max._entrypoints.cli.generate.accelerator_api", lambda: api
    )
    args = PipelineArgs(
        runtime=PipelineRuntimeConfig.model_validate(runtime_fields)
    )
    runtime = _with_debug_defaults(args).runtime
    assert runtime.disable_vendor_blas_fallback is expected


def test_cli_stream_cancelled_between_steps_releases_the_row() -> None:
    pipeline = FakePipeline()
    [context] = _built_contexts(NEVER)

    async def cancel_after_first_step() -> None:
        first_step = asyncio.Event()

        async def on_tokens(tokens: list[int]) -> None:
            first_step.set()

        task = asyncio.create_task(stream_context(pipeline, context, on_tokens))
        await first_step.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_after_first_step())

    assert pipeline.steps_run == 1
    _assert_nothing_held(pipeline, 1)
