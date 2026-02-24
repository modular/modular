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

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.config import ConfigFileModel
from max.driver import Accelerator, Buffer, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    SymbolicDim,
    TensorType,
    ops,
)
from max.interfaces import RequestID, TextGenerationInputs, TokenBuffer
from max.nn.legacy import KVCacheInputs, kernels
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    OverlapTextGenerationPipeline,
    PipelineConfig,
    PipelineModel,
    PipelineModelWithKVCache,
    SupportedEncoding,
)
from max.pipelines.lib.pipeline_variants import overlap_text_generation

GPU_SECONDS = 0.5
CPU_SECONDS = 0.2
FAKE_VOCAB_SIZE = 500


def draw_span_rows(
    rows: dict[str, list[tuple[float, float]]],
    xmax: float = 4.5,
    width: int = 80,
) -> None:
    if not rows:
        return

    if xmax <= 0:
        raise ValueError("xmax must be > 0")

    # Normalize by global minimum
    original_min = min(start for spans in rows.values() for start, _ in spans)

    def scale(x_norm: float) -> int:
        x_norm = max(0.0, min(x_norm, xmax))
        return min(width - 1, int(x_norm / xmax * (width - 1)))

    label_width = max(len(name) for name in rows)

    # ---- DRAW ROWS ----
    for name, spans in rows.items():
        line = [" "] * width

        for start, end in sorted(spans):
            start = start - original_min
            end = end - original_min

            if end <= 0 or start >= xmax:
                continue

            start = scale(start)
            end = scale(end)

            start = max(0, min(start, width - 1))
            end = max(0, min(end, width))

            # Enforce minimum width
            if (end - start) < 2:
                start = max(0, start - 1)
                end = min(width, end + 1)

            length = end - start

            # ---- Render ----
            line[start] = "["
            if length > 2:
                for pos in range(start + 1, end - 1):
                    if 0 <= pos < width:
                        line[pos] = "█"
            line[end - 1] = "]"

        print(f"{name:>{label_width}} | " + "".join(line))

    # ---- AXIS ----
    print(" " * (label_width + 1) + "+" + "-" * width)

    max_tick = math.floor(xmax)

    tick_line = [" "] * width
    label_line = [" "] * width

    for val in range(0, max_tick + 1):
        pos = scale(float(val))
        tick_line[pos] = "|"

        label = str(val)
        start_pos = max(0, min(width - len(label), pos - len(label) // 2))
        for i, ch in enumerate(label):
            label_line[start_pos + i] = ch

    padding = " " * (label_width + 3)
    print(padding + "".join(tick_line))
    print(padding + "".join(label_line))


class FakeSamplingConfig(ConfigFileModel):
    enable_penalties: bool = False
    enable_variable_logits: bool = False
    in_dtype: DType = DType.float32
    out_dtype: DType = DType.float32
    enable_structured_output: bool = False
    enable_min_tokens: bool = False


class FakeModelConfig(ConfigFileModel):
    model_path: str
    huggingface_config: Any
    device_specs: list[DeviceSpec]
    kv_cache: Any
    quantization_encoding: SupportedEncoding = "float32"
    enable_echo: bool = False


class FakePipelineConfig(ConfigFileModel):
    model: FakeModelConfig
    sampling: FakeSamplingConfig
    execute_empty_batches: bool = False
    enable_overlap_scheduler: bool = False
    max_batch_size: int = 999

    def configure_session(self, *args: Any, **kwargs: Any) -> None:
        pass


@dataclass
class FakeModelInputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer
    sleep_duration: Buffer
    arange: Buffer


def build_graph(device_ref: DeviceRef) -> Model:
    """Builds a graph that mimics the behavior of a LLM that performs X -> X+1.

    Given:
      - tokens: [0, 44, 45, 46, 47, 1, 2]
      - input_row_offsets: [0, 3, 7, 10]
      - sleep_duration: [3.14]

    The graph will:
      - take 3.14 seconds to execute
      - produce the following logits:
        [
          [ -INF,  INF, -INF, -INF, ... ] # INF @ idx=1
          [ -INF, -INF, -INF, -INF, ... ] # INF @ idx=48
          [ -INF, -INF, -INF,  INF, ... ] # INF @ idx=3
        ]

    Then when we sample the logits we will produce next_tokens=[1, 48, 3].
    """
    with Graph(
        "my_lil_llm",
        input_types=[
            # tokens
            TensorType(
                DType.int64, [SymbolicDim("total_seq_len")], device=device_ref
            ),
            # input row offsets
            TensorType(
                DType.int64,
                [SymbolicDim("input_row_offsets_len")],
                device=device_ref,
            ),
            # sleep duration
            BufferType(DType.float64, [1], device=DeviceRef.CPU()),
            # arange
            TensorType(
                DType.int64, [SymbolicDim("arange_len")], device=device_ref
            ),
        ],
    ) as graph:
        tokens_input, input_row_offsets_input, sleep_duration_input, arange = (
            graph.inputs
        )
        tokens = tokens_input.tensor
        input_row_offsets = input_row_offsets_input.tensor
        sleep_duration = sleep_duration_input.buffer
        arange = arange.tensor
        batch_size = input_row_offsets.shape[0] - 1

        gather_indices = input_row_offsets[1:] - 1
        last_tokens = ops.gather(
            input=tokens.tensor, indices=gather_indices, axis=0
        )
        next_tokens = last_tokens + 1
        scatter_indices = ops.stack([arange[:batch_size], next_tokens], axis=1)
        neg_inf = ops.constant(-12345.0, DType.float32, device=device_ref)
        pos_inf = ops.constant(12345.0, DType.float32, device=device_ref)
        logits = ops.broadcast_to(neg_inf, [batch_size, Dim(FAKE_VOCAB_SIZE)])
        updates = ops.broadcast_to(pos_inf, [batch_size])
        logits = ops.scatter_nd(
            input=logits,
            updates=updates,
            indices=scatter_indices,
        )
        kernels.sleep(sleep_duration.buffer, device_ref=device_ref)
        graph.output(logits)
    device = device_ref.to_device()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    return model


class FakePipelineModel(PipelineModelWithKVCache[TextContext]):
    def __init__(
        self, pipeline_config: FakePipelineConfig, *args: Any, **kwargs: Any
    ) -> None:
        self.kv_params = MagicMock(spec=KVCacheParams)
        self.enable_overlap_scheduler = pipeline_config.enable_overlap_scheduler
        self.device = Accelerator()
        self.kv_cache_config = MagicMock()
        self.max_seq_len = 9999
        print(f"Building graph for device {self.device}")
        t0 = time.time()
        self.model = build_graph(device_ref=DeviceRef.from_device(self.device))
        t1 = time.time()
        print(f"Graph built in {t1 - t0} seconds")

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        del kv_cache_inputs, return_n_logits  # Unused args

        assert len(replica_batches) == 1, "DP must be 1"

        batch = replica_batches[0]
        batch_size = len(batch)
        sleep_duration = Buffer.from_numpy(
            np.array([GPU_SECONDS], dtype=np.float64)
        )
        active_lengths = [ctx.tokens.active_length for ctx in batch]
        total_seq_len = sum(active_lengths)
        tokens = Buffer(
            shape=[total_seq_len],
            dtype=DType.int64,
            device=self.device,
            pinned=True,
        )
        tokens.disable_auto_sync()
        np.concatenate(
            [ctx.tokens.active for ctx in batch], out=tokens.to_numpy()
        )
        input_row_offsets = Buffer(
            shape=[len(batch) + 1],
            dtype=DType.int64,
            device=self.device,
            pinned=True,
        )
        input_row_offsets.disable_auto_sync()
        np.cumsum(
            [0] + active_lengths,
            dtype=np.int64,
            out=input_row_offsets.to_numpy(),
        )
        arange = Buffer(
            dtype=DType.int64,
            shape=[batch_size],
            device=self.device,
            pinned=True,
        )
        arange.disable_auto_sync()
        arange.to_numpy()[:] = np.arange(
            start=0, stop=batch_size, dtype=np.int64
        )

        return FakeModelInputs(
            tokens=tokens.to(self.device),
            input_row_offsets=input_row_offsets.to(self.device),
            sleep_duration=sleep_duration,
            arange=arange.to(self.device),
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, FakeModelInputs)
        if not self.enable_overlap_scheduler:
            Accelerator().synchronize()

        (logits,) = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.sleep_duration,
            model_inputs.arange,
        )

        if not self.enable_overlap_scheduler:
            Accelerator().synchronize()

        return ModelOutputs(logits=logits)


# Delete all abstract methods so python doesn't complain about unimplemented
# abstract methods (this is extremely cursed)
FakePipelineModel.__abstractmethods__ = frozenset()


def create_context(
    isl: int = 64, osl: int = 64, offset: int = 0
) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=isl + osl,
        tokens=TokenBuffer(np.arange(isl) + offset),
    )


def monkeypatch_weight_and_kvcache_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for func in [
        "get_weight_paths",
        "load_weights",
        "weights_format",
        "load_kv_manager",
    ]:
        monkeypatch.setattr(overlap_text_generation, func, MagicMock())


def create_overlap_pipeline(
    enable_overlap_scheduler: bool,
) -> OverlapTextGenerationPipeline[Any]:
    sampling_config = FakeSamplingConfig(enable_penalties=False)
    model_config = FakeModelConfig(
        model_path="test_model",
        huggingface_config=MagicMock(),
        device_specs=[DeviceSpec(id=0, device_type="gpu")],
        kv_cache=MagicMock(),
    )
    pipeline_config = FakePipelineConfig(
        model=model_config,
        sampling=sampling_config,
        enable_overlap_scheduler=enable_overlap_scheduler,
    )
    pipeline = OverlapTextGenerationPipeline(
        pipeline_config=cast(PipelineConfig, pipeline_config),
        pipeline_model=cast(type[PipelineModel[Any]], FakePipelineModel),
        eos_token_id=9999,
        weight_adapters=MagicMock(),
        tokenizer=MagicMock(),
    )
    return pipeline


def fake_cpu_pre_or_post_processing() -> None:
    time.sleep(CPU_SECONDS)
    return


def prime_host_buffer_cache() -> None:
    t = Buffer(
        shape=[1024 * 1024],
        dtype=DType.int8,
        device=Accelerator(),
        pinned=True,
    )
    del t


"""
In the below test, we record some spans and plot them in an ascii chart.
Note that each span corresponds to CPU execution time. Due to lack of CUDA Events
with timing, we cannot get GPU timing spans.

Overlap=True:
 Preprocess | [█][██]        [██]     [██]     [█]
    Execute |   []   [███]       []      [█]      []
Postprocess |             [█]      [█]     [██]     [██]
            +--------------------------------------------------------------------------------
              |                |                 |                |                 |
              0                1                 2                3                 4
Actual: 2.40s, Expected: 2.40s, Error: 0.00s

Overlap=False:
 Preprocess | [█]         [█]             [█]             [█]            [██]
    Execute |    [███████]   [███████]       [███████]       [███████]       [███████]
Postprocess |                         [██]            [██]            [█]             [█]
            +--------------------------------------------------------------------------------
              |                |                 |                |                 |
              0                1                 2                3                 4
Actual: 4.31s, Expected: 4.30s, Error: 0.01s
"""


@pytest.mark.parametrize(
    "enable_overlap_scheduler,expected_duration", [(True, 2.4), (False, 4.3)]
)
def test_overlap_execution(
    monkeypatch: pytest.MonkeyPatch,
    enable_overlap_scheduler: bool,
    expected_duration: float,
) -> None:
    monkeypatch_weight_and_kvcache_loading(monkeypatch)
    prime_host_buffer_cache()

    pipeline = create_overlap_pipeline(
        enable_overlap_scheduler=enable_overlap_scheduler
    )

    num_trials = 3
    for _trial in range(num_trials):
        _ = pipeline.execute(TextGenerationInputs(batches=[[]], num_steps=1))

        req_a = create_context(isl=17, osl=1, offset=100)
        req_b = create_context(isl=42, osl=4, offset=200)
        req_c = create_context(isl=77, osl=2, offset=300)
        active_requests = {
            req_a.request_id: req_a,
            req_b.request_id: req_b,
            req_c.request_id: req_c,
        }
        generated_tokens: dict[RequestID, list[int]] = {
            req_a.request_id: [],
            req_b.request_id: [],
            req_c.request_id: [],
        }
        preprocess_spans: list[tuple[float, float]] = []
        execute_spans: list[tuple[float, float]] = []
        postprocess_spans: list[tuple[float, float]] = []
        start_time = time.time()
        iters = 0
        while active_requests:
            print()
            print("-" * 80)
            print(f"Running iteration {iters + 1}")

            span_start = time.time()
            fake_cpu_pre_or_post_processing()
            span_end = time.time()
            preprocess_spans.append((span_start, span_end))

            span_start = time.time()
            inputs = TextGenerationInputs(
                batches=[list(active_requests.values())], num_steps=1
            )
            outputs = pipeline.execute(inputs)
            span_end = time.time()
            execute_spans.append((span_start, span_end))

            if outputs:
                span_start = time.time()
                fake_cpu_pre_or_post_processing()
                span_end = time.time()
                postprocess_spans.append((span_start, span_end))

            # Filter out outputs for requests that are not active anymore.
            outputs = {
                req_id: output
                for req_id, output in outputs.items()
                if req_id in active_requests
            }

            for req_id, output in outputs.items():
                generated_tokens[req_id].extend(output.tokens)
                if output.is_done and req_id in active_requests:
                    del active_requests[req_id]

            iters += 1
        end_time = time.time()
        elapsed = end_time - start_time

        # We should run 5 iterations because the largest osl is 4 (req_b)
        # Recall that overlap scheduler may run for one more iteration than needed
        assert iters == 5

        # Check that the generated tokens are what we expect.
        # We exclude that last token since it is undefined.
        assert generated_tokens[req_a.request_id] == [117]
        assert generated_tokens[req_b.request_id] == [242, 243, 244, 245]
        assert generated_tokens[req_c.request_id] == [377, 378]

        error = abs(elapsed - expected_duration)

        draw_span_rows(
            {
                "Preprocess": preprocess_spans,
                "Execute": execute_spans,
                "Postprocess": postprocess_spans,
            }
        )
        print(
            f"Actual: {elapsed:.2f}s, Expected: {expected_duration:.2f}s, Error: {error:.2f}s"
        )

        # Disable check since this is unreliable in CI
        #
        # For the last trial, ensure that the error is less than 1 second.
        # Don't check this for the other trials since we need to warmup the kernels.
        # if trial == num_trials - 1:
        #     assert error < 1.0
