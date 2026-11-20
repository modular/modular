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
"""Runs a tiny random LFM2 over several requests, through prefill and decode.

Each request's logits in a shared batch must match its logits run alone, so
a convolution state that lands in another request's row, or that a step fails
to carry forward, changes the result.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from unittest.mock import NonCallableMock

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.lfm2 import LFM2, LFM2Config
from max.pipelines.architectures.lfm2.model_config import (
    ATTN_CACHE_KEY,
    CONV_LEAF_ID,
    STATE_CACHE_KEY,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import JengaKVCacheManager
from max.pipelines.lib import GraphRole, GraphSpec, compile_and_init_graphs
from max.pipelines.lib.interfaces import BatchProcessorRuntime
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderBatchProcessor,
    _DecoderGraph,
    graph_api_decoder_input_types,
)
from test_common.context_utils import create_text_context

_CPU = DeviceRef.CPU()
_HIDDEN = 32
_CONV_KERNEL = 3
_MAX_BATCH_SIZE = 2
_FIRST_DECODE_TOKEN = 40
_PROMPTS = {
    "a": [1, 2, 3],
    "b": [4, 5, 6, 7, 8],
    "c": [9, 10],
}

# Each step's batch, in its order: a and b prefill together, decode in both
# orders, then a finishes and c prefills beside b's decode in the state rows
# a freed, which the pool hands out first.
_SCHEDULE = [["a", "b"], ["b", "a"], ["a", "b"], ["b", "c"], ["c", "b"]]


def _kv_params() -> MultiKVCacheParams:
    attn = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        devices=[_CPU],
    )
    state = RecurrentStateParams(
        regions=(
            RecurrentStateRegion(
                leaf_id=CONV_LEAF_ID,
                num_layers=2,
                row_shape=(_HIDDEN, _CONV_KERNEL - 1),
                dtype=DType.float32,
            ),
        ),
        devices=[_CPU],
    )
    return MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
    )


def _config() -> LFM2Config:
    return LFM2Config(
        hidden_size=_HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=3,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=64,
        intermediate_size=64,
        interleaved_rope_weights=False,
        vocab_size=48,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=_kv_params(),
        attention_multiplier=0.25,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        rms_norm_eps=1e-5,
        clip_qkv=None,
        devices=[_CPU],
        layer_types=["conv", "full_attention", "conv"],
        conv_L_cache=_CONV_KERNEL,
    )


def _compile(session: InferenceSession, config: LFM2Config) -> Model:
    decoder = LFM2(config)
    rng = np.random.default_rng(0)
    decoder.load_state_dict(
        {
            name: rng.normal(0.0, 0.2, weight.shape.static_dims).astype(
                np.float32
            )
            for name, weight in decoder.raw_state_dict().items()
        },
        weight_alignment=1,
    )
    input_types = graph_api_decoder_input_types(config.kv_params, [_CPU])
    compiled = compile_and_init_graphs(
        session,
        {
            GraphRole.DECODER: GraphSpec(
                _DecoderGraph(decoder, input_types, ReturnHiddenStates.NONE),
                "lfm2",
                input_types,
                weights=decoder.state_dict(auto_initialize=False),
            )
        },
    )
    return compiled[GraphRole.DECODER].engine_model


class _Server:
    """Steps batches through the decoder as the registered pipeline does."""

    def __init__(self, model: Model, config: LFM2Config) -> None:
        self._model = model
        self._kv = JengaKVCacheManager.create(
            params=config.kv_params,
            available_bytes=1 << 20,
            max_batch_size=_MAX_BATCH_SIZE,
            max_seq_len=config.max_seq_len,
        )
        self._processor = GraphApiDecoderBatchProcessor(
            config,
            BatchProcessorRuntime(
                pipeline_config=NonCallableMock(),
                devices=[CPU()],
                return_logits=ReturnLogits.LAST_TOKEN,
                max_batch_size=_MAX_BATCH_SIZE,
                max_seq_len=config.max_seq_len,
                max_batch_input_tokens=64,
            ),
        )
        self._processor.bind_input_tree(
            OrderedDict[str, object](
                graph_api_decoder_input_types(config.kv_params, [_CPU])
            )
        )
        self._steps: dict[str, int] = {}

    def step(self, batch: Mapping[str, TextContext]) -> dict[str, np.ndarray]:
        """Runs one step and feeds each request its next token.

        Returns:
            Each request's next-token logits.
        """
        contexts = list(batch.values())
        for ctx in contexts:
            if not self._kv.contains(ctx):
                self._kv.claim(ctx)
            self._kv.alloc(ctx)
        inputs = self._processor.prepare_initial_token_inputs(
            [contexts], kv_cache_inputs=self._kv.runtime_inputs([contexts])
        )
        (logits,) = self._model.execute(*inputs.buffers)
        assert isinstance(logits, Buffer)
        rows = logits.to_numpy()
        # Every request decodes the same tokens whichever batch it is in.
        for name, ctx in batch.items():
            step = self._steps.get(name, 0)
            ctx.update(_FIRST_DECODE_TOKEN + step)
            self._kv.step(ctx)
            self._steps[name] = step + 1
        return dict(zip(batch, rows, strict=True))

    def release(self, ctx: TextContext) -> None:
        self._kv.release(ctx)


def _contexts(names: Sequence[str]) -> dict[str, TextContext]:
    return {
        name: create_text_context(np.array(_PROMPTS[name]), max_length=64)
        for name in names
    }


def test_batched_requests_match_running_alone() -> None:
    config = _config()
    session = InferenceSession(devices=[CPU()])
    model = _compile(session, config)

    batched: dict[str, list[np.ndarray]] = {name: [] for name in _PROMPTS}
    server = _Server(model, config)
    contexts = _contexts(list(_PROMPTS))
    for step, names in enumerate(_SCHEDULE):
        logits = server.step({name: contexts[name] for name in names})
        for name, row in logits.items():
            batched[name].append(row)
        if step == 2:
            server.release(contexts["a"])

    for name, steps in batched.items():
        alone = _Server(model, config)
        (ctx,) = _contexts([name]).values()
        for step, row in enumerate(steps):
            np.testing.assert_allclose(
                row,
                alone.step({name: ctx})[name],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"request {name!r} at step {step}",
            )
