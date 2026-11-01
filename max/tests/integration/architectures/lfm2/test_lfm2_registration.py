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
"""Tests LFM2's registration and its per-request convolution states."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.export import batch_processor_dims, per_step_dims
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.kv_cache import MHAKVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.lfm2 import LFM2, ConvStateStore, lfm2_arch
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportsSSMStateWarmup
from max.pipelines.lib.interfaces import BatchProcessorRuntime
from max.pipelines.lib.module_registration import bind_state_store
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderBatchProcessor,
    GraphApiDecoderPipelineModel,
    graph_api_decoder_input_types,
)
from max.pipelines.modeling.types import RequestID
from max.tensor.sharding.types import as_layout

_HIDDEN = 4
_KERNEL = 3


def _context() -> TextContext:
    return NonCallableMock(spec=TextContext, request_id=RequestID())


def _config() -> Any:
    return SimpleNamespace(
        layer_types=["conv", "full_attention", "conv"],
        hidden_size=_HIDDEN,
        conv_L_cache=_KERNEL,
        dtype=DType.float32,
    )


def _store(max_slots: int) -> ConvStateStore:
    return ConvStateStore(
        num_conv_layers=2,
        hidden_size=_HIDDEN,
        conv_kernel=_KERNEL,
        dtype=DType.float32,
        max_slots=max_slots,
        device=CPU(),
    )


def _layer_states(*rows: float) -> list[Buffer]:
    """Each conv layer's states for a batch, one row value per request.

    Layer ``i`` adds ``100 * i`` so a swapped layer shows as well as a
    swapped row.
    """
    return [
        Buffer.from_numpy(
            np.stack(
                [np.full((_HIDDEN, _KERNEL), row + 100 * layer) for row in rows]
            ).astype(np.float32)
        )
        for layer in range(2)
    ]


def _assert_rows(states: tuple[Buffer, ...], *rows: float | None) -> None:
    """Checks each request's row, where ``None`` is a fresh, zeroed slot."""
    for layer, state in enumerate(states):
        np.testing.assert_array_equal(
            state.to_numpy(),
            np.stack(
                [
                    np.full(
                        (_HIDDEN, _KERNEL),
                        0 if row is None else row + 100 * layer,
                    )
                    for row in rows
                ]
            ),
        )


def test_lfm2_registers_its_decoder_with_conv_state() -> None:
    model = lfm2_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert issubclass(model, SupportsSSMStateWarmup)
    assert lfm2_arch.modules == {GraphRole.DECODER: LFM2}
    assert LFM2.graph_name == "lfm2"


def test_the_conv_state_has_one_bounded_input_per_conv_layer() -> None:
    store = LFM2.state_store(_config(), max_batch_size=2, devices=[CPU()])
    assert isinstance(store, ConvStateStore)
    assert (
        store.input_types()
        == (
            TensorType(
                DType.float32,
                ["conv_batch", _HIDDEN, _KERNEL],
                device=DeviceRef.CPU(),
            ),
        )
        * 2
    )
    assert store.max_dims() == {"conv_batch": 2}


def test_the_conv_state_keeps_each_requests_row() -> None:
    store = _store(max_slots=2)
    first, second = _context(), _context()
    _assert_rows(store.prepare([first, second]), None, None)

    store.update([first.request_id, second.request_id], _layer_states(1, 2))
    # The next step lists the requests in the other order.
    _assert_rows(store.prepare([second, first]), 2, 1)
    store.update([second.request_id, first.request_id], _layer_states(20, 10))
    _assert_rows(store.prepare([first]), 10)
    _assert_rows(store.prepare([second]), 20)

    # Releasing a request frees its slot for a new one, which starts clean.
    store.release(first.request_id)
    third = _context()
    _assert_rows(store.prepare([second, third]), 20, None)
    with pytest.raises(RuntimeError, match="No free"):
        store.prepare([_context()])

    store.release_warmup([second.request_id, third.request_id])
    _assert_rows(store.prepare([_context(), _context()]), None, None)


def test_the_conv_state_checks_the_states_a_step_returns() -> None:
    store = _store(max_slots=2)
    first, second = _context(), _context()
    store.prepare([first, second])
    request_ids = [first.request_id, second.request_id]
    with pytest.raises(ValueError, match="expected 2 conv states"):
        store.update(request_ids, [])
    with pytest.raises(ValueError, match="expected 2 conv states"):
        store.update(request_ids, _layer_states(1, 2)[:1])
    with pytest.raises(ValueError, match="shape"):
        store.update(request_ids, _layer_states(1))


def test_lfm2s_input_tree_bounds_every_per_step_dim() -> None:
    """Artifact export bounds each per-step dim of the bound input tree."""
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        devices=[DeviceRef.CPU()],
    )
    runtime = BatchProcessorRuntime(
        pipeline_config=NonCallableMock(),
        devices=[],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=2,
        max_seq_len=64,
        max_batch_input_tokens=32,
    )
    processor = GraphApiDecoderBatchProcessor(_config(), runtime)
    store = bind_state_store(
        LFM2, _config(), processor, max_batch_size=2, devices=[CPU()]
    )
    assert store is not None
    input_tree = OrderedDict[str, object](
        graph_api_decoder_input_types(
            kv_params, [DeviceRef.CPU()], state=store.input_types()
        )
    )
    processor.bind_input_tree(input_tree)
    layouts = [as_layout(t) for t in tree.leaves(input_tree)]
    model: Any = SimpleNamespace(batch_processor=processor, kv_params=kv_params)
    decoder: Any = SimpleNamespace(in_layouts=layouts)

    bounds = batch_processor_dims(model, decoder)
    assert isinstance(bounds, BatchProcessorDims)
    assert list(bounds.dims) == per_step_dims(layouts)
    assert bounds.dims["conv_batch"] == 2
    assert bounds.dims["total_seq_len"] == runtime.max_batch_active_tokens
