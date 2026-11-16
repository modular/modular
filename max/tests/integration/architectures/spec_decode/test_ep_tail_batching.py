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
"""What a fused spec graph stages when its target declares an EP tail.

An expert-parallel target at one attention replica (Mistral Large 4's TP8
attention beside EP8 routed experts) declares the collective buffers on an
undistributed signature, and the batch that feeds it comes from *its own*
batching rather than the spec-decode one. That batching is the target's, so
what it carries is the target's too: ``DeepseekV3Inputs`` holds the tokens, the
offsets, the logit count, the context lengths and the EP buffers, and no signal
buffers at all, because a ``ModuleV3`` trace appends those itself.

Both facts are pinned here, against a batch shaped like that one: the runtime's
signal buffers stand in for the absent field, and a batch that stages a
different number of collective buffers than the graph declares fails at
staging rather than binding the tail's slots to each other at execute.

The width comes from the graph, not from the architecture: once a pairing's
modules are built, its arch config's ``target`` is the target's *config*
(``spec_decode_pipeline_model._SpecDecodeModuleV3PipelineModel.load_model``
reassigns it from ``build_module``'s second return), and the module itself
lives only inside the fused ``Module``. So this file's config holds a config
there too -- asking the adapter's ``ep_input_types`` for the module would be
the ``AssertionError`` this staging was fixed from.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.pipelines.architectures.dspark_mistral_large4_modulev3.arch import (
    dspark_mistral_large4_modulev3_draft_arch,
)
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodeArchConfig,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

ARCH = "MistralLarge4ForConditionalGeneration_ModuleV3"
"""The adapter whose EP declarations the fused graph is built from."""

RANKS = 8
"""One node of B200s: TP8 attention, EP8 routed experts."""

AIR = Buffer.zeros(shape=[1], dtype=DType.uint8)
"""A stand-in for one of the mesh's signal buffers."""


def _ep_manager() -> EPBatchManager:
    """The collectives of the TP8/EP8/DP1 arm: 128 experts, four per token."""
    return EPBatchManager(
        EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=6144,
            top_k=4,
            n_experts=128,
            max_tokens_per_rank=1024,
            n_gpus_per_node=RANKS,
            n_nodes=1,
            use_allreduce=True,
        )
    )


def _target_config() -> Any:
    """The arch config a built pairing keeps in ``config.target``.

    A stand-in for ``MistralLarge4Config``: after ``_build_modules`` this slot
    holds the target's *config*, and the batch processor must not need the
    module it cannot reach.
    """
    return SimpleNamespace(
        ep_config=_ep_manager().config,
        hidden_size=6144,
        quant_config=None,
    )


def _config(*, ep_tail: bool, arity: int | None = None) -> SpecDecodeArchConfig:
    """A pairing's arch config, as the pipeline builds it."""
    manager = _ep_manager()
    # What the graph recorded where it declared the tail, or nothing for a
    # graph that declares none.
    recorded = (
        len(manager.input_types()) if ep_tail and arity is None else arity or 0
    )
    return SpecDecodeArchConfig(
        target=_target_config(),
        draft_architecture=dspark_mistral_large4_modulev3_draft_arch,
        kv_params=cast(Any, SimpleNamespace()),
        input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU()],
            distributed=False,
            include_signal_buffers=True,
            trace_owns_signal_buffers=True,
            ep_tail=ep_tail,
            ep_tail_arity=recorded,
        ),
    )


def _batch(ep_inputs: tuple[Buffer, ...]) -> Any:
    """A ``DeepseekV3Inputs``-shaped batch: no ``signal_buffers`` field."""
    return SimpleNamespace(
        tokens=Buffer.zeros(shape=[3], dtype=DType.int64),
        input_row_offsets=Buffer.zeros(shape=[3], dtype=DType.uint32),
        return_n_logits=Buffer.from_numpy(np.array([1], dtype=np.int64)),
        kv_cache_inputs=None,
        # The target's own batching prepares these, and only these.
        batch_context_lengths=[Buffer.zeros(shape=[1], dtype=DType.int32)],
        ep_inputs=ep_inputs,
    )


def _tail(manager: EPBatchManager) -> tuple[Buffer, ...]:
    """One staged buffer per collective the graph declares."""
    return tuple(
        Buffer.zeros(shape=[1], dtype=DType.int32)
        for _ in manager.input_types()
    )


def _processor(config: SpecDecodeArchConfig, batching: Any) -> Any:
    """The plumbing's batch processor over a stubbed pairing.

    ``__init__`` builds the target's batching, which needs a served pipeline;
    the method under test reads only what is set here.
    """
    assert not spec_decode_pipeline_model._BatchProcessor.__abstractmethods__, (
        "the processor is instantiable"
    )
    processor = cast(
        Any, object.__new__(spec_decode_pipeline_model._BatchProcessor)
    )
    processor.spec_config = config
    processor._pairing = SimpleNamespace(adapter_arch=ARCH)
    processor._method = "dspark"
    processor.runtime = SimpleNamespace(devices=[CPU()], signal_buffers=[AIR])
    processor._target_batching = batching
    processor._seed_counter = 0
    return processor


def _batching(batch: Any) -> Any:
    return SimpleNamespace(prepare_initial_token_inputs=lambda *args: batch)


def test_an_ep_tail_batch_takes_the_runtimes_signal_buffers() -> None:
    """The target's batching holds none, so the runtime's are the ones bound.

    ``DeepseekV3Inputs`` has no ``signal_buffers`` field, and a ``ModuleV3``
    trace appends the mesh's own; reading the field off the target's inputs
    raised ``AttributeError`` on the first batch of such a pairing.
    """
    manager = _ep_manager()
    config = _config(ep_tail=True)
    batch = _batch(_tail(manager))
    processor = _processor(config, _batching(batch))
    inputs = processor._target_inputs([], None, 1)
    assert inputs.signal_buffers == [AIR]
    assert not hasattr(batch, "signal_buffers")
    assert len(inputs.ep_inputs) == len(manager.input_types())


def test_the_tail_width_is_the_one_the_graph_recorded() -> None:
    """The count comes from the spec, not from the architecture.

    A pairing's arch config holds the target's *config* once its modules are
    built, so a processor that asked the adapter for the module would fail here
    with an ``AssertionError`` (and, in a served boot, on the first batch).
    This config carries a config in that slot deliberately.
    """
    manager = _ep_manager()
    config = _config(ep_tail=True)
    assert isinstance(config.target, SimpleNamespace)
    processor = _processor(config, _batching(_batch(_tail(manager))))
    inputs = processor._target_inputs([], None, 1)
    assert config.input_spec is not None
    assert config.input_spec.ep_tail_arity == len(manager.input_types())
    assert len(inputs.ep_inputs) == config.input_spec.ep_tail_arity


def test_a_batch_that_stages_a_different_tail_width_is_refused() -> None:
    """A short or long tail would land the counters on the pointer tables.

    The target's batching binds its EP initializer separately from the spec
    graph's build, so a pairing whose batching never got one stages none of the
    collectives -- and the sampling run after them would read collective state
    as its own scalars.
    """
    config = _config(ep_tail=True)
    processor = _processor(config, _batching(_batch(())))
    with pytest.raises(ValueError, match="expert-parallel buffers"):
        processor._target_inputs([], None, 1)
    # One buffer too many is the same failure.
    manager = _ep_manager()
    over = _tail(manager) + (Buffer.zeros(shape=[1], dtype=DType.int32),)
    processor = _processor(_config(ep_tail=True), _batching(_batch(over)))
    with pytest.raises(ValueError, match="expert-parallel buffers"):
        processor._target_inputs([], None, 1)


def test_a_batch_without_an_ep_tail_stages_no_collectives() -> None:
    """A build without expert parallelism takes none of them, and says so."""
    config = _config(ep_tail=False, arity=0)
    processor = _processor(config, _batching(_batch(())))
    inputs = processor._target_inputs([], None, 1)
    assert inputs.ep_inputs == ()
    assert inputs.signal_buffers == [AIR]


def test_the_tail_width_a_non_ep_spec_records_is_zero() -> None:
    """The width is only recorded for a tail that is declared."""
    spec = spec_decode_pipeline_model._input_type_spec(
        cast(
            Any,
            SimpleNamespace(model=SimpleNamespace(data_parallel_degree=1)),
        ),
        [DeviceRef.CPU()],
        ARCH,
        # A target whose build bound no EP initializer declares no tail...
        SimpleNamespace(ep_batch_manager=None),
        distributed=False,
        include_signal_buffers=True,
        trace_owns_signal_buffers=True,
        signal_buffer_index=None,
        thinking=False,
        vision=False,
        ep_tail_arity=RANKS,
    )
    assert not spec.ep_tail
    # ... so the caller's count is dropped rather than recorded against it.
    assert spec.ep_tail_arity == 0


def test_the_manager_declares_the_slots_this_file_stages() -> None:
    """The 22 slots are the manager's own: two counter groups and three tables."""
    entries = _ep_manager().input_types()
    assert len(entries) == 2 * RANKS + 3 * 2
    dtypes = [str(entry.dtype) for entry in entries]
    assert dtypes.count("DType.int32") == 2 * RANKS
    assert dtypes.count("DType.uint64") == 3 * 2
