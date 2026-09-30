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
"""Tests the fused MTP graph's batch inputs against its declared signature."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.pipelines.architectures.qwen3_5.model import Qwen3_5Inputs
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import (
    STATE_CACHE_KEY,
    Qwen3_5SpecShadowPools,
    attn_cache,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.batch_processor import (
    UnifiedMTPQwen3_5BatchProcessor,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.inputs import (
    UnifiedMTPQwen3_5Inputs,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.graph_input_stager import GraphInputStager
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ragged_token_descriptors,
)
from max.pipelines.speculative import RecurrentStateRollback, SpeculativeConfig
from max.tree import leaves as tree_leaves

NUM_DRAFTS = 3
HIDDEN = 32
HEAD_DIM = 16


def _config() -> Qwen3_5Config:
    devices = [DeviceRef.CPU()]
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=64,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            devices=devices,
            n_kv_heads=2,
            head_dim=HEAD_DIM,
            num_layers=1,
            page_size=HEAD_DIM,
        ),
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        mrope_section=None,
    )


def _spec_kv(config: Qwen3_5Config) -> MultiKVCacheParams:
    attn = attn_cache(config.kv_params)
    return MultiKVCacheParams.from_params(
        {"target": attn, "draft": replace(attn, num_layers=1)}
    )


def _dummy() -> Buffer:
    return Buffer.zeros(shape=[1], dtype=DType.int64)


def _inputs_matching(
    config: Qwen3_5Config, num_kv_leaves: int, *, ring: bool
) -> UnifiedMTPQwen3_5Inputs:
    """Returns inputs with one buffer per slot the graph declares."""
    return UnifiedMTPQwen3_5Inputs(
        tokens=_dummy(),
        input_row_offsets=_dummy(),
        host_input_row_offsets=_dummy(),
        return_n_logits=_dummy(),
        data_parallel_splits=_dummy(),
        signal_buffers=[_dummy()],
        batch_context_lengths=[_dummy()],
        # Only the leaf count matters.
        kv_cache_inputs=cast(
            "KVCacheInputs[Buffer, Buffer]",
            {"kv": [_dummy() for _ in range(num_kv_leaves)]},
        ),
        live_conv_pools=[_dummy()],
        live_recurrent_pools=[_dummy()],
        live_conv_row_ids=[_dummy()],
        live_recurrent_row_ids=[_dummy()],
        shadow_recurrent_pools=[] if ring else [_dummy()],
        ring_pools=[_dummy()] if ring else [],
        ring_row_ids=[_dummy()] if ring else [],
        draft_tokens=_dummy(),
        seed=_dummy(),
        temperature=_dummy(),
        top_k=_dummy(),
        max_k=_dummy(),
        top_p=_dummy(),
        min_top_p=_dummy(),
        in_thinking_phase=_dummy(),
        pinned_bitmask=_dummy(),
        wait_payload=_dummy(),
        device_bitmask_scratch=_dummy(),
        structured_output=True,
    )


@pytest.mark.parametrize("rollback", ["snapshot", "ring"])
def test_the_batch_fills_every_slot_the_graph_declares(
    rollback: RecurrentStateRollback,
) -> None:
    """Checks the batch packs one buffer per declared input."""
    config = _config()
    spec_kv = _spec_kv(config)
    driver = UnifiedMTPQwen3_5(
        config,
        speculative_config=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=NUM_DRAFTS,
            recurrent_state_rollback=rollback,
        ),
        enable_structured_output=True,
    )
    declared = driver.input_types(spec_kv)
    num_kv_leaves = len(spec_kv.flattened_kv_inputs())

    packed = _inputs_matching(
        config, num_kv_leaves, ring=(rollback == "ring")
    ).buffers

    assert len(packed) == len(declared)


def _processor() -> UnifiedMTPQwen3_5BatchProcessor:
    """Returns a stand-in with only what the methods under test use."""
    return cast(
        "UnifiedMTPQwen3_5BatchProcessor",
        SimpleNamespace(
            runtime=SimpleNamespace(devices=[CPU()]),
            _reject_image_prompts=(
                UnifiedMTPQwen3_5BatchProcessor._reject_image_prompts
            ),
            _state_tail=UnifiedMTPQwen3_5BatchProcessor._state_tail,
        ),
    )


def test_a_text_batch_is_accepted() -> None:
    ctx = TextContext(
        max_length=64, tokens=TokenBuffer(np.arange(4, dtype=np.int64) + 1)
    )

    UnifiedMTPQwen3_5BatchProcessor._reject_image_prompts(_processor(), [ctx])


def test_an_image_batch_is_refused_with_the_reason() -> None:
    """Checks a request with images is rejected."""
    ctx = cast("TextContext", SimpleNamespace(images=[object()]))

    with pytest.raises(ValueError, match="no vision encoder"):
        UnifiedMTPQwen3_5BatchProcessor._reject_image_prompts(
            _processor(), [ctx]
        )


def test_the_state_child_leaves_the_kv_slice() -> None:
    """Checks the state child is split out of the attention children."""
    state = (RecurrentStateInputsPerDevice(leaves=()),)
    tree = cast(
        "KVCacheInputs[Buffer, Buffer]",
        OrderedDict(
            [("target", ["t"]), ("draft", ["d"]), (STATE_CACHE_KEY, state)]
        ),
    )

    attention, split_state = UnifiedMTPQwen3_5BatchProcessor._state_tail(
        _processor(), tree
    )

    assert split_state == state
    # `tree.leaves` sorts a plain dict's keys, so check flatten order.
    assert tree_leaves(attention) == ["t", "d"]


def test_a_cache_without_a_state_child_is_a_loud_failure() -> None:
    """Checks a cache with no state child raises."""
    with pytest.raises(AssertionError, match="recurrent state child"):
        UnifiedMTPQwen3_5BatchProcessor._state_tail(
            _processor(),
            cast(
                "KVCacheInputs[Buffer, Buffer]",
                {"target": ["t"], "draft": ["d"]},
            ),
        )


def test_the_inputs_satisfy_the_qwen3_5_overrides() -> None:
    """Checks the inputs are the ``Qwen3_5Inputs`` the base overrides assert."""
    assert issubclass(UnifiedMTPQwen3_5Inputs, Qwen3_5Inputs)


@pytest.mark.parametrize("ring", [False, True])
def test_a_batch_stages_through_the_shared_ragged_path(ring: bool) -> None:
    """Checks a real batch reaches the graph inputs through the base stager.

    The other tests stand in for the processor, so a call that drifts from
    the base class's staging signature passes them and fails the first
    served request.
    """
    runtime = SimpleNamespace(
        devices=[CPU()],
        max_batch_active_tokens=16,
        max_global_batch_size=2,
        signal_buffers=[],
        pipeline_config=SimpleNamespace(needs_bitmask_constraints=False),
    )
    processor = object.__new__(UnifiedMTPQwen3_5BatchProcessor)
    processor.runtime = cast("BatchProcessorRuntime", runtime)
    processor._stager = GraphInputStager(
        ragged_token_descriptors(processor.runtime)
    )
    processor._batch_context_lengths = [
        Buffer.zeros(shape=[1], dtype=DType.int32)
    ]
    processor.bind_runtime_state(
        cast(
            "Qwen3_5SpecShadowPools",
            SimpleNamespace(shadow_pools=lambda _: []),
        ),
        mrope_enabled=False,
    )

    def leaf() -> RecurrentLeafInputs[Buffer, Buffer]:
        return RecurrentLeafInputs(pool=_dummy(), live_row_ids=_dummy())

    leaves = (leaf(), leaf(), leaf()) if ring else (leaf(), leaf())
    kv = cast(
        "KVCacheInputs[Buffer, Buffer]",
        OrderedDict(
            [
                ("target", [_dummy()]),
                (STATE_CACHE_KEY, (RecurrentStateInputsPerDevice(leaves),)),
            ]
        ),
    )
    batch = [
        TextContext(
            max_length=64, tokens=TokenBuffer(np.arange(n, dtype=np.int64) + 1)
        )
        for n in (4, 3)
    ]

    inputs = processor.prepare_initial_token_inputs([batch], kv)

    np.testing.assert_array_equal(
        inputs.tokens.to_numpy(), [1, 2, 3, 4, 1, 2, 3]
    )
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 4, 7]
    )
    assert len(inputs.ring_pools) == (1 if ring else 0)
