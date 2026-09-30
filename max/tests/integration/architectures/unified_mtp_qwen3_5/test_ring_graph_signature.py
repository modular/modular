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
"""Tests the fused MTP graph's state-tail signature on each rollback.

An engine driving the exported graph binds this tail by position, so the
order and the arity of each rollback are pinned here.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    recurrent_leaf,
)
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import (
    COMPILED_RING_LENS,
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    RING_LEAF_ID,
    STATE_CACHE_KEY,
    attn_cache,
    linear_state_regions,
    ring_len_for_window,
    shadowed_leaf_ids,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.model import (
    UnifiedMTPQwen3_5Model,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.model_config import (
    UnifiedMTPQwen3_5Config,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.spec_state import (
    POSITION_IDS,
    graph_kv_params,
    state_tail,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.speculative import RecurrentStateRollback, SpeculativeConfig

NUM_DRAFTS = 3


HIDDEN = 32
HEAD_DIM = 16


def _config(num_devices: int) -> Qwen3_5Config:
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=devices,
        n_kv_heads=2,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
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
        kv_params=kv_params,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        # The only head shape the gated-delta kernels are compiled for.
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
    )


def _input_types(
    rollback: RecurrentStateRollback, num_devices: int = 1
) -> tuple[TensorType | BufferType, ...]:
    config = _config(num_devices)
    attn = attn_cache(config.kv_params)
    spec_kv = MultiKVCacheParams.from_params(
        {"target": attn, "draft": replace(attn, num_layers=1)}
    )
    driver = UnifiedMTPQwen3_5(
        config,
        speculative_config=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=NUM_DRAFTS,
            recurrent_state_rollback=rollback,
        ),
    )
    return tuple(driver.input_types(spec_kv))


def _tail_dims(
    types: tuple[TensorType | BufferType, ...],
) -> tuple[list[str], list[str]]:
    """Returns the state tail's pool row dims and row-table layer dims."""
    pools = [
        str(t.shape[0])
        for t in types
        if isinstance(t, BufferType) and str(t.shape[0]).endswith("_rows")
    ]
    tables = [
        f"{t.shape[0]}"
        for t in types
        if isinstance(t, TensorType)
        and t.dtype == DType.uint32
        and len(t.shape) == 2
        and str(t.shape[1]) == "batch_size"
    ]
    return pools, tables


def test_the_ring_arm_trades_the_recurrent_shadow_for_a_ring() -> None:
    """Checks the ring arm adds a ring pool and drops the shadow."""
    snapshot, _ = _tail_dims(_input_types("snapshot"))
    ring, _ = _tail_dims(_input_types("ring"))

    def rows(leaf: str) -> str:
        return f"{leaf.replace('/', '_')}_rows"

    assert snapshot == [
        rows(CONV_LEAF_ID),
        rows(RECURRENT_LEAF_ID),
        f"shadow_{rows(RECURRENT_LEAF_ID)}",
    ]
    assert ring == [
        rows(CONV_LEAF_ID),
        rows(RECURRENT_LEAF_ID),
        rows(RING_LEAF_ID),
    ]


def test_every_cache_leaf_takes_a_row_table_and_no_shadow_does() -> None:
    """Checks the ring takes engine rows, like the live leaves."""
    _, snapshot = _tail_dims(_input_types("snapshot"))
    _, ring = _tail_dims(_input_types("ring"))
    assert len(snapshot) == 2, f"snapshot: got {snapshot}"
    assert len(ring) == 3, f"ring: got {ring}"


def test_the_ring_arm_adds_one_slot_per_device() -> None:
    """Checks the ring arm's pool and row table replace one shadow, per
    device, after the same prefix."""
    num_devices = 1
    snapshot = _input_types("snapshot", num_devices)
    ring = _input_types("ring", num_devices)
    assert len(ring) - len(snapshot) == num_devices
    first_pool = next(
        i
        for i, t in enumerate(snapshot)
        if isinstance(t, BufferType) and str(t.shape[0]).endswith("_rows")
    )
    assert ring[:first_pool] == snapshot[:first_pool]


def test_the_ring_is_float32_whatever_the_pool_dtype() -> None:
    """Checks the ring is float32."""
    types = _input_types("ring")
    rings = [
        t.dtype
        for t in types
        if isinstance(t, BufferType) and str(t.shape[0]).endswith("_ring_rows")
    ]
    assert rings == [DType.float32]


def test_only_the_leaves_the_verify_writes_are_shadowed() -> None:
    """Checks only the recurrent leaf is shadowed, and only without a ring."""
    assert shadowed_leaf_ids(0) == (RECURRENT_LEAF_ID,)
    assert shadowed_leaf_ids(4) == ()


def test_the_verify_window_rounds_up_to_a_compiled_ring() -> None:
    """Checks the verify window rounds up to a compiled ring length."""
    assert ring_len_for_window(1 + 1) == 2
    assert ring_len_for_window(1 + 2) == 4
    assert ring_len_for_window(1 + NUM_DRAFTS) == 4
    assert ring_len_for_window(1 + 7) == 8
    with pytest.raises(ValueError, match="longer than any"):
        ring_len_for_window(max(COMPILED_RING_LENS) + 1)


@pytest.mark.parametrize("ring_len", [2, 4, 8])
@pytest.mark.parametrize("num_devices", [1, 2])
def test_the_ring_page_divides_the_live_pages(
    ring_len: int, num_devices: int
) -> None:
    """Checks the ring's padded page leaves the huge block unchanged.

    At the published geometry a record holds the raw key and three value
    heads' delta rows and decays, 515 elements, padded to 640.
    """
    regions = linear_state_regions(
        num_linear_layers=48,
        key_head_dim=128,
        num_key_heads=16,
        value_head_dim=128,
        num_value_heads=48,
        conv_kernel_dim=4,
        dtype=DType.bfloat16,
        num_devices=num_devices,
        ring_len=ring_len,
    )
    *live, ring = regions
    assert ring.leaf_id == RING_LEAF_ID
    assert ring.row_shape == (16 // num_devices, ring_len, 640)
    live_lcm = math.lcm(*(region.bytes_per_page for region in live))
    assert live_lcm % ring.bytes_per_page == 0


def _build_graph(
    rollback: RecurrentStateRollback,
) -> tuple[Graph, dict[str, Any]]:
    """Builds the fused graph the way the model does."""
    config = _config(1)
    driver = UnifiedMTPQwen3_5(
        config,
        speculative_config=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=NUM_DRAFTS,
            recurrent_state_rollback=rollback,
        ),
        enable_structured_output=True,
    )
    raw = driver.raw_state_dict()
    driver.load_state_dict(
        {
            name: Buffer.zeros(shape=w.shape.static_dims, dtype=w.dtype)
            for name, w in raw.items()
        },
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,
    )

    attn = attn_cache(config.kv_params)
    spec_kv = MultiKVCacheParams.from_params(
        {"target": attn, "draft": replace(attn, num_layers=1)}
    )
    with Graph(
        f"mtp_{rollback}_rollback", input_types=driver.input_types(spec_kv)
    ) as graph:
        graph_inputs = driver.decode_inputs(graph.inputs, spec_kv)
        trailing = iter(graph_inputs.trailing)
        state = state_tail(trailing, driver.state_regions, driver.ring_len, 1)
        outputs = driver(
            graph_inputs.tokens,
            graph_inputs.input_row_offsets,
            graph_inputs.draft_tokens,
            kv_collections=graph_inputs.kv("target"),
            draft_kv_collections=graph_inputs.kv("draft"),
            return_n_logits=graph_inputs.return_n_logits,
            signal_buffers=graph_inputs.signal_buffers,
            host_input_row_offsets=graph_inputs.host_offsets,
            data_parallel_splits=graph_inputs.dp_splits,
            seed=graph_inputs.seed,
            temperature=graph_inputs.temperature,
            top_k=graph_inputs.top_k,
            max_k=graph_inputs.max_k,
            top_p=graph_inputs.top_p,
            min_top_p=graph_inputs.min_top_p,
            in_thinking_phase=graph_inputs.thinking_phase,
            pinned_bitmask=graph_inputs.pinned_bitmask,
            wait_payload=graph_inputs.wait_payload,
            device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            extra={**state, POSITION_IDS: None},
        )
        graph.output(*outputs)
    return graph, driver.state_dict()


@pytest.mark.parametrize("rollback", ["snapshot", "ring"])
def test_the_fused_graph_compiles_on_either_rollback(
    rollback: RecurrentStateRollback,
) -> None:
    """Checks the fused graph builds and compiles on each rollback."""
    graph, weights = _build_graph(rollback)
    session = InferenceSession(devices=[Accelerator()])
    assert session.load(graph, weights_registry=weights) is not None


def _served_kv_params(num_devices: int = 1) -> MultiKVCacheParams:
    """Returns the allocated cache: the graph's plus the state child."""
    config = _config(num_devices)
    attn = attn_cache(config.kv_params)
    state = RecurrentStateParams(
        regions=linear_state_regions(
            num_linear_layers=1,
            key_head_dim=config.linear_key_head_dim,
            num_key_heads=config.linear_num_key_heads,
            value_head_dim=config.linear_value_head_dim,
            num_value_heads=config.linear_num_value_heads,
            conv_kernel_dim=config.linear_conv_kernel_dim,
            dtype=config.state_dtype,
            num_devices=num_devices,
        ),
        devices=attn.devices,
        data_parallel_degree=attn.data_parallel_degree,
    )
    return MultiKVCacheParams.from_params(
        {
            "target": attn,
            "draft": replace(attn, num_layers=1),
            STATE_CACHE_KEY: state,
        }
    )


def test_the_graph_drops_the_state_child_the_cache_holds() -> None:
    """Checks ``graph_kv_params`` drops the allocated cache's state child."""
    served = _served_kv_params()

    assert recurrent_leaf(served) is not None
    assert recurrent_leaf(graph_kv_params(served)) is None
    assert set(graph_kv_params(served).children) == {"target", "draft"}


@pytest.mark.parametrize("rollback", ["snapshot", "ring"])
def test_the_state_child_does_not_move_the_signature(
    rollback: RecurrentStateRollback,
) -> None:
    """Checks the allocated cache yields the attention-only signature."""
    config = _config(1)
    driver = UnifiedMTPQwen3_5(
        config,
        speculative_config=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=NUM_DRAFTS,
            recurrent_state_rollback=rollback,
        ),
    )

    view = graph_kv_params(_served_kv_params())

    assert tuple(driver.input_types(view)) == _input_types(rollback)


def test_the_model_allocates_its_cache_with_the_ring() -> None:
    """Checks the model builds its cache from the config that declares the
    ring, so the cache it allocates holds the leaves the graph reads."""
    assert UnifiedMTPQwen3_5Model.model_config_cls is UnifiedMTPQwen3_5Config
    ring = SimpleNamespace(
        speculative=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=NUM_DRAFTS,
            recurrent_state_rollback="ring",
        )
    )
    assert UnifiedMTPQwen3_5Config._verify_ring_len(
        cast("PipelineConfig", ring)
    ) == ring_len_for_window(1 + NUM_DRAFTS)
