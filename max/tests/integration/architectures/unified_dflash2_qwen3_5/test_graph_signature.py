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
"""The fused DFlash2 graph's input signature, pinned slot by slot.

Mach's spec-step executor binds this signature positionally, so a reordering
or an added slot is an ABI break that the engine can only report as an arity
mismatch (or, worse, cannot report at all when the count happens to match).
The MTP graph is asserted from Mach's side; this asserts the
DFlash2 graph from MAX's side, and asserts that the two agree slot for slot
apart from the draft leaf's geometry, which is the one thing that differs.

The KV group is seven slots here since MXSERV-527 added ``page_stride``.
Mach's executors still pin six, so the two disagree until they are updated.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    recurrent_leaf,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import (
    ATTN_CACHE_KEY,
    RING_LEAF_ID,
    STATE_CACHE_KEY,
    attn_cache,
    linear_state_regions,
    ring_len_for_window,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.memory_planner import (
    UnifiedDflash2Qwen3_5MemoryPlanner,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.model_config import (
    DRAFT_SLIDING_WINDOW,
    UnifiedDflash2Qwen3_5Config,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.unified_dflash2_qwen3_5 import (
    UnifiedDflash2Qwen3_5,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.inputs import (
    UnifiedMTPQwen3_5Inputs,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.spec_state import (
    graph_kv_params,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.speculative.config import SpeculativeConfig

HIDDEN = 64
VOCAB = 128
BLOCK = 8
DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 2
DRAFT_HEAD_DIM = 16
PAGE_SIZE = 32
# Three linear-attention layers and one full-attention layer: the pool tail is
# per linear layer, so a mix is what makes the count formula falsifiable.
LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
NUM_LINEAR = 3


def _target_config() -> Qwen3_5Config:
    """Returns a target whose cache holds its state and a block-long ring."""
    config = _target_config_without_state()
    attn = attn_cache(config.kv_params)
    config.kv_params = MultiKVCacheParams.from_params(
        {
            ATTN_CACHE_KEY: attn,
            STATE_CACHE_KEY: RecurrentStateParams(
                devices=attn.devices,
                data_parallel_degree=attn.data_parallel_degree,
                regions=linear_state_regions(
                    num_linear_layers=NUM_LINEAR,
                    key_head_dim=config.linear_key_head_dim,
                    num_key_heads=config.linear_num_key_heads,
                    value_head_dim=config.linear_value_head_dim,
                    num_value_heads=config.linear_num_value_heads,
                    conv_kernel_dim=config.linear_conv_kernel_dim,
                    dtype=config.state_dtype,
                    num_devices=1,
                    ring_len=ring_len_for_window(BLOCK),
                ),
            ),
        }
    )
    return config


def _target_config_without_state() -> Qwen3_5Config:
    device = DeviceRef.CPU()
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(LAYER_TYPES),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            devices=[device],
            n_kv_heads=1,
            head_dim=16,
            num_layers=1,
            page_size=PAGE_SIZE,
        ),
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=16**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
    )


def _fused_config() -> UnifiedDflash2Qwen3_5Config:
    device = DeviceRef.CPU()
    target = _target_config()
    draft_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=DRAFT_KV_HEADS,
        head_dim=DRAFT_HEAD_DIM,
        num_layers=DRAFT_LAYERS,
        page_size=PAGE_SIZE,
        window_size=DRAFT_SLIDING_WINDOW,
    )
    draft = Llama3Config(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=DRAFT_KV_HEADS,
        num_hidden_layers=DRAFT_LAYERS,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=False,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=draft_kv,
        rms_norm_eps=1e-6,
        attention_multiplier=DRAFT_HEAD_DIM**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        sliding_window=DRAFT_SLIDING_WINDOW,
    )
    return UnifiedDflash2Qwen3_5Config(
        target=target,
        draft=draft,
        draft_kv_params=draft_kv,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=BLOCK - 1
        ),
        target_layer_ids=[0, 1, 2, 3, 3],
        layer_types=["sliding_attention"] * DRAFT_LAYERS,
        mask_token_id=VOCAB - 1,
        block_size=BLOCK,
        conv_kernel_size=2,
        conv_group_size=8,
        selector_rank=4,
        selector_top_k=3,
    )


def _signature(
    enable_structured_output: bool = True,
) -> tuple[TensorType | BufferType, ...]:
    config = _fused_config()
    module = UnifiedDflash2Qwen3_5(
        config, enable_structured_output=enable_structured_output
    )
    return module.input_types(graph_kv_params(config.get_kv_params()))


def test_the_served_cache_holds_the_state_and_its_ring() -> None:
    """Checks the allocated cache carries the state child, ring included.

    MAX serves this graph from that cache, and the signature drops the child
    because the tail declares the state instead.
    """
    tree = _fused_config().get_kv_params()
    assert isinstance(tree, MultiKVCacheParams)
    assert set(tree.children) == {"target", "draft", STATE_CACHE_KEY}
    state = recurrent_leaf(tree)
    assert state is not None
    (ring,) = (r for r in state.regions if r.leaf_id == RING_LEAF_ID)
    assert ring.scratch and ring.row_shape[1] == BLOCK
    assert set(graph_kv_params(tree).children) == {"target", "draft"}


def test_slot_count_matches_the_declared_formula() -> None:
    """``15 + 23D`` at ``D = 1``.

    The same formula the Qwen3.5 MTP graph satisfies: five ragged/host inputs,
    signals, two 7-slot KV leaves, batch_context_lengths, the eight-entry
    sampling tail, the bitmask triple, then a pool and a row table for each
    of the three state leaves, the ring included. The layer count does not
    enter.
    """
    types = _signature()
    assert len(types) == 15 + 23 * 1


def test_the_prefix_and_sampling_tail_are_the_canonical_ones() -> None:
    types = _signature()
    prefix = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[:5]]
    assert prefix == [
        (DType.int64, ("total_seq_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.int64, ("return_n_logits",)),
        (DType.int64, ("2",)),
    ]
    # host_input_row_offsets and return_n_logits are host-side.
    cpu = DeviceRef.CPU()
    assert types[2].device == cpu and types[3].device == cpu

    # draft_tokens .. in_thinking_phase, then the bitmask triple.
    tail = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[21:32]]
    assert tail == [
        (DType.int64, ("batch_size", "num_steps")),
        (DType.uint64, ("batch_size",)),
        (DType.float32, ("batch_size",)),
        (DType.int64, ("batch_size",)),
        (DType.int64, ()),
        (DType.float32, ("batch_size",)),
        (DType.float32, ()),
        (DType.bool, ("batch_size",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
        (DType.int64, ("2",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
    ]


def test_the_draft_leaf_is_the_drafters_own_windowed_geometry() -> None:
    """Slots 13-19: five drafter layers, bf16, bounded at the drafter's window.

    Every one of these is a silent-wrong-answer if it drifts from the Mach
    registry's draft group: a wrong layer count reads another layer's K/V, a
    wrong dtype reinterprets the bytes, and an unbounded leaf turns a
    per-slot constant into a cost that scales with ``--max-length``.
    """
    config = _fused_config()
    tree = config.get_kv_params()
    assert isinstance(tree, MultiKVCacheParams)
    draft = tree.children["draft"]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.num_layers == DRAFT_LAYERS
    assert draft.n_kv_heads == DRAFT_KV_HEADS
    assert draft.head_dim == DRAFT_HEAD_DIM
    assert draft.dtype == DType.bfloat16
    assert draft.window_size == DRAFT_SLIDING_WINDOW
    assert draft.group_id.is_sliding_window()
    # One page size across the tree, or the manager and the graph disagree.
    target = tree.children["target"]
    assert isinstance(target, MHAKVCacheParams)
    assert draft.page_size == target.page_size

    # Slot 13: the draft leaf's blocks, right after the target leaf's seven.
    types = _signature()
    blocks = types[13]
    assert isinstance(blocks, BufferType)
    assert blocks.dtype == DType.bfloat16
    assert [str(d) for d in blocks.shape[1:]] == [
        "2",
        str(DRAFT_LAYERS),
        str(PAGE_SIZE),
        str(DRAFT_KV_HEADS),
        str(DRAFT_HEAD_DIM),
    ]


def test_a_quantized_target_leaf_does_not_quantize_the_draft_leaf() -> None:
    """``--kv-cache-dtype float8_e4m3fn`` must not reach the drafter's cache.

    The drafter fills it with its own unquantized projections, so a rewritten
    dtype is not a precision trade -- it is a reinterpretation of the bytes.
    """
    config = _fused_config()
    state = recurrent_leaf(config.target.kv_params)
    assert state is not None
    config.target.kv_params = MultiKVCacheParams.from_params(
        {
            ATTN_CACHE_KEY: replace(
                attn_cache(config.target.kv_params), dtype=DType.float8_e4m3fn
            ),
            STATE_CACHE_KEY: state,
        }
    )
    tree = config.get_kv_params()
    assert isinstance(tree, MultiKVCacheParams)
    target, draft = tree.children["target"], tree.children["draft"]
    assert isinstance(target, MHAKVCacheParams)
    assert isinstance(draft, MHAKVCacheParams)
    assert target.dtype == DType.float8_e4m3fn
    assert draft.dtype == DType.bfloat16


def test_the_state_pool_tail_matches_the_mtp_graphs() -> None:
    """Checks the state tail matches the MTP graph's, slot for slot."""
    config = _fused_config()
    module = UnifiedDflash2Qwen3_5(config, enable_structured_output=True)
    fused = module.input_types(graph_kv_params(config.get_kv_params()))
    mtp_kv = MultiKVCacheParams.from_params(
        {
            "target": attn_cache(config.target.kv_params),
            "draft": replace(attn_cache(config.target.kv_params), num_layers=1),
        }
    )
    mtp = UnifiedMTPQwen3_5(
        config.target,
        speculative_config=config.speculative_config,
        enable_structured_output=True,
    ).input_types(mtp_kv)

    assert len(fused) == len(mtp), (
        "the two graphs must present the same slot count, so Mach's Qwen"
        " layout binds both"
    )
    # A pool and a row table per state leaf, the ring included.
    tail_start = len(fused) - 2 * len(module.state_regions)
    for a, b in zip(fused[tail_start:], mtp[tail_start:], strict=True):
        assert a.dtype == b.dtype
        assert [str(d) for d in a.shape] == [str(d) for d in b.shape]


def test_structured_output_off_drops_exactly_the_bitmask_triple() -> None:
    assert len(_signature(False)) == len(_signature(True)) - 3


def test_the_shared_batch_fills_every_slot_the_graph_declares() -> None:
    """Checks the MTP graph's inputs pack one buffer per DFlash2 slot.

    MAX batches this graph with the MTP graph's processor, which never
    declares M-RoPE positions here.
    """
    config = _fused_config()
    view = graph_kv_params(config.get_kv_params())

    def dummy() -> Buffer:
        return Buffer.zeros(shape=[1], dtype=DType.int64)

    inputs = UnifiedMTPQwen3_5Inputs(
        tokens=dummy(),
        input_row_offsets=dummy(),
        host_input_row_offsets=dummy(),
        return_n_logits=dummy(),
        data_parallel_splits=dummy(),
        signal_buffers=[dummy()],
        batch_context_lengths=[dummy()],
        # Only the leaf count matters.
        kv_cache_inputs=cast(
            "KVCacheInputs[Buffer, Buffer]",
            {"kv": [dummy() for _ in view.flattened_kv_inputs()]},
        ),
        live_conv_pools=[dummy()],
        live_recurrent_pools=[dummy()],
        live_conv_row_ids=[dummy()],
        live_recurrent_row_ids=[dummy()],
        ring_pools=[dummy()],
        ring_row_ids=[dummy()],
        draft_tokens=dummy(),
        seed=dummy(),
        temperature=dummy(),
        top_k=dummy(),
        max_k=dummy(),
        top_p=dummy(),
        min_top_p=dummy(),
        in_thinking_phase=dummy(),
        pinned_bitmask=dummy(),
        wait_payload=dummy(),
        device_bitmask_scratch=dummy(),
        structured_output=True,
    )

    assert len(inputs.buffers) == len(_signature())


def test_a_request_is_priced_its_block_long_ring() -> None:
    """Checks the planner prices the ring the served cache holds."""
    config = _fused_config()
    planner = UnifiedDflash2Qwen3_5MemoryPlanner(config)
    pipeline_config = cast(
        "PipelineConfig",
        SimpleNamespace(speculative=config.speculative_config),
    )

    ring_bytes = config.target._per_request_ring_bytes(
        ring_len_for_window(BLOCK)
    )
    assert ring_bytes > 0
    assert planner.spec_state_bytes(pipeline_config) == ring_bytes
