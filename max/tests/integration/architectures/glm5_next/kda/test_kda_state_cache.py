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
"""The KDA state leaves: shapes, the cache tree they join, and their row ids.

Four things are checked that a forward pass would catch late, or not at all:

* Each leaf's per-device row shape equals the *layer shard's* own. Two files
  compute the tensor-parallel division and a disagreement is a shape error at
  the first forward, long after load.
* ``construct_kv_params`` declares the state beside the two attention caches,
  and the leaves it declares cost what memory planning divides its budget by.
  When these diverged in the sibling architecture the symptom was an inflated
  batch size that OOMed at load, nowhere near the cause.
* The tail ring, the third state kind, does not widen the pool's block. A block
  is the least common multiple of its leaves' pages, so a leaf whose layer
  count is coprime to the rest multiplies it; the ring is sized over the
  indexer cache's layer set precisely so it cannot.
* A KDA layer runs in *its own* row of each leaf. Every layer shares one pool
  per leaf and differs only in the column of the folded row ids it reads, so
  reading the wrong column is neither a shape error nor a crash, just a worse
  model. This one is checked by executing the graph and comparing numbers,
  because that is the only way a wrong-but-valid index shows up.

The pool lifecycle itself -- the wipe a fresh request gets, the copy a
checkpoint and a prefix hit make -- belongs to the cache group and is covered
by ``max/tests/integration/kv_cache/jenga``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    Type,
)
from max.nn.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
    recurrent_leaf,
)
from max.pipelines.architectures.glm5_next.layers.kda import (
    kda_sublayer_inputs,
)
from max.pipelines.architectures.glm5_next.layers.kimi_delta_attention import (
    KimiDeltaAttention,
)
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.pipelines.architectures.glm5_next.state_cache import (
    CONV_LEAF_ID,
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    RECURRENT_LEAF_ID,
    STATE_CACHE_KEY,
    TAIL_RING_DTYPE,
    TAIL_RING_LEAF_ID,
    state_regions,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    compute_jenga_ratios,
)
from max.pipelines.lib import KVCacheConfig

HEADS = 64
HEAD_DIM = 128
KERNEL = 4
HIDDEN = 4096

KDA_LAYERS = (0, 1, 2, 4, 5, 6)
"""Two periods of the real schedule -- every index where ``i % 4 != 3``."""

SPARSE_LAYERS = (3, 7)
"""The same two periods' sparse-MLA layers -- the tail ring's layer set.

Deliberately a different length from ``KDA_LAYERS``: a ring sized or strided
by the KDA count would still be self-consistent, so only a fixture where the
two counts differ can catch it.
"""

KPOOL = 4
INDEX_HEAD_DIM = 128

POOL_DTYPE = DType.float32


def _regions(
    num_devices: int,
    *,
    num_heads: int = HEADS,
    head_dim: int = HEAD_DIM,
) -> tuple[Any, ...]:
    return state_regions(
        num_kda_layers=len(KDA_LAYERS),
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_dim=KERNEL,
        num_sparse_layers=len(SPARSE_LAYERS),
        index_kpool=KPOOL,
        index_head_dim=INDEX_HEAD_DIM,
        dtype=POOL_DTYPE,
        num_devices=num_devices,
    )


def _by_leaf(num_devices: int) -> dict[str, Any]:
    return {region.leaf_id: region for region in _regions(num_devices)}


# ===--------------------------------------------------------------------=== #
# Geometry
# ===--------------------------------------------------------------------=== #


@pytest.mark.parametrize("num_devices", [1, 4, 8])
def test_region_shapes_match_the_layer_shard(num_devices: int) -> None:
    """The cache and the layer agree on what one device holds per layer.

    ``KimiDeltaAttention`` splits 64 heads inside each of q, k and v; the
    cache multiplies the per-device head count by three. The two are different
    derivations of the same number and nothing else checks them against each
    other.
    """
    layer = KimiDeltaAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        dtype=DType.bfloat16,
        device=DeviceRef.CPU(),
        rms_norm_eps=1e-5,
        lower_bound=-5.0,
    )
    layer.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    shard = layer.shard([DeviceRef.CPU()] * num_devices)[0]

    conv_row, recurrent_row = shard.state_row_shapes()
    regions = _by_leaf(num_devices)
    assert regions[CONV_LEAF_ID].row_shape == conv_row
    assert regions[RECURRENT_LEAF_ID].row_shape == recurrent_row


def test_state_regions_reject_an_indivisible_device_count() -> None:
    """Rounding the split would hand one rank heads another rank also owns."""
    with pytest.raises(ValueError, match="divisible by the device count"):
        _regions(5)


def test_the_tail_ring_is_its_own_family() -> None:
    """A third state kind, over its own layer set, at its own dtype.

    The ring holds indexer keys and their gate scores, which stay bfloat16
    whatever the KDA pools are stored at, and it runs over the sparse-MLA
    layers rather than the KDA ones. Unlike those, it is not sharded: the
    indexer scores one key per token rather than one per head, so every device
    holds the whole ring.
    """
    one, eight = _by_leaf(1), _by_leaf(8)
    ring = eight[TAIL_RING_LEAF_ID]

    assert ring.num_layers == len(SPARSE_LAYERS)
    assert ring.num_layers != eight[CONV_LEAF_ID].num_layers
    assert ring.dtype == TAIL_RING_DTYPE != POOL_DTYPE
    assert ring.row_shape == (2, KPOOL, INDEX_HEAD_DIM)
    assert ring.row_shape == one[TAIL_RING_LEAF_ID].row_shape, (
        "the ring is replicated, not sharded"
    )
    assert eight[CONV_LEAF_ID].row_shape != one[CONV_LEAF_ID].row_shape, (
        "the KDA leaves are sharded, so this fixture proves the contrast"
    )


# ===--------------------------------------------------------------------=== #
# The cache tree
# ===--------------------------------------------------------------------=== #


def _text_config(num_hidden_layers: int = 8) -> SimpleNamespace:
    """A GLM-5.3-Flash text config, cut down to what the caches read."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=INDEX_HEAD_DIM,
        index_kpool=KPOOL,
        layer_types=[
            "deepseek_sparse_attention" if (i % 4) == 3 else "linear_attention"
            for i in range(num_hidden_layers)
        ],
        # The key names GLM-5.3-Flash's published config uses, not the
        # dataclass field names -- reading the latter defaults every dim.
        linear_attn_config={
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": KERNEL,
        },
    )


def _cache_params(
    num_hidden_layers: int = 8, num_devices: int = 1
) -> MultiKVCacheParams:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None
    params = Glm5NextConfig.construct_kv_params(
        huggingface_config=_text_config(num_hidden_layers),
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU(i) for i in range(num_devices)],
        kv_cache_config=KVCacheConfig(state_pool_dtype="float32"),
        cache_dtype=DType.bfloat16,
    )
    assert isinstance(params, MultiKVCacheParams)
    return params


def test_the_state_joins_the_attention_caches() -> None:
    """The cache declares three children, one of them the recurrent state.

    The graph's input types are built from these params, so a state derived
    any later could not appear among them. The tree keeps two attention
    children, which is what the pool reads its page size and geometry off.
    """
    params = _cache_params()
    assert set(params.children) == {
        MLA_CACHE_KEY,
        INDEXER_CACHE_KEY,
        STATE_CACHE_KEY,
    }
    state = recurrent_leaf(params)
    assert isinstance(state, RecurrentStateParams)
    assert [region.leaf_id for region in state.regions] == [
        CONV_LEAF_ID,
        RECURRENT_LEAF_ID,
        TAIL_RING_LEAF_ID,
    ]

    leaves = params.leaves()
    recurrent = {
        leaf_id
        for leaf_id, leaf in leaves.items()
        if leaf.group_id.is_recurrent()
    }
    assert recurrent == {CONV_LEAF_ID, RECURRENT_LEAF_ID, TAIL_RING_LEAF_ID}
    assert set(leaves) - recurrent, "the attention leaves must still be there"

    # The invariant the ring's dead-byte argument rests on: a checkpoint
    # boundary is always a pool boundary, so the ring holds no live token
    # wherever the group publishes.
    assert params.page_size % KPOOL == 0


def test_the_ring_rides_the_indexers_layer_set() -> None:
    """Its page divides the indexer's, so it cannot widen the pool's block.

    A block is the least common multiple of its leaves' pages, so a leaf whose
    layer count brings a new prime factor multiplies it. Twelve hidden layers
    puts three layers in the indexer cache and nine in the KDA ones: the ring
    is free at three, and the second arm shows what a count off that layer set
    would have cost.
    """
    params = _cache_params(num_hidden_layers=12)
    pages = {
        leaf_id: leaf.bytes_per_page
        for leaf_id, leaf in params.leaves().items()
    }
    without_ring = {
        leaf_id: size
        for leaf_id, size in pages.items()
        if leaf_id != TAIL_RING_LEAF_ID
    }
    assert pages[TAIL_RING_LEAF_ID] > 0

    budget = 64 * 1024**3
    _, block, _ = compute_jenga_ratios(budget, pages)
    _, bare_block, _ = compute_jenga_ratios(budget, without_ring)
    assert block == bare_block, (
        f"the tail ring widened the pool block from {bare_block} to {block} B"
    )

    # The falsification arm: five is a layer count no other leaf carries.
    stray = dict(without_ring)
    stray[TAIL_RING_LEAF_ID] = (
        5 * 2 * KPOOL * INDEX_HEAD_DIM * TAIL_RING_DTYPE.size_in_bytes
    )
    _, stray_block, _ = compute_jenga_ratios(budget, stray)
    assert stray_block == 5 * bare_block, (
        "a ring off the indexer's layer set should cost a multiplier, so this"
        " fixture no longer proves the ring's layer set is what saves it"
    )


@pytest.mark.parametrize("num_devices", [1, 4])
def test_the_declared_state_is_what_a_request_costs(num_devices: int) -> None:
    """Sharding conserves the KDA state and replicates the ring.

    Memory planning divides a budget by this number, so an inflated one
    promises concurrency the pool cannot hold. The KDA leaves split by head,
    so their total is the same at any device count; the ring does not split,
    so every device pays for it.
    """
    state = recurrent_leaf(_cache_params(num_devices=num_devices))
    assert isinstance(state, RecurrentStateParams)

    element = POOL_DTYPE.size_in_bytes
    kda = (
        len(KDA_LAYERS)
        * element
        * (3 * HEADS * HEAD_DIM * (KERNEL - 1) + HEADS * HEAD_DIM * HEAD_DIM)
    )
    ring = (
        len(SPARSE_LAYERS)
        * 2
        * KPOOL
        * INDEX_HEAD_DIM
        * TAIL_RING_DTYPE.size_in_bytes
    )
    assert state.bytes_per_state * num_devices == kda + ring * num_devices


def test_the_kda_dims_come_from_the_published_key_names() -> None:
    """The pools and the layers must read one set of HF key names.

    Every dim here differs from its fallback, so a lookup that misses the key
    returns the fallback and fails. The fixtures elsewhere cannot catch this:
    their dims *are* the fallbacks.
    """
    dims = Glm5NextConfig.declared_kda_dims(
        SimpleNamespace(
            linear_attn_config={
                "num_heads": 16,
                "head_dim": 32,
                "short_conv_kernel_size": 5,
            }
        )
    )
    assert dims == (16, 32, 5)

    # The dataclass field names are not the config's, and reading them was the
    # bug: three silent fallbacks rather than an error.
    defaulted = Glm5NextConfig.declared_kda_dims(
        SimpleNamespace(
            linear_attn_config={
                "linear_num_heads": 16,
                "linear_head_dim": 32,
                "linear_conv_kernel_dim": 5,
            }
        )
    )
    assert defaulted == (64, 128, 4)
    assert defaulted != dims, (
        "the fallbacks must differ, or this proves nothing"
    )


# ===--------------------------------------------------------------------=== #
# Row ids
# ===--------------------------------------------------------------------=== #

SMALL_HEADS = 4
SMALL_HEAD_DIM = 8
"""Small on purpose: the row-id test allocates the pools for real."""

BLOCKS = 4
"""Blocks each leaf's pool holds, so a row id can name any of them."""


def _small_state(num_devices: int) -> RecurrentStateParams:
    return RecurrentStateParams(
        regions=_regions(
            num_devices, num_heads=SMALL_HEADS, head_dim=SMALL_HEAD_DIM
        ),
        devices=[DeviceRef.CPU(i) for i in range(num_devices)],
    )


@pytest.mark.parametrize("num_devices", [1, 2])
def test_each_layer_runs_in_its_own_row(num_devices: int) -> None:
    """Layer ``l`` reads row ``l`` of each leaf's folded row ids.

    Executed rather than inspected: a bundle that read the decoder index, or
    the other leaf's ids, would build and compile and return a valid row --
    just the wrong one. Only the numbers separate them, so the two leaves are
    staged from *different* blocks and every column is distinct.
    """
    state_params = _small_state(num_devices)
    regions = {r.leaf_id: r for r in state_params.regions}
    num_layers = len(KDA_LAYERS)

    signal_types = [
        BufferType(DType.uint8, [16], device=DeviceRef.CPU(i))
        for i in range(num_devices)
    ]
    offset_types = [
        TensorType(DType.uint32, ["batch_plus_one"], device=DeviceRef.CPU(i))
        for i in range(num_devices)
    ]
    state_types = tree.leaves(state_params.get_symbolic_inputs(), leaf=Type)

    with Graph(
        "KdaRowIdWiring",
        input_types=[*state_types, *signal_types, *offset_types],
    ) as graph:
        values = list(graph.inputs)
        state = tree.leaves(
            state_params.unflatten_kv_inputs(iter(values)),
            leaf=RecurrentStateInputsPerDevice,
        )
        rest = values[len(state_types) :]
        bundles = kda_sublayer_inputs(
            kda_layers=KDA_LAYERS,
            state=state,
            signal_buffers=[v.buffer for v in rest[:num_devices]],
            input_row_offsets=[v.tensor for v in rest[num_devices:]],
        )
        assert set(bundles) == set(KDA_LAYERS)
        graph.output(
            *(bundles[layer].conv_row_ids[0] for layer in KDA_LAYERS),
            *(bundles[layer].recurrent_row_ids[0] for layer in KDA_LAYERS),
        )

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)

    batch = 2
    # Distinct blocks per leaf, so ids taken from the wrong leaf land on
    # rows no column of the right one names.
    conv_blocks, recurrent_blocks = [0, 1], [2, 3]
    staged = {
        CONV_LEAF_ID: conv_blocks,
        RECURRENT_LEAF_ID: recurrent_blocks,
        TAIL_RING_LEAF_ID: [0, 1],
    }
    # `[num_layers, batch]`, which is how `_layer_row_ids` indexes it: one
    # row per layer, one column per sequence. `rows_of` runs the other way,
    # a block's rows layer by layer, so the stack of blocks transposes.
    row_ids = {
        leaf_id: np.array(
            [regions[leaf_id].rows_of(block) for block in blocks],
            dtype=np.uint32,
        ).T.copy()
        for leaf_id, blocks in staged.items()
    }
    pools = {
        leaf_id: Buffer.zeros(
            [BLOCKS * region.num_layers, *region.row_shape],
            region.dtype,
            CPU(),
        )
        for leaf_id, region in regions.items()
    }
    # Leaf-major per device, the order `RecurrentStateInputsPerDevice`
    # flattens in: each leaf's pool followed by that leaf's row ids.
    inputs: list[Buffer] = []
    for _ in range(num_devices):
        for r in state_params.regions:
            inputs.append(pools[r.leaf_id])
            inputs.append(Buffer.from_numpy(row_ids[r.leaf_id]).to(CPU()))
    inputs.extend(
        Buffer.zeros([16], DType.uint8, CPU()) for _ in range(num_devices)
    )
    inputs.extend(
        Buffer.from_numpy(np.arange(batch + 1, dtype=np.uint32)).to(CPU())
        for _ in range(num_devices)
    )

    outputs = model.execute(*inputs)
    assert len(outputs) == 2 * num_layers
    for position, layer_idx in enumerate(KDA_LAYERS):
        conv = np.asarray(outputs[position].to_numpy())
        recurrent = np.asarray(outputs[num_layers + position].to_numpy())
        expected_conv = [block * num_layers + position for block in conv_blocks]
        expected_recurrent = [
            block * num_layers + position for block in recurrent_blocks
        ]
        assert conv.tolist() == expected_conv, (
            f"KDA layer {layer_idx} (position {position}) read conv rows"
            f" {conv.tolist()}, not {expected_conv}"
        )
        assert recurrent.tolist() == expected_recurrent, (
            f"KDA layer {layer_idx} (position {position}) read recurrent rows"
            f" {recurrent.tolist()}, not {expected_recurrent}"
        )


def test_bundles_reject_a_mismatched_device_count() -> None:
    """A per-device list of the wrong length is a wiring bug, not a resize."""
    state_params = _small_state(2)
    with Graph(
        "KdaRowIdArity",
        input_types=tree.leaves(state_params.get_symbolic_inputs(), leaf=Type),
    ) as graph:
        state = tree.leaves(
            state_params.unflatten_kv_inputs(iter(graph.inputs)),
            leaf=RecurrentStateInputsPerDevice,
        )
        with pytest.raises(
            ValueError, match="signal_buffers must have one entry per device"
        ):
            kda_sublayer_inputs(
                kda_layers=KDA_LAYERS,
                state=state,
                signal_buffers=[],
                input_row_offsets=[],
            )
        graph.output()
