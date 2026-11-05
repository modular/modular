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
"""The Kimi K3 (ModuleV3) target adapter for the DSpark pairing.

Graph-level checks, ahead of the end-to-end pair in
``test_spec_parity.py`` (neither of them needs an accelerator):

* the adapter registry routes the ModuleV3 target to the K3 adapter, past
  the V2 graph-API entry its base name maps to;
* the verify hands the target's ``forward`` the very graph inputs the
  driver decoded -- so the KDA pools the rollback folds are the buffers
  the forward mutated, never copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

from max import tree
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Graph, TensorValue
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.spec_adapters import (
    ADAPTER,
    target_kv_inputs,
)
from max.pipelines.architectures.kimi_k3_modulev3.model import KimiK3Model
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    kda_state_regions,
    with_verify_ring,
)
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.speculative import spec_adapters as spec_adapters_registry
from max.pipelines.speculative.block_driver import BlockBatch
from max.tensor.realization_context import ensure_context

_HEADS = 2
_HEAD_DIM = 32
_CONV_KERNEL = 4
_NUM_KDA = 1
_RING_LEN = 8
_DRAFT_WIDTH = 7
_DEVICE = DeviceRef.CPU(0)


def _target_kv_tree() -> MultiKVCacheParams:
    """A K3 target's cache tree at the spec-decode shape: MLA, state, ring."""
    attn = KVCacheConfig().to_params(
        dtype=DType.bfloat16,
        n_kv_heads=_HEADS,
        head_dim=12,
        num_layers=1,
        devices=[_DEVICE],
        data_parallel_degree=1,
        speculative_method="dspark",
        num_draft_tokens=_DRAFT_WIDTH,
    )
    assert isinstance(attn, KVCacheParams)
    state = RecurrentStateParams(
        regions=with_verify_ring(
            kda_state_regions(
                num_kda_layers=_NUM_KDA,
                conv_dim=3 * _HEADS * _HEAD_DIM,
                conv_kernel_size=_CONV_KERNEL,
                num_heads=_HEADS,
                head_dim=_HEAD_DIM,
                conv_dtype=DType.bfloat16,
                recurrent_dtype=DType.float32,
                num_devices=1,
            ),
            _RING_LEN,
        ),
        devices=[_DEVICE],
        data_parallel_degree=1,
    )
    return MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
    )


def _per_device_leaves(
    kv_tree: Mapping[str, object],
) -> tuple[
    list[KVCacheInputsPerDevice[TensorValue, BufferValue]],
    list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
]:
    """The per-device leaf objects the driver's decode hands the batch."""
    return (
        list(tree.leaves(kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice)),
        list(
            tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            )
        ),
    )


def test_the_registry_routes_the_modulev3_target_to_its_adapter() -> None:
    """The ModuleV3 variant reaches the K3 adapter; the base name keeps the
    V2 graph-API entry a private build serves."""
    assert (
        spec_adapters_registry._TARGET_ADAPTERS[
            "KimiK3ForConditionalGeneration"
        ]
        == "kimi_k3_spec"
    )
    # The production gate: the spec pipeline pairs this target only when
    # an adapter is registered under its exact architecture name.
    assert spec_adapters_registry.has_adapter(
        "KimiK3ForConditionalGeneration_ModuleV3"
    )
    adapter = spec_adapters_registry._target_adapter(
        "KimiK3ForConditionalGeneration_ModuleV3"
    )
    assert adapter is ADAPTER
    assert adapter.primary_kv == ATTN_CACHE_KEY
    assert adapter.verify is ADAPTER.verify
    assert adapter.block_proposer is ADAPTER.block_proposer


def test_the_verify_binds_the_buffers_the_model_mutates() -> None:
    """The flat list the verify rebuilds is the graph's input list itself.

    The driver's decode unflattens the graph inputs into per-device leaves,
    the model's ``forward`` unflattens ``*kv_inputs`` back -- so every value
    the verify wraps must be the one the decode consumed, or the rollback's
    fold lands on a buffer nobody reads.
    """
    kv_params = _target_kv_tree()
    with Graph(
        "k3_verify_binding", input_types=kv_params.flattened_kv_inputs()
    ) as graph:
        kv_tree = kv_params.unflatten_kv_inputs(iter(graph.inputs))
        assert isinstance(kv_tree, dict)
        attn, state = _per_device_leaves(kv_tree)
        batch = cast(
            BlockBatch,
            SimpleNamespace(
                kv_collections=attn,
                passthrough_kv={STATE_CACHE_KEY: state},
            ),
        )
        target = cast(KimiK3Model, SimpleNamespace(kv_params=kv_params))
        with ensure_context():
            kv_inputs = target_kv_inputs(target, batch)
        assert [tensor._graph_value for tensor in kv_inputs] == list(
            graph.inputs
        )
        # The model's own unflatten, over the values the verify rebuilt:
        # same per-device leaves, same pools -- the ring's included.
        consumed = kv_params.unflatten_kv_inputs(
            iter(tensor._graph_value for tensor in kv_inputs)
        )
        assert isinstance(consumed, dict)
        rebuilt_attn, rebuilt_state = _per_device_leaves(consumed)
        for leaf, was in zip(rebuilt_attn, attn, strict=True):
            assert leaf.kv_blocks is was.kv_blocks
        for leaves, was_leaves in zip(rebuilt_state, state, strict=True):
            for bound, decoded in zip(
                leaves.leaves, was_leaves.leaves, strict=True
            ):
                assert bound.pool is decoded.pool
