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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from _dspark_test_data import TINY_DRAFT_CONFIG
from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
)
from max.nn.legacy.comm import Signals
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.spec_adapters import (
    ADAPTER,
    export_contract,
    target_kv_inputs,
)
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    kda_state_regions,
    with_verify_ring,
)
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.speculative import spec_adapters as spec_adapters_registry
from max.pipelines.speculative.block_driver import BlockBatch
from max.tensor.realization_context import ensure_context

_HEADS = 2
_HEAD_DIM = 32
_CONV_KERNEL = 4
_NUM_KDA = 1
_RING_LEN = 8
_DRAFT_WIDTH = 7
_PAGE_SIZE = 128
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


def _mla_leaf(
    num_layers: int, num_q_heads: int, devices: list[DeviceRef]
) -> MLAKVCacheParams:
    """One side of the pairing's MLA latent leaf."""
    params = KVCacheConfig().to_params(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=num_layers,
        devices=devices,
        data_parallel_degree=1,
        is_mla=True,
        num_q_heads=num_q_heads,
        speculative_method="dspark",
        num_draft_tokens=_DRAFT_WIDTH,
    )
    assert isinstance(params, MLAKVCacheParams)
    return params


def _pairing_kv_tree(
    devices: list[DeviceRef],
    *,
    with_ring: bool = True,
    target_kind: str = "mla",
) -> MultiKVCacheParams:
    """The pairing's cache tree: target MLA + KDA state, draft MLA.

    ``target_kind`` builds the states the hook must tolerate beside the
    frozen one: ``decompressed`` swaps the target's attention leaf for a
    plain paged (non-MLA) cache, and ``no_kda`` a target with no state
    child at all.
    """
    state = RecurrentStateParams(
        regions=kda_state_regions(
            num_kda_layers=_NUM_KDA,
            conv_dim=3 * 4 * _HEAD_DIM,
            conv_kernel_size=_CONV_KERNEL,
            num_heads=4,
            head_dim=_HEAD_DIM,
            conv_dtype=DType.bfloat16,
            recurrent_dtype=DType.float32,
            num_devices=len(devices),
            ring_len=_RING_LEN if with_ring else 0,
        ),
        devices=devices,
        data_parallel_degree=1,
    )
    if target_kind == "decompressed":
        target: Any = MultiKVCacheParams.from_params(
            {
                ATTN_CACHE_KEY: KVCacheConfig().to_params(
                    dtype=DType.bfloat16,
                    n_kv_heads=_HEADS,
                    head_dim=12,
                    num_layers=1,
                    devices=devices,
                    data_parallel_degree=1,
                    speculative_method="dspark",
                    num_draft_tokens=_DRAFT_WIDTH,
                ),
                STATE_CACHE_KEY: state,
            }
        )
    elif target_kind == "no_kda":
        target = _mla_leaf(24, 96, devices)
    else:
        target = MultiKVCacheParams.from_params(
            {
                ATTN_CACHE_KEY: _mla_leaf(24, 96, devices),
                STATE_CACHE_KEY: state,
            }
        )
    return MultiKVCacheParams.from_params(
        {
            "target": target,
            "draft": _mla_leaf(5, 64, devices),
        }
    )


def _declared_signature(
    kv_params: MultiKVCacheParams,
) -> list[TensorType | BufferType]:
    """The declared input list the driver builds, extent pair last."""
    device = kv_params.devices[0]
    return [
        TensorType(DType.int64, ["total_seq_len"], device=device),
        TensorType(DType.uint32, ["input_row_offsets_len"], device=device),
        TensorType(DType.int64, ["return_n_logits"], device=DeviceRef.CPU()),
        *kv_params.flattened_kv_inputs(),
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(DType.uint64, ["batch_size"], device=device),
        TensorType(DType.float32, ["batch_size"], device=device),
        TensorType(DType.int64, ["batch_size"], device=device),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(DType.int32, [2], device=DeviceRef.CPU()),
    ]


def test_the_export_contract_records_the_frozen_geometry() -> None:
    """The sidecar record Mach's validator checks the graph against.

    Derived from the built pairing itself: the devices, the traced signal
    buffers' byte width, both MLA leaves, the KDA state with its verify
    ring, the draft widths, and what the frozen export pins off. A pairing
    that is not the frozen one is refused here rather than recorded.
    """
    devices = [DeviceRef.CPU(0), DeviceRef.CPU(1)]
    kv_params = _pairing_kv_tree(devices)
    speculative = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=_DRAFT_WIDTH,
        use_greedy_acceptance=True,
    )
    draft_config = parse_k3_dspark_config(TINY_DRAFT_CONFIG)
    contract = export_contract(
        speculative,
        draft_config,
        None,
        kv_params,
        _declared_signature(kv_params),
        False,
    )
    ring = kv_params.children["target"]
    assert isinstance(ring, MultiKVCacheParams)
    state = ring.children[STATE_CACHE_KEY]
    assert isinstance(state, RecurrentStateParams)
    assert contract == {
        "devices": [0, 1],
        "signal_bytes": Signals.NUM_BYTES,
        "target": {
            "kind": "mla",
            "layers": 24,
            "kv_heads": 1,
            "head_dim": 576,
            "page_size": _PAGE_SIZE,
            "dtype": "DType.bfloat16",
        },
        "draft": {
            "kind": "mla",
            "layers": 5,
            "kv_heads": 1,
            "head_dim": 576,
            "page_size": _PAGE_SIZE,
            "dtype": "DType.bfloat16",
        },
        "state": {
            "layers": _NUM_KDA,
            "conv_dim": 3 * 4 * _HEAD_DIM,
            "conv_span": _CONV_KERNEL - 1,
            "heads": 4,
            "head_dim": _HEAD_DIM,
            "conv_dtype": "DType.bfloat16",
            "recurrent_dtype": "DType.float32",
            "ring": {
                "length": _RING_LEN,
                "record_stride": state.regions[2].row_shape[2],
                "dtype": "DType.float32",
                "scratch": True,
            },
        },
        "spec_k": _DRAFT_WIDTH,
        "block_size": _DRAFT_WIDTH,
        "verify_width": _DRAFT_WIDTH + 1,
        "sample_from_anchor": True,
        "refused": {
            "logit_penalties": True,
            "pipeline_stages": True,
            "in_thinking_phase": True,
            "structured_output": True,
            "stochastic_acceptance": True,
        },
    }


def test_the_export_contract_records_nothing_outside_the_frozen_contract() -> (
    None
):
    """A build MAX serves records no contract instead of raising.

    ``_driver_call`` calls the hook on every fused build of this pairing,
    serving included: the narrow-latent parity fixture builds its target
    through the decompressed (non-MLA) cache, and a target without KDA
    layers builds no state child at all. Both record nothing -- MAX keeps
    serving, and Mach's exporter fails its own missing-contract refusal.
    """
    devices = [DeviceRef.CPU(0), DeviceRef.CPU(1)]
    speculative = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=_DRAFT_WIDTH,
        use_greedy_acceptance=True,
    )
    draft_config = parse_k3_dspark_config(TINY_DRAFT_CONFIG)
    for outside in (
        _pairing_kv_tree(devices, with_ring=False),
        _pairing_kv_tree(devices, target_kind="decompressed"),
        _pairing_kv_tree(devices, target_kind="no_kda"),
    ):
        assert (
            export_contract(
                speculative,
                draft_config,
                None,
                outside,
                _declared_signature(outside),
                False,
            )
            is None
        )


def test_the_export_contract_raises_only_on_unbuildable_states() -> None:
    """A block past the draft width, or no trailing extent."""
    devices = [DeviceRef.CPU(0), DeviceRef.CPU(1)]
    speculative = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=_DRAFT_WIDTH,
        use_greedy_acceptance=True,
    )
    draft_config = parse_k3_dspark_config(TINY_DRAFT_CONFIG)
    kv_params = _pairing_kv_tree(devices)
    wide_block = replace(draft_config, block_size=_DRAFT_WIDTH + 1)
    with pytest.raises(ValueError, match="block_size"):
        export_contract(
            speculative,
            wide_block,
            None,
            kv_params,
            _declared_signature(kv_params),
            False,
        )
    signature = _declared_signature(kv_params)
    with pytest.raises(ValueError, match="extent pair"):
        export_contract(
            speculative,
            draft_config,
            None,
            kv_params,
            signature[:-1],
            False,
        )
    # A stochastic build is recorded as serving sampled acceptance, so the
    # frozen contract's validator can refuse the sidecar.
    stochastic = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=_DRAFT_WIDTH,
    )
    contract = export_contract(
        stochastic,
        draft_config,
        None,
        kv_params,
        signature,
        False,
    )
    record = cast(dict[str, object], contract)
    refused = cast(dict[str, bool], record["refused"])
    assert refused["stochastic_acceptance"] is False
