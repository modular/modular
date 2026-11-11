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
"""Tests for the ModuleV3 distributed ``PagedCacheValues``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from max import functional as F
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    SymbolicDim,
    TensorValue,
)
from max.nn import _graph_ops as kernels
from max.nn.common_layers.kv_cache import (
    PagedCacheValues,
    kv_input_values,
    per_device_caches,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    SplitKVHeadDims,
)
from max.tensor import Tensor
from max.tensor.realization_context import GraphRealizationContext
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated
from max.tensor.tensor import realization_context


def _per_device(
    params: MHAKVCacheParams,
) -> tuple[KVCacheInputsPerDevice[TensorValue, BufferValue], ...]:
    """One cache per device, over zero tensors shaped like ``params``'."""
    values: list[BufferValue | TensorValue] = []
    for field_type in tree.leaves(params.get_symbolic_inputs()):
        shape = [
            1 if isinstance(d, SymbolicDim) else int(d)
            for d in field_type.shape
        ]
        t = Tensor.zeros(
            shape, dtype=field_type.dtype, device=field_type.device.to_device()
        )
        is_buffer = isinstance(field_type, BufferType)
        values.append(BufferValue(t) if is_buffer else TensorValue(t))
    return params.unflatten_kv_inputs(iter(values))


def _collection(params: MHAKVCacheParams) -> PagedCacheValues:
    """The ModuleV3 cache over zero tensors shaped like ``params``' inputs."""
    mapping = DeviceMapping(DeviceMesh.single(CPU()), (Replicated(),))
    return PagedCacheValues.from_upstream(_per_device(params), mapping)


def test_split_marker_survives_to_graph_values() -> None:
    """The marker the cache ops key on reaches the graph values a ModuleV3
    model hands them, so an op that cannot read the layout still refuses it."""
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=192,
        v_head_dim=128,
        num_layers=2,
        devices=[DeviceRef.CPU()],
        page_size=128,
    )
    with F.lazy():
        graph_values = _collection(params).to_graph_values()
    assert graph_values.split_head_dims == SplitKVHeadDims()
    unused: dict[str, Any] = {
        "x_k": None,
        "valid_lengths": None,
        "layer_idx": None,
    }
    with pytest.raises(ValueError, match="head dims differ"):
        kernels.store_k_cache_padded(kv_collection=graph_values, **unused)


def test_a_device_without_the_split_record_is_refused() -> None:
    """The record spreads one value per device, so a device that drops it is
    a mismatch here rather than a field shorter than the replica."""
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=192,
        v_head_dim=128,
        num_layers=2,
        devices=[DeviceRef.CPU()],
        page_size=128,
    )
    mapping = DeviceMapping(
        DeviceMesh((CPU(), CPU()), (2,), ("tp",)), (Replicated(),)
    )
    with F.lazy():
        (only,) = _per_device(params)
        assert only.mixed_step_split is not None
        with pytest.raises(AssertionError, match="every device of a replica"):
            PagedCacheValues.from_upstream(
                [only, dataclasses.replace(only, mixed_step_split=None)],
                mapping,
            )


def _cpu_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=8,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=16,
    )


def test_per_device_caches_of_real_tensors() -> None:
    """An eager decoder's real tensors become a cache whose blocks the
    attention ops can write."""
    params = _cpu_params()
    tensors = _real_inputs(params)

    (cache,) = per_device_caches(params, tensors)

    assert isinstance(cache, PagedCacheValues)
    assert any(cache.kv_blocks is t for t in tensors)
    with F.lazy():
        assert isinstance(cache.to_graph_values().kv_blocks, BufferValue)


def test_per_device_caches_while_tracing() -> None:
    """A traced decoder gets the cache as graph values."""
    params = _cpu_params()
    graph = Graph("kv", input_types=tree.leaves(params.get_symbolic_inputs()))
    with (
        GraphRealizationContext(graph) as ctx,
        realization_context(ctx),
    ):
        tensors = [Tensor.from_graph_value(v) for v in graph.inputs]

        (cache,) = per_device_caches(params, tuple(tensors))

    assert not isinstance(cache, PagedCacheValues)
    assert isinstance(cache.kv_blocks, BufferValue)


def _real_inputs(params: MHAKVCacheParams) -> tuple[Tensor, ...]:
    """Zero tensors with data, shaped like ``params``' cache inputs."""
    tensors = []
    for field_type in tree.leaves(params.get_symbolic_inputs()):
        shape = [
            1 if isinstance(d, SymbolicDim) else int(d)
            for d in field_type.shape
        ]
        tensors.append(
            Tensor(storage=Buffer.zeros(shape, field_type.dtype, CPU()))
        )
    return tuple(tensors)


def test_kv_input_values_keeps_real_tensors() -> None:
    tensors = _real_inputs(_cpu_params())
    values = kv_input_values(tensors)
    assert len(values) == len(tensors)
    assert all(v is t for v, t in zip(values, tensors, strict=True))


def test_from_upstream_keeps_a_real_cache_writable() -> None:
    """One device's cache of real tensors stays those tensors, so the
    attention ops still get its blocks as a buffer."""
    params = _cpu_params()
    tensors = _real_inputs(params)
    per_device = params.unflatten_kv_inputs(iter(kv_input_values(tensors)))
    mapping = DeviceMapping(DeviceMesh.single(CPU()), (Replicated(),))

    cache = PagedCacheValues.from_upstream(per_device, mapping)

    assert any(cache.kv_blocks is t for t in tensors)
    with F.lazy():
        assert isinstance(cache.to_graph_values().kv_blocks, BufferValue)


def test_from_upstream_keeps_each_device_real_cache() -> None:
    """Several devices' real caches stay their own buffers, so the attention
    ops write each device's blocks."""
    params = _cpu_params()
    inputs = [_real_inputs(params) for _ in range(2)]
    per_device = [
        params.unflatten_kv_inputs(iter(kv_input_values(values)))[0]
        for values in inputs
    ]
    mesh = DeviceMesh((CPU(), CPU()), (2,), ("tp",))

    cache = PagedCacheValues.from_upstream(
        per_device, DeviceMapping(mesh, (Replicated(),))
    )

    assert cache.kv_blocks.real
    for shard, values in zip(cache.kv_blocks.buffers, inputs, strict=True):
        assert any(shard is t.driver_tensor for t in values)
