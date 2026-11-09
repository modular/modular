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
from max.driver import CPU
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    SymbolicDim,
    TensorValue,
)
from max.nn import _graph_ops as kernels
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    SplitKVHeadDims,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated


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
