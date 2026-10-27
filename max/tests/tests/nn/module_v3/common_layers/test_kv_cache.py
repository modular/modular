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

from typing import Any

import pytest
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMapping, DeviceMesh, Replicated
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    SymbolicDim,
    TensorValue,
)
from max.nn.legacy import kernels
from max.nn.legacy.kv_cache import MHAKVCacheParams, SplitKVHeadDims


def _collection(params: MHAKVCacheParams) -> PagedCacheValues:
    """The ModuleV3 cache over zero tensors shaped like ``params``' inputs."""
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
    per_device = params.unflatten_kv_inputs(iter(values))
    mapping = DeviceMapping(DeviceMesh.single(CPU()), (Replicated(),))
    return PagedCacheValues.from_upstream(per_device, mapping)


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
