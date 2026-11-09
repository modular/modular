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
"""The DeepSeek ModuleV3 residual collectives on a simulated ``(dp, tp)`` mesh.

Two data-parallel replicas of four tensor-parallel CPUs: the collectives
after attention and after the MoE must run within each replica, and the
config refuses the layouts those collectives can't serve.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
    check_replica_layout,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.transformer_block import (
    ParallelismMode,
    attention_tp_axis,
    post_attention,
    post_mlp,
)
from max.pipelines.architectures.deepseekV3_modulev3.model_config import (
    DeepseekV3Config,
)
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
)

REPLICAS = 2
TP = 4
MESH = DeviceMesh(
    devices=tuple(CPU() for _ in range(REPLICAS * TP)),
    mesh_shape=(REPLICAS, TP),
    axis_names=("dp", "tp"),
)


def _on_mesh(
    fragments: list[np.ndarray[Any, Any]], placements: tuple[Placement, ...]
) -> Tensor:
    """Places fragment ``i`` on device ``i`` under ``placements``."""
    shards = []
    for fragment, device in zip(fragments, MESH.devices, strict=True):
        t = Tensor(fragment)
        if t.device != device:
            t = Tensor(storage=t.driver_tensor.to(device))
        shards.append(t.driver_tensor)
    return Tensor._from_shards(tuple(shards), DeviceMapping(MESH, placements))


def test_attention_tp_axis_is_the_tp_axis_or_the_only_axis() -> None:
    assert attention_tp_axis(MESH) == 1
    for name in ("tp", "dp"):
        one_axis = DeviceMesh(MESH.devices[:TP], (TP,), (name,))
        assert attention_tp_axis(one_axis) == 0
    no_tp = DeviceMesh(MESH.devices, (REPLICAS, TP), ("dp", "ep"))
    with pytest.raises(ValueError):
        attention_tp_axis(no_tp)


def test_tp_ep_collectives_stay_within_each_replica() -> None:
    """Each replica sums only its own ranks' attention partials.

    Peer-to-peer EP reduce-scatters the tokens over the ``tp`` axis after
    attention and all-gathers them after the MoE, so every device ends with
    its own replica's rows. (Allreduce EP is refused on this mesh; see
    :func:`test_replica_layout_refusals`.)

    Traces the collectives without compiling or running them. The replicas
    hold different token counts, so a collective that crossed replicas would
    fail to trace or leave a device with another replica's shape.
    """
    tokens, hidden = [8, 4], 3  # Per replica.
    residual = _on_mesh(
        [
            np.zeros((tokens[d], hidden), np.float32)
            for d in range(REPLICAS)
            for _ in range(TP)
        ],
        (Sharded(0), Replicated()),
    )
    attn_out = _on_mesh(
        [
            np.zeros((tokens[d], hidden), np.float32)
            for d in range(REPLICAS)
            for _ in range(TP)
        ],
        (Sharded(0), Partial()),
    )
    config = cast(
        DeepseekV3Config,
        SimpleNamespace(ep_config=SimpleNamespace(use_allreduce=False)),
    )

    with F.lazy():
        h = post_attention(ParallelismMode.TP_EP, config, residual, attn_out)
        out = post_mlp(ParallelismMode.TP_EP, config, h, h)
    assert not h.real and not out.real
    assert h.placements == (Sharded(0), Sharded(0))
    assert out.placements == (Sharded(0), Replicated())
    for i, (h_shard, out_shard) in enumerate(
        zip(h.local_shards, out.local_shards, strict=True)
    ):
        d = i // TP
        assert h_shard.shape == [tokens[d] // TP, hidden]
        assert out_shard.shape == [tokens[d], hidden]
        assert out_shard.dtype == DType.float32


def _check_layout(
    replicas: int,
    *,
    dcp: int = 1,
    ep: bool = True,
    use_allreduce: bool = False,
) -> None:
    config = cast(
        DeepseekV3_2Config,
        SimpleNamespace(decode_context_parallel_degree=dcp),
    )
    runtime: Any = SimpleNamespace(
        mesh=MESH,
        data_parallel_degree=replicas,
        ep=SimpleNamespace(use_allreduce=use_allreduce) if ep else None,
        ep_use_allreduce=use_allreduce,
    )
    check_replica_layout(config, runtime)


def test_replica_layout_accepts_p2p_ep_replicas_and_dcp_per_replica() -> None:
    _check_layout(REPLICAS)
    _check_layout(REPLICAS, dcp=TP)
    # One replica per device keeps either EP backend.
    _check_layout(REPLICAS * TP, use_allreduce=True)
    _check_layout(1, dcp=REPLICAS * TP, use_allreduce=True)


def test_replica_layout_refusals() -> None:
    with pytest.raises(NotImplementedError, match="needs expert parallelism"):
        _check_layout(REPLICAS, ep=False)
    # Allreduce EP's per-device partials need a sum over all eight devices;
    # each replica's post-MoE collective reduces only its own four.
    with pytest.raises(NotImplementedError, match="ep_use_allreduce"):
        _check_layout(REPLICAS, use_allreduce=True)
    with pytest.raises(
        NotImplementedError, match=r"attention tensor-parallel degree \(4"
    ):
        _check_layout(REPLICAS, dcp=2)
    with pytest.raises(
        NotImplementedError, match=r"attention tensor-parallel degree \(8"
    ):
        _check_layout(1, dcp=4)
