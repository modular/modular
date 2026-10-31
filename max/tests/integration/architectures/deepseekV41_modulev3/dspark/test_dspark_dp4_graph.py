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
"""The DP4 DSpark context store builds its graph on virtual devices.

Under data parallelism the verify payload (taps, row offsets, positions)
and the sliding-window leaf are per-rank tensors whose row dims differ per
shard. The drafter's context store has to run per rank over those shards,
as the trunk's attention does; a mesh-level store hands the per-shard batch
dim to the range ops and fails at graph build, which a DP4/EP4 export hit.
The graph is only staged, never executed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from dspark_test_support import _full_hf, _small_config, _text
from max import functional as F
from max import tree
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import Module
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.spec_adapters import (
    WINDOW_LEAF,
    DSparkDeepseekV41Proposer,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

_DP = 4


@contextmanager
def _virtual_devices(n_devices: int) -> Iterator[None]:
    """SM100 virtual devices: graph staging with no GPU attached."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    try:
        yield
    finally:
        set_virtual_device_count(0)


class _ContextStore(Module[..., None]):
    """The proposer's ``materialize`` over per-rank graph inputs."""

    def __init__(
        self,
        drafter: DSparkDrafter,
        config: DeepseekV41Config,
        kv_params: MultiKVCacheParams,
    ) -> None:
        super().__init__()
        self.drafter = drafter
        self.config = config
        self.kv_params = kv_params

    def forward(self, *inputs: Tensor) -> None:
        # Traced from a copy of the module with graph-constant weights, so
        # the proposer is built here over that copy's drafter.
        proposer = DSparkDeepseekV41Proposer(
            self.drafter, cast(Any, None), self.config
        )
        mapping = DeviceMapping(self.drafter.mesh, (Sharded(0),))
        taps, offsets, positions = (
            Tensor.from_shard_values(inputs[i * _DP : (i + 1) * _DP], mapping)
            for i in range(3)
        )
        upstream = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in inputs[3 * _DP :])
        )
        # The payload is row-sharded as the trunk's DP forward leaves it;
        # materialize reads nothing off the batch.
        proposer.materialize(
            cast(Any, None),
            (taps, offsets, positions),
            tree.leaves(upstream[WINDOW_LEAF], leaf=KVCacheInputsPerDevice),
        )


def test_dp4_context_store_stages_per_rank() -> None:
    config = _small_config()
    text = _text()
    with _virtual_devices(_DP):
        devices = tuple(Accelerator(i) for i in range(_DP))
        refs = [DeviceRef.from_device(d) for d in devices]
        mesh = DeviceMesh(devices, (_DP,), ("dp",))
        kv_params = build_kv_params(
            V41Config.from_hf(_full_hf()),
            KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
            refs,
            data_parallel_degree=_DP,
        )
        with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
            drafter = DSparkDrafter(
                config,
                mesh,
                rope=RopeParams.from_text_config(text),
                data_parallel_degree=_DP,
            )
            module = _ContextStore(drafter, config, kv_params)
        taps_width = config.hidden_size * len(config.dspark_target_layer_ids)
        inputs: list[TensorType | BufferType] = []
        inputs += [
            TensorType(DType.bfloat16, [f"rows_{r}", taps_width], refs[r])
            for r in range(_DP)
        ]
        inputs += [
            TensorType(DType.uint32, [f"offsets_{r}"], refs[r])
            for r in range(_DP)
        ]
        inputs += [
            TensorType(DType.int32, [f"rows_{r}"], refs[r]) for r in range(_DP)
        ]
        inputs += kv_params.flattened_kv_inputs()
        staged = module.trace(*inputs)
    assert "window_kv_store" in str(staged)
