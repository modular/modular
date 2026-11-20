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
"""Graph-construction coverage for DP4 DSpark fusion paths."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

from dspark_test_support import (
    _full_hf,
    _real_config,
    _small_config,
    _text,
)
from max import functional as F
from max import tree
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
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
    block_row_positions,
    window_starts,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative.block_driver import SampledDraftSampler
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


class _SampledHead(Module[..., tuple[Tensor, Tensor]]):
    """The proposer's ``head`` with the driver's sampled-draft sampler."""

    def __init__(self, drafter: DSparkDrafter, config: DeepseekV41Config):
        super().__init__()
        self.drafter = drafter
        self.config = config

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        mesh = self.drafter.mesh
        rows = DeviceMapping(mesh, (Sharded(0),))
        block_hs = Tensor.from_shard_values(inputs[:_DP], rows)
        head_weight = Tensor.from_shard_values(inputs[_DP : 2 * _DP], rows)
        anchors, seed, temperature, top_k, top_p = inputs[2 * _DP :]
        proposer = DSparkDeepseekV41Proposer(
            self.drafter,
            cast(
                Any, SimpleNamespace(head=SimpleNamespace(weight=head_weight))
            ),
            self.config,
        )
        sampler = SampledDraftSampler(
            seed=TensorValue(seed),
            temperature=TensorValue(temperature),
            top_k=TensorValue(top_k),
            top_p=TensorValue(top_p),
            vocab_size=self.config.vocab_size,
        )
        drafts = proposer.head(
            cast(Any, None),
            block_hs,
            cast(Any, SimpleNamespace(next_tokens=anchors)),
            sampler,
        )
        return (
            Tensor.from_graph_value(drafts),
            Tensor.from_graph_value(sampler.distributions()),
        )


def test_dp4_sampled_head_samples_on_one_device() -> None:
    """A sampled draft stages on a DP4 mesh, drawing on the first device.

    The head's step logits are replicated on the mesh, but the driver's
    sampler takes graph values and holds each row's sampling parameters on
    device 0; handing it the mesh tensor fails the graph build.
    """
    # The target head shards by vocabulary over the replica axis.
    config = dataclasses.replace(_small_config(), vocab_size=128)
    text = _text()
    with _virtual_devices(_DP):
        devices = tuple(Accelerator(i) for i in range(_DP))
        refs = [DeviceRef.from_device(d) for d in devices]
        mesh = DeviceMesh(devices, (_DP,), ("dp",))
        with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
            drafter = DSparkDrafter(
                config,
                mesh,
                rope=RopeParams.from_text_config(text),
                data_parallel_degree=_DP,
            )
            module = _SampledHead(drafter, config)
        inputs: list[TensorType] = []
        inputs += [
            TensorType(
                DType.bfloat16, [f"rows_{r}", config.hidden_size], refs[r]
            )
            for r in range(_DP)
        ]
        inputs += [
            TensorType(
                DType.bfloat16,
                [config.vocab_size // _DP, config.hidden_size],
                refs[r],
            )
            for r in range(_DP)
        ]
        inputs += [
            TensorType(dtype, ["batch_size"], refs[0])
            for dtype in (
                DType.int64,
                DType.uint64,
                DType.float32,
                DType.int64,
                DType.float32,
            )
        ]
        staged = module.trace(*inputs)
    assert "sampler.gumbel_argmax_from_probs" in str(staged)


class _StageBlock(Module[..., Tensor]):
    """The proposer's block forward, with the embed supplied as an input."""

    def __init__(
        self, drafter: DSparkDrafter, kv_params: MultiKVCacheParams
    ) -> None:
        super().__init__()
        self.drafter = drafter
        self.kv_params = kv_params

    def forward(self, *inputs: Tensor) -> Tensor:
        ids, hidden, offsets = inputs[:3]
        upstream = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in inputs[3:])
        )
        swa = PagedCacheValues.from_upstream(
            tree.leaves(upstream[WINDOW_LEAF], leaf=KVCacheInputsPerDevice),
            DeviceMapping.replicated(self.drafter.mesh),
        )
        positions = block_row_positions(
            offsets, swa.cache_lengths, ids.shape[0]
        )
        windows = self.drafter.windows(swa, window_starts(swa))
        return self.drafter.forward_block(
            ids, positions, windows, embed=lambda _: hidden
        )


def test_stages_lower_to_the_graph_api_models_fused_kernels() -> None:
    config = _real_config()
    stages = config.num_dspark_layers
    with _virtual_devices(1):
        device = Accelerator(0)
        ref = DeviceRef.from_device(device)
        mesh = DeviceMesh((device,), (1,), ("tp",))
        kv_params = build_kv_params(
            V41Config.from_hf(_full_hf()),
            KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
            [ref],
        )
        with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
            drafter = DSparkDrafter(
                config, mesh, rope=RopeParams.from_text_config(_text())
            )
            module = _StageBlock(drafter, kv_params)
        inputs: list[TensorType | BufferType] = [
            TensorType(DType.int32, ["block_rows"], ref),
            TensorType(DType.bfloat16, ["block_rows", config.hidden_size], ref),
            TensorType(DType.uint32, ["block_offsets"], ref),
            *kv_params.flattened_kv_inputs(),
        ]
        staged = str(module.trace(*inputs))
    kernels = Counter(re.findall(r'symbol = "([\w.]+)"', staged))
    assert kernels["attention.csa.rms_norm_quantize_q_store_kv"] == stages
    assert kernels["attention.csa.window_kv_store"] == 0
    assert kernels["matmul.mxfp8.rope_tail"] == stages
    assert kernels["moe.create.indices.with.slots"] == stages
    assert kernels["moe.create.indices"] == 0
