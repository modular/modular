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
from typing import cast

import pytest
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
from max.nn.legacy.comm import Signals
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
)
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
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    SampledDraftSampler,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

_DP = 4


@pytest.fixture(params=[4, 8], autouse=True)
def _replica_count(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "_DP", request.param)


@contextmanager
def _virtual_devices(n_devices: int) -> Iterator[None]:
    """SM100 virtual devices: graph staging with no GPU attached."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(max(n_devices, 8))
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
            self.drafter, cast(DeepseekV41, None), self.config
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
            cast(BlockBatch, None),
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
                DeepseekV41,
                SimpleNamespace(head=SimpleNamespace(weight=head_weight)),
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
            cast(BlockBatch, None),
            block_hs,
            cast(Accepted, SimpleNamespace(next_tokens=anchors)),
            sampler,
        )
        # The sampled head returns the drafts alone; only the greedy arm
        # ever carries the confidence tuple beside them.
        return (
            Tensor.from_graph_value(cast("TensorValue", drafts)),
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


class _ReplicaHead(Module[..., tuple[Tensor, ...]]):
    """The proposer's per-replica ``head`` over a replicated LM head."""

    def __init__(
        self,
        drafter: DSparkDrafter,
        config: DeepseekV41Config,
        replica_probs: bool,
    ) -> None:
        super().__init__()
        self.drafter = drafter
        self.config = config
        self.replica_probs = replica_probs

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        mesh = self.drafter.mesh
        block_hs = Tensor.from_shard_values(
            inputs[:_DP], DeviceMapping(mesh, (Sharded(0),))
        )
        head_weight = Tensor.from_shard_values(
            inputs[_DP : 2 * _DP], DeviceMapping.replicated(mesh)
        )
        splits, next_tokens, seed, temperature, top_k, top_p = inputs[
            2 * _DP : 2 * _DP + 6
        ]
        signals = [x._graph_value for x in inputs[2 * _DP + 6 :]]
        target = SimpleNamespace(
            head=SimpleNamespace(weight=head_weight),
            replicated_head_weight=lambda: head_weight,
        )
        proposer = DSparkDeepseekV41Proposer(
            self.drafter,
            cast(DeepseekV41, target),
            self.config,
            per_replica_head=True,
            per_replica_probs=self.replica_probs,
        )
        sampler = SampledDraftSampler(
            seed=TensorValue(seed),
            temperature=TensorValue(temperature),
            top_k=TensorValue(top_k),
            top_p=TensorValue(top_p),
            vocab_size=self.config.vocab_size,
        )
        drafts = proposer.head(
            cast(
                BlockBatch,
                SimpleNamespace(
                    data_parallel_splits=splits, signal_buffers=signals
                ),
            ),
            block_hs,
            cast(
                Accepted, SimpleNamespace(next_tokens=TensorValue(next_tokens))
            ),
            sampler,
        )
        assert isinstance(drafts, TensorValue)
        # Drafts keep the drafter's dtype, as the vocab-sharded head's do.
        assert drafts.dtype == DType.int32
        if not self.replica_probs:
            return (
                Tensor.from_graph_value(drafts),
                Tensor.from_graph_value(sampler.distributions()),
            )
        dists = sampler.replica_distributions
        assert dists is not None
        for replica, dist in enumerate(dists):
            assert dist.device == DeviceRef.from_device(mesh.devices[replica])
        return (
            Tensor.from_graph_value(drafts),
            *(Tensor.from_graph_value(dist) for dist in dists),
        )


@pytest.mark.parametrize("replica_probs", [False, True])
def test_dp4_replica_head_samples_on_every_device(replica_probs: bool) -> None:
    """A per-replica sampled draft stages one sampler per device.

    Each rank draws its own rows every position from its whole copy of
    the LM head, so the graph holds a draw per device per position and
    never joins the head weight. With per-replica distributions each one
    stays on the device that drew it.
    """
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
            module = _ReplicaHead(drafter, config, replica_probs)
        inputs: list[TensorType | BufferType] = []
        inputs += [
            TensorType(
                DType.bfloat16,
                [f"dspark_block_rows_r{r}", config.hidden_size],
                refs[r],
            )
            for r in range(_DP)
        ]
        inputs += [
            TensorType(
                DType.bfloat16, [config.vocab_size, config.hidden_size], refs[r]
            )
            for r in range(_DP)
        ]
        inputs += [TensorType(DType.int64, [_DP + 1], DeviceRef.CPU())]
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
        inputs += Signals(refs).input_types()
        staged = str(module.trace(*inputs))
    draws = staged.count("sampler.gumbel_argmax_from_probs")
    assert draws == _DP * config.dspark_block_size
    # Only the drafted tokens, and the distributions unless each replica
    # keeps its own, cross the mesh.
    assert staged.count("mo.distributed.allgather") == (
        1 if replica_probs else 2
    )


class _GreedyHead(Module[..., tuple[Tensor, ...]]):
    """The proposer's greedy head, with scores off or on."""

    def __init__(
        self,
        drafter: DSparkDrafter,
        config: DeepseekV41Config,
        *,
        scores: bool = False,
        head_scores: bool = False,
    ):
        super().__init__()
        self.drafter = drafter
        self.config = config
        self.scores = scores
        self.head_scores = head_scores

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        mesh = self.drafter.mesh
        rows = DeviceMapping(mesh, (Sharded(0),))
        block_hs = Tensor.from_shard_values(inputs[:_DP], rows)
        head_weight = Tensor.from_shard_values(inputs[_DP : 2 * _DP], rows)
        anchors = inputs[2 * _DP]
        proposer = DSparkDeepseekV41Proposer(
            self.drafter,
            cast(
                DeepseekV41,
                SimpleNamespace(head=SimpleNamespace(weight=head_weight)),
            ),
            self.config,
            scores=self.scores,
            head_scores=self.head_scores,
        )
        headed = proposer.head(
            cast(BlockBatch, None),
            block_hs,
            cast(Accepted, SimpleNamespace(next_tokens=anchors)),
            ArgmaxDraftSampler(),
        )
        if isinstance(headed, tuple):
            return tuple(
                Tensor.from_graph_value(cast("TensorValue", t)) for t in headed
            )
        return (Tensor.from_graph_value(cast("TensorValue", headed)),)


def _greedy_head_inputs(
    config: DeepseekV41Config, refs: list[DeviceRef]
) -> list[TensorType]:
    """The greedy head's staged inputs: sharded hidden rows, the head's
    vocab shards, and the batch's anchors."""
    inputs: list[TensorType] = []
    inputs += [
        TensorType(DType.bfloat16, [f"rows_{r}", config.hidden_size], refs[r])
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
    inputs += [TensorType(DType.int32, ["batch_size"], refs[0])]
    return inputs


def test_the_head_free_source_takes_the_vocab_parallel_route_under_dp() -> None:
    """The head-free score rides the vocab-parallel draft: the same one
    allgather per position carries the excluded-mass column beside the
    ``(max, index)`` pair, and one combine over the whole block joins the
    shards' pieces into the second output."""
    config = dataclasses.replace(_small_config(), vocab_size=128)
    text = _text()
    with _virtual_devices(_DP):
        devices = tuple(Accelerator(i) for i in range(_DP))
        refs = [DeviceRef.from_device(d) for d in devices]
        mesh = DeviceMesh(devices, (_DP,), ("dp",))
        inputs = _greedy_head_inputs(config, refs)
        traces = []
        for scores in (False, True):
            with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
                drafter = DSparkDrafter(
                    config,
                    mesh,
                    rope=RopeParams.from_text_config(text),
                    data_parallel_degree=_DP,
                )
                module = _GreedyHead(drafter, config, scores=scores)
            traces.append(str(module.trace(*inputs)))
    # The scores ride the pair's own allgather: the count is unchanged.
    assert traces[1].count("mo.distributed.allgather") == traces[0].count(
        "mo.distributed.allgather"
    )
    # The fused per-shard pass emits no exp of its own; the combine adds
    # one exp and one log for the whole block.
    assert traces[1].count("mo.exp") - traces[0].count("mo.exp") == 1
    assert traces[1].count("mo.log") - traces[0].count("mo.log") == 1


def test_the_head_source_builds_no_head_free_score_op() -> None:
    """On the head source the graph adds the confidence GEMM per rank and
    none of the head-free work: no fused pieces launch, no combine exp or
    log, and no extra allgather column."""
    config = dataclasses.replace(_small_config(), vocab_size=128)
    text = _text()
    with _virtual_devices(_DP):
        devices = tuple(Accelerator(i) for i in range(_DP))
        refs = [DeviceRef.from_device(d) for d in devices]
        mesh = DeviceMesh(devices, (_DP,), ("dp",))
        inputs = _greedy_head_inputs(config, refs)
        traces = []
        for scores, mounted in ((False, False), (True, False), (True, True)):
            with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
                drafter = DSparkDrafter(
                    config,
                    mesh,
                    rope=RopeParams.from_text_config(text),
                    data_parallel_degree=_DP,
                    load_confidence_head=mounted,
                )
                module = _GreedyHead(
                    drafter, config, scores=scores, head_scores=mounted
                )
            traces.append(module.trace(*inputs))
    off, free, head_on = (str(t) for t in traces)
    assert "top_token_pieces" in free
    assert "top_token_pieces" not in head_on
    assert head_on.count("mo.exp") == off.count("mo.exp")
    assert head_on.count("mo.log") == off.count("mo.log")
    assert head_on.count("matmul") - off.count("matmul") == _DP
    # The head returns the drafts and one float32 score per position.
    assert len(traces[2].out_layouts) == 2


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
