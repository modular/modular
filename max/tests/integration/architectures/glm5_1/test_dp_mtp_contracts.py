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
"""Exercises the GLM-5.x ModuleV3 MTP adapters under data parallelism on a
simulated CPU mesh.

The decoder and caches are boundary doubles. Splitting, row selection and
collective concatenation execute the production Tensor/graph operations.
These tests do not establish sparse-attention or EP kernel correctness.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.kv_cache.cache_params import MLAKVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3_2_modulev3.deepseekV3_2 import (
    DeepseekV3_2TextModel,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.nextn import (
    DeepseekV3_2NextN,
)
from max.pipelines.architectures.deepseekV3_modulev3.deepseekV3 import (
    split_replicated_batch,
)
from max.pipelines.architectures.glm5_1_modulev3 import spec_adapters
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.speculative.driver import (
    DistributedInputs,
    DraftStepInput,
    ReuseSpec,
    SequentialBatch,
)
from max.tensor import default_device
from max.tensor import realization_context as rc
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Unknown,
)
from max.tensor.tensor import Tensor, realization_context

_PENDING: list[tuple[Tensor, Tensor | object]] = []


def _expect(actual: Tensor, expected: Tensor | object) -> None:
    """Checks ``actual`` against ``expected`` once the test's graph realizes.

    Values realize only when the realization context closes, after the code
    under test has run, so value checks queue here; placements are checked
    where they are built.
    """
    _PENDING.append((actual, expected))


@pytest.fixture(autouse=True)
def _realization() -> Iterator[None]:
    """Builds each test in one eager graph on the CPU, which the simulated
    mesh is all of, even on a GPU host, then runs its queued value checks."""
    _PENDING.clear()
    with (
        default_device(CPU()),
        rc.EagerRealizationContext() as ctx,
        realization_context(ctx),
    ):
        yield
    for actual, expected in _PENDING:
        if isinstance(expected, Tensor):
            expected = expected.to_numpy()
        np.testing.assert_array_equal(actual.to_numpy(), expected)
    _PENDING.clear()


def _mesh(dp: int) -> DeviceMesh:
    return DeviceMesh((CPU(),) * dp, (dp,), ("dp" if dp > 1 else "tp",))


def _shards(values: list[list[list[float]]], mesh: DeviceMesh) -> Tensor:
    return Tensor.from_shard_values(
        [Tensor(v, dtype=DType.float32) for v in values],
        _batch_mapping(mesh),
    )


def _batch_mapping(mesh: DeviceMesh) -> DeviceMapping:
    return DeviceMapping(
        mesh, (Sharded(0) if mesh.num_devices > 1 else Replicated(),)
    )


def _config(mesh: DeviceMesh) -> DeepseekV3_2Config:
    devices = [DeviceRef.CPU()] * mesh.num_devices
    return DeepseekV3_2Config(
        dtype=DType.float32,
        # MLA params because the target's caches are MLA; the concrete type
        # also satisfies the abstract hooks the base class declares.
        kv_params=MLAKVCacheParams(
            dtype=DType.float32,
            head_dim=1,
            num_layers=1,
            devices=devices,
            # Must divide the device count; 64 is the real model's.
            num_q_heads=64,
        ),
        devices=devices,
        mesh=mesh,
        data_parallel_degree=mesh.num_devices,
        hidden_size=1,
        max_seq_len=128,
    )


def _text_model(mesh: DeviceMesh) -> DeepseekV3_2TextModel:
    # These tests isolate adapter/output methods from decoder construction.
    model = object.__new__(DeepseekV3_2TextModel)
    model.mesh = mesh
    model.config = _config(mesh)
    return model


def _cache(rank: int, count: int) -> KVCacheInputsPerDevice[Tensor, Tensor]:
    return KVCacheInputsPerDevice(
        kv_blocks=Tensor([float(rank)]),
        cache_lengths=Tensor([rank + 3] * count, dtype=DType.uint32),
        lookup_table=Tensor(np.full((count, 1), rank, dtype=np.uint32)),
        max_prompt_length=Tensor([2], dtype=DType.uint32),
        max_cache_length=Tensor([rank + 5], dtype=DType.uint32),
        page_stride=Tensor([1], dtype=DType.int64),
    )


def _batch(
    mesh: DeviceMesh, counts: list[int], width: int = 2
) -> SequentialBatch[Tensor, Tensor, Tensor]:
    dp = mesh.num_devices
    tokens = Tensor(list(range(width * sum(counts))), dtype=DType.int64)
    offsets = Tensor(
        list(range(0, width * sum(counts) + 1, width)), dtype=DType.uint32
    )
    splits = Tensor(np.cumsum([0] + counts).astype(np.int64))
    caches = [_cache(rank, count) for rank, count in enumerate(counts)]
    return SequentialBatch(
        tokens=tokens,
        input_row_offsets=offsets,
        draft_tokens=tokens,
        signal_buffers=[],
        kv_collections=caches,
        draft_kv_collections=caches,
        passthrough_kv={"indexer": caches, "draft_indexer": caches},
        draft_cache_lengths=[kv.cache_lengths for kv in caches],
        return_n_logits=Tensor([1], dtype=DType.int64),
        distributed=DistributedInputs(offsets, offsets, offsets, splits)
        if dp > 1
        else None,
        batch_context_lengths=[
            Tensor([rank + 3], dtype=DType.int64) for rank in range(dp)
        ],
        ep_inputs=None,
        devices=[DeviceRef.CPU()] * dp,
        data_parallel_degree=dp,
        vision_embeddings=[],
        vision_scatter_indices=[],
        merged_tokens=tokens,
        merged_offsets=offsets,
        merged_offsets_per_dev=[offsets] * dp,
        query_offsets_per_dev=[offsets] * dp,
        extra={},
        num_accepted=None,
    )


def _check_host_lengths(lengths: Tensor, dp: int) -> None:
    """Each replica's own context length, on the host as the base DP
    decoder builds it so the graph stays capturable."""
    assert lengths.mapping.placements == (Replicated(),)
    assert all(d == CPU() for d in lengths.mapping.mesh.devices)
    for rank, shard in enumerate(lengths.local_shards):
        _expect(shard, [rank + 3])


def _check_cache_mapping(
    cache: spec_adapters.PagedCacheValues, dp: int
) -> None:
    placement = Unknown() if dp > 1 else Replicated()
    assert cache.kv_blocks.mapping.placements == (placement,)
    for rank, shard in enumerate(cache.kv_blocks.local_shards):
        _expect(shard, [float(rank)])


@pytest.mark.parametrize("dp", [1, 8])
def test_variable_output_modes_accept_dp(dp: int) -> None:
    model = _text_model(_mesh(dp))
    model.return_logits = ReturnLogits.VARIABLE
    model.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
    model.emit_last_token_logits = False
    DeepseekV3_2TextModel._check_output_modes(model)


@pytest.mark.parametrize("counts", [[2], [2, 0, 1, 0, 3, 0, 1, 0]])
def test_variable_logits_gather_rows_before_vocab_projection(
    counts: list[int],
) -> None:
    mesh = _mesh(len(counts))
    rows = []
    offsets = []
    expected = []
    for rank, count in enumerate(counts):
        # Two tokens per request, select both; rank sentinels expose ordering.
        values = [[float(rank * 100 + row)] for row in range(2 * count)]
        rows.append(Tensor(np.array(values, dtype=np.float32).reshape(-1, 1)))
        offsets.append(
            Tensor(list(range(0, 2 * count + 1, 2)), dtype=DType.uint32)
        )
        expected.extend(values)
    mapping = DeviceMapping(
        mesh, (Sharded(0) if len(counts) > 1 else Replicated(),)
    )
    h = Tensor.from_shard_values(rows, mapping)
    query = Tensor.from_shard_values(offsets, mapping)

    def project(selected: Tensor) -> Tensor:
        assert selected.mapping.placements == (Replicated(),)
        return selected.local_shards[0]

    model = _text_model(mesh)
    with patch.object(model, "_project", side_effect=project):
        logits, logit_offsets = model._variable_logits(
            h, query, Tensor([2], dtype=DType.int64)
        )
    _expect(logits, expected)
    _expect(logit_offsets, np.arange(sum(counts) + 1) * 2)


@pytest.mark.parametrize("dp", [1, 8])
def test_verify_preserves_host_offsets_contexts_and_cache_ownership(
    dp: int,
) -> None:
    mesh = _mesh(dp)
    counts = [1] * dp
    offsets = Tensor(list(range(dp + 1)), dtype=DType.uint32)
    hidden = _shards([[[float(rank)]] for rank in range(dp)], mesh)

    def forward(
        tokens: Tensor,
        kv: spec_adapters.PagedCacheValues,
        indexer: spec_adapters.PagedCacheValues,
        n: Tensor,
        query: Tensor,
        context: Tensor,
        boundaries: Tensor,
        host: Tensor,
        comm: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if dp > 1:
            # Exercise the actual split: uint32 host bounds fail constructing
            # the mixed static/runtime slice indices before a decoder runs.
            embedded = Tensor.from_shard_values(
                [tokens.reshape((dp, 1))] * dp, DeviceMapping.replicated(mesh)
            )
            split_replicated_batch(
                embedded,
                Tensor.from_shard_values(
                    [query] * dp, DeviceMapping.replicated(mesh)
                ),
                host,
                boundaries,
                DeviceMapping(mesh, (Sharded(0),)),
            )
            assert host.dtype == DType.int64
            _check_host_lengths(context, dp)
        else:
            _expect(context, [3])
        _check_cache_mapping(kv, dp)
        _check_cache_mapping(indexer, dp)
        return tokens, offsets, hidden

    language_model = _text_model(mesh)
    target = object.__new__(Glm5_1Decoder)
    target.language_model = language_model
    target.ep_batch_manager = None
    batch = _batch(mesh, counts, width=1)
    with patch.object(language_model, "forward", side_effect=forward):
        spec_adapters.Glm5_1Target(target)._verify(batch)


def test_step_returns_global_hidden_for_driver_last_per_device_slicing() -> (
    None
):
    mesh = _mesh(8)
    local = _shards([[[float(rank)]] for rank in range(8)], mesh)
    query = Tensor.from_shard_values(
        [Tensor([0, 1], dtype=DType.uint32) for _ in range(8)], local.mapping
    )
    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.draft = object.__new__(DeepseekV3_2NextN)
    proposer.draft.config = _config(mesh)
    # What __init__ declares outside decode context parallelism: reuse step
    # 0's selection.
    proposer.reuse = ReuseSpec(dim=1, split_prefix="mtp_topk")
    with (
        patch.object(
            proposer.draft,
            "shared_head_norm",
            side_effect=lambda h: h,
            create=True,
        ),
        patch.object(
            proposer, "_decode", return_value=(local, local, query)
        ) as decode,
        patch.object(
            proposer, "_project", side_effect=lambda h: h.local_shards[0]
        ),
    ):
        result = proposer.step(
            _batch(mesh, [1] * 8, width=1),
            DraftStepInput(Tensor([1] * 8), list(local.local_shards), []),
            1,
        )
    for shard in result.hidden:
        _expect(shard, np.arange(8).reshape(-1, 1))
    normalized = decode.call_args.args[2]
    assert normalized.mapping.placements == (Sharded(0),)


@pytest.mark.parametrize("counts", [[2], [2, 0, 1, 0, 3, 0, 1, 0]])
@pytest.mark.parametrize("reuse_selection", [False, True])
def test_decode_splits_embeddings_and_preserves_local_carries(
    counts: list[int], reuse_selection: bool
) -> None:
    mesh = _mesh(len(counts))
    boundaries = np.cumsum([0] + counts).tolist()
    tokens = Tensor(list(range(2 * sum(counts))), dtype=DType.int64)
    local_values = []
    for start, end in itertools.pairwise(boundaries):
        local_values.append([[float(v)] for v in range(2 * start, 2 * end)])
    hidden = [
        Tensor(np.array(v, dtype=np.float32).reshape(-1, 1))
        for v in local_values
    ]
    topk = [
        Tensor(np.array(v, dtype=np.int32).reshape(-1, 1)) for v in local_values
    ]
    metadata = SimpleNamespace(buffer_lengths=None)
    attention = SimpleNamespace(
        graph_mode="prefill",
        create_mla_prefill_metadata=Mock(return_value=metadata),
    )

    def decode(
        embedded: Tensor,
        carried: Tensor,
        kv: spec_adapters.PagedCacheValues,
        indexer: spec_adapters.PagedCacheValues,
        query: Tensor,
        freqs: Tensor,
        plan: SimpleNamespace,
        comm: object,
        prev: Tensor | None,
        reuse: bool,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        for rank, (start, end) in enumerate(itertools.pairwise(boundaries)):
            _expect(
                embedded.local_shards[rank],
                np.array(local_values[rank], dtype=np.float32).reshape(-1, 1),
            )
            _expect(carried.local_shards[rank], embedded.local_shards[rank])
            _expect(query.local_shards[rank], np.arange(end - start + 1) * 2)
            assert carried.mapping.placements == embedded.mapping.placements
            if prev is not None:
                assert prev.mapping.placements == embedded.mapping.placements
                _expect(prev.local_shards[rank], embedded.local_shards[rank])
        if len(counts) > 1:
            _check_host_lengths(plan.buffer_lengths, len(counts))
        _check_cache_mapping(kv, len(counts))
        _check_cache_mapping(indexer, len(counts))
        return carried, carried

    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.language_model = _text_model(mesh)
    proposer.draft = object.__new__(DeepseekV3_2NextN)
    proposer.draft.config = _config(mesh)
    proposer.draft.ep_batch_manager = None
    with (
        patch.object(
            proposer.language_model,
            "embed_tokens",
            side_effect=lambda t: t.cast(DType.float32).reshape((-1, 1)),
            create=True,
        ),
        patch.object(
            proposer.language_model,
            "rope",
            SimpleNamespace(freqs_cis=Tensor([[1.0]])),
            create=True,
        ),
        patch.object(
            proposer.draft,
            "decoder_layer",
            SimpleNamespace(self_attn=attention),
            create=True,
        ),
        patch.object(proposer.draft, "forward", side_effect=decode),
    ):
        proposer._decode(
            _batch(mesh, counts),
            tokens,
            hidden,
            topk if reuse_selection else None,
        )
