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
"""CPU mesh fixtures for speculative decoding contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.kv_cache.cache_params import MLAKVCacheParams
from max.pipelines.architectures.deepseekV3_2_modulev3.deepseekV3_2 import (
    DeepseekV3_2TextModel,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.sparse_mla import (
    Selection,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.nextn import (
    DeepseekV3_2NextN,
)
from max.pipelines.architectures.glm5_1_modulev3 import spec_adapters
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.speculative import driver as spec_driver
from max.pipelines.speculative.driver import (
    DistributedInputs,
    SequentialBatch,
)
from max.tensor import default_device
from max.tensor import realization_context as rc
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.tensor.tensor import Tensor, realization_context

_PENDING: list[tuple[Tensor, Tensor | object]] = []


def expect_values(actual: Tensor, expected: Tensor | object) -> None:
    """Checks ``actual`` against ``expected`` once the test's graph realizes.

    Values realize only when the realization context closes, after the code
    under test has run, so value checks queue here; placements are checked
    where they are built.
    """
    _PENDING.append((actual, expected))


@pytest.fixture(autouse=True)
def realize_contract() -> Iterator[None]:
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


def cpu_mesh(dp: int) -> DeviceMesh:
    """Builds a one-axis CPU mesh for the requested replica count."""
    return DeviceMesh((CPU(),) * dp, (dp,), ("dp" if dp > 1 else "tp",))


def dp_tp_mesh(dp: int, tp: int) -> DeviceMesh:
    """Groups consecutive CPU devices into tensor-parallel replicas."""
    return DeviceMesh((CPU(),) * (dp * tp), (dp, tp), ("dp", "tp"))


def batch_shards(values: list[list[list[float]]], mesh: DeviceMesh) -> Tensor:
    """Builds a batch tensor from per-device float32 values.

    Args:
        values: The rows on each device in mesh order.
        mesh: The mesh on which to place the shards.

    Returns:
        The tensor sharded on the data-parallel axis and replicated elsewhere.
    """
    mapping = DeviceMapping(
        mesh,
        tuple(
            Sharded(0)
            if name == "dp" and mesh.num_devices > 1
            else Replicated()
            for name in mesh.axis_names
        ),
    )
    return Tensor.from_shard_values(
        [Tensor(v, dtype=DType.float32) for v in values], mapping
    )


def _replicas(mesh: DeviceMesh) -> int:
    """Returns the number of data-parallel replicas."""
    if "dp" in mesh.axis_names:
        return mesh.axis_size("dp")
    return mesh.num_devices


def cpu_config(mesh: DeviceMesh) -> DeepseekV3_2Config:
    """Builds a decoder configuration with MLA caches on the given mesh."""
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
            # Every simulated device needs a whole number of query heads.
            num_q_heads=64,
        ),
        devices=devices,
        mesh=mesh,
        data_parallel_degree=_replicas(mesh),
        hidden_size=1,
        max_seq_len=128,
    )


def text_model(mesh: DeviceMesh) -> DeepseekV3_2TextModel:
    """Builds the decoder state needed by the adapter and projection checks."""
    model = object.__new__(DeepseekV3_2TextModel)
    model.mesh = mesh
    model.config = cpu_config(mesh)
    model.ep_config = None
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


def make_batch(
    mesh: DeviceMesh, counts: list[int], width: int = 2
) -> SequentialBatch[Tensor, Tensor, Tensor]:
    """Builds a batch with the requested row count per replica and tokens per row."""
    dp = _replicas(mesh)
    devices_per_replica = mesh.num_devices // dp
    tokens = Tensor(list(range(width * sum(counts))), dtype=DType.int64)
    offsets = Tensor(
        list(range(0, width * sum(counts) + 1, width)), dtype=DType.uint32
    )
    splits = Tensor(np.cumsum([0] + counts).astype(np.int64))
    # Every device of a replica holds that replica's cache, as the paged
    # caches' placement declares.
    caches = [
        _cache(rank, counts[rank // devices_per_replica])
        for rank in range(mesh.num_devices)
    ]
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
        devices=[DeviceRef.CPU()] * mesh.num_devices,
        data_parallel_degree=dp,
        vision_embeddings=[],
        vision_scatter_indices=[],
        merged_tokens=tokens,
        merged_offsets=offsets,
        merged_offsets_per_dev=[offsets] * mesh.num_devices,
        query_offsets_per_dev=[offsets] * mesh.num_devices,
        extra={},
        num_accepted=None,
    )


def _prefix_fixture(
    mesh: DeviceMesh, k: int
) -> tuple[Tensor, Tensor, Tensor, list[list[float]], list[int]]:
    """Builds one request per replica.

    Args:
        mesh: The simulated CPU mesh.
        k: The number of draft positions.

    Returns:
        Per-replica hidden rows, local query offsets, global verify lengths,
        expected gathered rows in global order, and request counts per replica.
    """
    dp = _replicas(mesh)
    devices_per_replica = mesh.num_devices // dp
    counts = [1] * dp
    rows = []
    offsets = []
    lengths: list[int] = []
    expected: list[list[float]] = []
    for rank in range(dp):
        # One prompt row plus K draft rows per request; the window the
        # per-row merge leaves holds only the first k_r of the drafts.
        values = [[float(rank * 100 + i)] for i in range(1 + k)]
        # Unequal k_r across the batch: 0, K, and spread between.
        k_r = [0, k, k // 2, max(k // 3, 1)][rank % 4] if dp > 1 else k // 2
        for _ in range(devices_per_replica):
            rows.append(Tensor(np.array(values, dtype=np.float32)))
            offsets.append(Tensor([0, 1 + k_r], dtype=DType.uint32))
        lengths.append(k_r)
        for t in range(k + 1):
            # The gathered rows keep h's column shape.
            expected.append([values[min(t, k_r)][0]])
    mapping = DeviceMapping(
        mesh,
        tuple(
            Sharded(0) if name == "dp" and dp > 1 else Replicated()
            for name in mesh.axis_names
        ),
    )
    h = Tensor.from_shard_values(rows, mapping)
    query = Tensor.from_shard_values(offsets, mapping)
    verify_lengths = Tensor(
        np.array(lengths, dtype=np.int32), dtype=DType.int32
    )
    return h, query, verify_lengths, expected, counts


def _spy_gather_prefix_rows() -> Callable[..., tuple[Tensor, Tensor]]:
    """Returns a ``gather_prefix_rows`` that checks the lengths' mesh.

    A simulated CPU mesh cannot reject a host tensor itself, so the spy
    checks the placement the real GPU failure turns on, then calls
    through.
    """
    real = spec_driver.gather_prefix_rows

    def spy(
        h: Tensor,
        input_row_offsets: Tensor,
        verify_lengths: Tensor,
        return_n_logits: Tensor,
        mesh: DeviceMesh,
        data_parallel_degree: int = 1,
    ) -> tuple[Tensor, Tensor]:
        assert verify_lengths.mapping.mesh.devices == mesh.devices
        assert verify_lengths.is_distributed == (mesh.num_devices > 1)
        return real(
            h,
            input_row_offsets,
            verify_lengths,
            return_n_logits,
            mesh,
            data_parallel_degree,
        )

    return spy


def _check_hidden_shards(shards: list[Tensor], expected: Tensor) -> None:
    """Checks that the hidden shards retain their per-device graph values.

    Args:
        shards: The hidden tensors passed through the adapter.
        expected: The distributed tensor supplying the shards in mesh order.
    """
    expected_shards = expected.local_shards
    assert len(shards) == len(expected_shards)
    for shard, want in zip(shards, expected_shards, strict=True):
        assert shard.graph_values[0] is want.graph_values[0]


def check_verified_prefix(
    dp: int,
    tp: int,
    k: int,
    *,
    target_hidden_rows: list[list[list[float]]] | None = None,
) -> None:
    """Checks target and draft prefix rows on the requested mesh.

    Args:
        dp: The number of data-parallel replicas.
        tp: The number of devices per replica.
        k: The number of draft positions.
        target_hidden_rows: The decoder stub's rows per device. Defaults to
            the draft fixture's hidden rows.
    """
    mesh = dp_tp_mesh(dp, tp) if tp > 1 else cpu_mesh(dp)
    h, query, verify_lengths, expected, counts = _prefix_fixture(mesh, k)
    batch = replace(
        make_batch(mesh, counts, width=1 + k),
        verify_lengths=verify_lengths,
        return_n_logits=Tensor([k + 1], dtype=DType.int64),
    )
    target_hidden = (
        batch_shards(target_hidden_rows, mesh)
        if target_hidden_rows is not None
        else h
    )
    language_model = text_model(mesh)

    def forward(
        tokens: Tensor,
        kv: spec_adapters.PagedCacheValues,
        indexer: spec_adapters.PagedCacheValues,
        n: Tensor,
        merged_offsets: Tensor,
        context: Tensor,
        boundaries: Tensor | None,
        host: Tensor,
        ep: object,
        hisparse: object,
        boundary: object = None,
        verify_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert hisparse is None
        assert boundary is None
        assert verify_lengths is not None

        def project(selected: Tensor) -> Tensor:
            assert all(p == Replicated() for p in selected.mapping.placements)
            return selected.local_shards[0]

        with patch.object(language_model, "_project", side_effect=project):
            logits, _ = language_model._variable_logits(
                h, query, n, verify_lengths, boundaries
            )
        return logits, query, target_hidden

    def decode(
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        tokens: Tensor,
        hidden: list[Tensor] | Tensor,
        reuse: list[Tensor] | None,
        selection_lag: int = 0,
        split_prefix: str = "mtp",
        prefill_context_parallel: bool = False,
    ) -> tuple[Tensor, Selection, Tensor]:
        assert isinstance(hidden, list)
        _check_hidden_shards(hidden, h)
        return h, h, query

    target = object.__new__(Glm5_1Decoder)
    target.language_model = language_model
    target.ep_config = None
    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.reuse = None
    with (
        patch.object(language_model, "forward", side_effect=forward),
        patch.object(proposer, "_decode", side_effect=decode),
        patch.object(
            proposer,
            "_project",
            side_effect=lambda rows: rows.local_shards[0],
        ),
        patch.object(
            spec_driver,
            "gather_prefix_rows",
            _spy_gather_prefix_rows(),
        ),
    ):
        verified = spec_adapters.Glm5_1Target(target)._verify(batch)
        draft_hidden = verified.hidden
        if target_hidden_rows is not None:
            _check_hidden_shards(verified.hidden, target_hidden)
            draft_hidden = list(h.local_shards)
        result = proposer.prefill(batch, batch.merged_tokens, draft_hidden)
    expect_values(verified.logits, expected)
    assert result.logits is not None
    expect_values(result.logits, expected)


def check_decode_replica_rows(dp: int, tp: int) -> None:
    """Checks the decoded embedding rows on every device in each replica."""
    mesh = dp_tp_mesh(dp, tp)
    counts = [2, 1]
    boundaries = np.cumsum([0] + counts).tolist()
    tokens = Tensor(list(range(2 * sum(counts))), dtype=DType.int64)
    hidden = [
        Tensor(np.zeros((2 * sum(counts), 1), dtype=np.float32))
        for _ in range(mesh.num_devices)
    ]
    attention = SimpleNamespace(
        graph_mode="prefill",
        create_mla_prefill_metadata=Mock(
            return_value=SimpleNamespace(buffer_lengths=None)
        ),
    )

    def decode(
        embedded: Tensor,
        carried: Tensor,
        kv: object,
        indexer: object,
        query: Tensor,
        freqs: Tensor,
        plan: object,
        ep: object,
        prev: Tensor | None,
        reuse: bool,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        for rank in range(mesh.num_devices):
            replica = rank // tp
            start, end = boundaries[replica], boundaries[replica + 1]
            expect_values(
                embedded.local_shards[rank],
                np.arange(2 * start, 2 * end, dtype=np.float32).reshape(-1, 1),
            )
        return carried, carried

    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.language_model = text_model(mesh)
    proposer.draft = object.__new__(DeepseekV3_2NextN)
    proposer.draft.config = cpu_config(mesh)
    proposer.draft.ep_config = None
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
        proposer._decode(make_batch(mesh, counts), tokens, hidden, None)
