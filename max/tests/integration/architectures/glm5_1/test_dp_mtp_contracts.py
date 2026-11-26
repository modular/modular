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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
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
from max.pipelines.architectures.glm5_1_modulev3 import spec_adapters
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.lib.utils import split_replicated_batch
from max.pipelines.speculative.driver import (
    DraftStepInput,
    ReuseSpec,
)
from max.tensor.sharding import (
    DeviceMapping,
    Replicated,
    Sharded,
    Unknown,
)
from max.tensor.tensor import Tensor
from mtp_contract_helpers import (
    batch_shards,
    check_decode_replica_rows,
    check_verified_prefix,
    cpu_config,
    cpu_mesh,
    expect_values,
    make_batch,
    text_model,
)
from mtp_contract_helpers import realize_contract as realize_contract


def _check_host_lengths(lengths: Tensor, dp: int) -> None:
    """Each replica's own context length, on the host as the base DP
    decoder builds it so the graph stays capturable."""
    assert lengths.mapping.placements == (Replicated(),)
    assert all(d == CPU() for d in lengths.mapping.mesh.devices)
    for rank, shard in enumerate(lengths.local_shards):
        expect_values(shard, [rank + 3])


def _check_cache_mapping(
    cache: spec_adapters.PagedCacheValues, dp: int
) -> None:
    placement = Unknown() if dp > 1 else Replicated()
    assert cache.kv_blocks.mapping.placements == (placement,)
    for rank, shard in enumerate(cache.kv_blocks.local_shards):
        expect_values(shard, [float(rank)])


@pytest.mark.parametrize("dp", [1, 8])
def test_variable_output_modes_accept_dp(dp: int) -> None:
    model = text_model(cpu_mesh(dp))
    model.return_logits = ReturnLogits.VARIABLE
    model.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
    model.emit_last_token_logits = False
    DeepseekV3_2TextModel._check_output_modes(model)


@pytest.mark.parametrize("counts", [[2], [2, 0, 1, 0, 3, 0, 1, 0]])
def test_variable_logits_gather_rows_before_vocab_projection(
    counts: list[int],
) -> None:
    mesh = cpu_mesh(len(counts))
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

    model = text_model(mesh)
    with patch.object(model, "_project", side_effect=project):
        logits, logit_offsets = model._variable_logits(
            h, query, Tensor([2], dtype=DType.int64)
        )
    expect_values(logits, expected)
    expect_values(logit_offsets, np.arange(sum(counts) + 1) * 2)


@pytest.mark.parametrize("dp", [1, 8])
def test_verify_preserves_host_offsets_contexts_and_cache_ownership(
    dp: int,
) -> None:
    mesh = cpu_mesh(dp)
    counts = [1] * dp
    offsets = Tensor(list(range(dp + 1)), dtype=DType.uint32)
    hidden = batch_shards([[[float(rank)]] for rank in range(dp)], mesh)

    def forward(
        tokens: Tensor,
        kv: spec_adapters.PagedCacheValues,
        indexer: spec_adapters.PagedCacheValues,
        n: Tensor,
        query: Tensor,
        context: Tensor,
        boundaries: Tensor,
        host: Tensor,
        ep: object,
        hisparse: object,
        boundary: object,
        verify_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # The batch carries no HiSparse tables and no stage boundary, so the
        # verify passes neither.
        assert hisparse is None
        assert boundary is None
        assert verify_lengths is None
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
            expect_values(context, [3])
        _check_cache_mapping(kv, dp)
        _check_cache_mapping(indexer, dp)
        return tokens, offsets, hidden

    language_model = text_model(mesh)
    target = object.__new__(Glm5_1Decoder)
    target.language_model = language_model
    target.ep_config = None
    batch = make_batch(mesh, counts, width=1)
    with patch.object(language_model, "forward", side_effect=forward):
        spec_adapters.Glm5_1Target(target)._verify(batch)


@pytest.mark.parametrize("dp, tp", [(1, 1), (1, 2), (4, 1), (2, 2)])
def test_target_and_draft_gather_each_rows_verified_prefix(
    dp: int, tp: int
) -> None:
    check_verified_prefix(dp, tp, k=2)


def test_step_returns_global_hidden_for_driver_last_per_device_slicing() -> (
    None
):
    mesh = cpu_mesh(8)
    local = batch_shards([[[float(rank)]] for rank in range(8)], mesh)
    query = Tensor.from_shard_values(
        [Tensor([0, 1], dtype=DType.uint32) for _ in range(8)], local.mapping
    )
    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.draft = object.__new__(DeepseekV3_2NextN)
    proposer.draft.config = cpu_config(mesh)
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
            make_batch(mesh, [1] * 8, width=1),
            DraftStepInput(Tensor([1] * 8), list(local.local_shards), []),
            1,
        )
    for shard in result.hidden:
        expect_values(shard, np.arange(8).reshape(-1, 1))
    normalized = decode.call_args.args[2]
    assert normalized.mapping.placements == (Sharded(0),)


@pytest.mark.parametrize("counts", [[2], [2, 0, 1, 0, 3, 0, 1, 0]])
@pytest.mark.parametrize("reuse_selection", [False, True])
def test_decode_splits_embeddings_and_preserves_local_carries(
    counts: list[int], reuse_selection: bool
) -> None:
    mesh = cpu_mesh(len(counts))
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
        ep: object,
        prev: Tensor | None,
        reuse: bool,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        for rank, (start, end) in enumerate(itertools.pairwise(boundaries)):
            expect_values(
                embedded.local_shards[rank],
                np.array(local_values[rank], dtype=np.float32).reshape(-1, 1),
            )
            expect_values(
                carried.local_shards[rank], embedded.local_shards[rank]
            )
            expect_values(
                query.local_shards[rank], np.arange(end - start + 1) * 2
            )
            assert carried.mapping.placements == embedded.mapping.placements
            if prev is not None:
                assert prev.mapping.placements == embedded.mapping.placements
                expect_values(
                    prev.local_shards[rank], embedded.local_shards[rank]
                )
        if len(counts) > 1:
            _check_host_lengths(plan.buffer_lengths, len(counts))
        _check_cache_mapping(kv, len(counts))
        _check_cache_mapping(indexer, len(counts))
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
        proposer._decode(
            make_batch(mesh, counts),
            tokens,
            hidden,
            topk if reuse_selection else None,
        )


def test_prefill_predicates_the_draft_rows_under_pcp() -> None:
    """A PCP model's draft prefill predicates its rows as the target does.

    The NextN layer gets a predicated step (its rows split, no per-layer
    ``ops.cond``) whose one plan covers the step's PCP rows.
    """
    mesh = cpu_mesh(1)
    tokens = Tensor(list(range(4)), dtype=DType.int64)
    hidden = [Tensor(np.zeros((4, 1), dtype=np.float32))]
    attention = SimpleNamespace(
        graph_mode="decode", dcp_size=2, dcp_interleave=1
    )
    planned: list[Tensor] = []
    steps: list[object] = []

    def with_plan(
        step: spec_adapters.PrefillContextParallelStep,
        offsets: Tensor,
        rows: Tensor,
        *caches: object,
    ) -> spec_adapters.PrefillContextParallelStep:
        planned.append(rows)
        return step

    def decode(
        embedded: Tensor, carried: Tensor, *args: object, **kwargs: object
    ) -> tuple[Tensor, Tensor]:
        steps.append(kwargs["pcp"])
        return carried, carried

    config = cpu_config(mesh)
    config.prefill_context_parallel = True
    proposer = object.__new__(spec_adapters.Glm5_1MTPProposer)
    proposer.mesh = mesh
    proposer.language_model = text_model(mesh)
    proposer.draft = object.__new__(DeepseekV3_2NextN)
    proposer.draft.config = config
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
        patch.object(
            DeepseekV3_2Config,
            "mla_kv_params",
            SimpleNamespace(page_size=128),
        ),
        patch.object(
            spec_adapters.PrefillContextParallelStep,
            "with_plan",
            autospec=True,
            side_effect=with_plan,
        ),
    ):
        proposer._decode(
            make_batch(mesh, [2]),
            tokens,
            hidden,
            None,
            split_prefix="mtp_prefill",
            prefill_context_parallel=True,
        )
    (step,) = steps
    assert isinstance(step, spec_adapters.PrefillContextParallelStep)
    assert step.cut is not None
    assert step.namespace == "pcp_mtp"
    # The plan covers the PCP rows: the step's trailing ``cut`` rows.
    (rows,) = planned
    assert str(rows.shape[0]) == "pcp_mtp_pcp_rows"


def test_decode_splits_each_replicas_rows_across_its_devices() -> None:
    check_decode_replica_rows(dp=2, tp=2)
