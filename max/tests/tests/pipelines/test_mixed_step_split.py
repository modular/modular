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
"""The mixed-step split record of a spec graph's KV leaves: the draft forwards
drop it, and the op operand that carries it is a host ``uint32[7]``."""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import fields
from typing import Any, cast

import numpy as np
import pytest
from max import mlir, tree
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn._graph_ops import (
    _mixed_step_split_value,
    flare_mla_decode_ragged,
    flare_mla_decode_ragged_scaled,
    flare_mla_prefill_plan,
    flare_mla_prefill_ragged,
    mla_prefill_graph,
)
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache.input_types import MIXED_STEP_SPLIT_SIZE
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockDriver,
)
from max.pipelines.speculative.driver import SequentialBatch, SequentialDriver
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
    decode_spec_decode_input_values,
    drop_mixed_step_split,
)
from max.pipelines.speculative.spec_target import Verified
from test_common.mixed_step_split_ops import record_operands

_Leaf = KVCacheInputsPerDevice[TensorValue, BufferValue]


def _unified_leaves() -> tuple[list[_Leaf], list[_Leaf]]:
    """The target and draft leaves a unified spec graph decodes."""
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        n_kv_heads=1,
    )
    unified = MultiKVCacheParams.from_params(
        {"target": kv_params, "draft": kv_params}
    )
    spec = SpecDecodeInputTypeSpec(devices=[DeviceRef.CPU()], distributed=False)
    with Graph(
        "decode",
        input_types=build_spec_decode_input_types(spec, kv_params=unified),
    ) as graph:
        decoded = decode_spec_decode_input_values(
            graph.inputs, spec, kv_params=unified
        )
        return list(decoded.kv("target")), list(decoded.kv("draft"))


def test_every_kv_leaf_declares_its_record() -> None:
    for leaves in _unified_leaves():
        assert leaves
        assert all(leaf.mixed_step_split is not None for leaf in leaves)


def test_dropping_the_records_keeps_other_leaves() -> None:
    target, draft = _unified_leaves()
    state = object()
    kept = drop_mixed_step_split([*target, *draft])
    assert all(leaf.mixed_step_split is None for leaf in kept)
    assert [leaf.kv_blocks for leaf in kept] == [
        leaf.kv_blocks for leaf in [*target, *draft]
    ]
    assert drop_mixed_step_split({"state": [state]}) == {"state": [state]}


@dataclasses.dataclass
class _KVOnlyBatch:
    """The fields of a :class:`SequentialBatch` the propose phase reads
    before it calls the proposer."""

    kv_collections: list[_Leaf]
    draft_kv_collections: list[_Leaf]
    passthrough_kv: dict[str, list[_Leaf]]
    num_accepted: TensorValue | None = None


class _ProposerCalled(Exception):
    def __init__(self, batch: _KVOnlyBatch) -> None:
        super().__init__()
        self.batch = batch


class _CapturingProposer:
    def prefill(self, batch: _KVOnlyBatch, *_: object) -> None:
        raise _ProposerCalled(batch)


def test_the_sequential_draft_sees_no_record() -> None:
    """Step 0 of the draft runs over the corrected rows, not the verify rows,
    so neither the target leaves it may read nor its own carry a record."""
    target, draft = _unified_leaves()
    batch = _KVOnlyBatch(
        kv_collections=target,
        draft_kv_collections=draft,
        passthrough_kv={"indexer": target},
    )
    # Only the propose phase runs, so the driver needs only its proposer.
    driver = SequentialDriver.__new__(SequentialDriver)
    vars(driver).update(
        _proposer=_CapturingProposer(), _draft_proposal="argmax"
    )
    with Graph("propose", input_types=[]):
        with pytest.raises(_ProposerCalled) as called:
            driver._propose(
                cast(
                    "SequentialBatch[TensorValue, BufferValue, Value[Any]]",
                    batch,
                ),
                cast("TensorValue", None),
                None,
                cast("TensorValue", None),
                seed=cast("TensorValue", None),
                temperature=cast("TensorValue", None),
                top_k=cast("TensorValue", None),
                top_p=cast("TensorValue", None),
            )
    seen = called.value.batch
    leaves = [
        *seen.kv_collections,
        *seen.draft_kv_collections,
        *seen.passthrough_kv["indexer"],
    ]
    assert all(leaf.mixed_step_split is None for leaf in leaves)


def _mla_leaf() -> KVCacheInputsPerDevice[TensorType, BufferType]:
    """One replica of an MLA cache's symbolic graph inputs."""
    params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        num_q_heads=1,
        devices=[DeviceRef.GPU(0)],
        page_size=64,
    )
    return params.get_symbolic_inputs()[0]


def test_the_mla_leaf_declares_its_record_last_and_no_mla_op_reads_it() -> None:
    """An MLA leaf carries the record but ignores it: the decode and prefill
    ops it feeds take no operand that is the record."""
    leaf = _mla_leaf()
    assert leaf.mixed_step_split == TensorType(
        DType.uint32, [MIXED_STEP_SPLIT_SIZE], device=DeviceRef.CPU()
    )
    # The KV ABI appends the record after every field a paged leaf holds, so
    # it is the last operand of the leaf, not one the MLA ops splice out.
    assert fields(KVCacheInputsPerDevice)[-1].name == "mixed_step_split"
    assert [f.name for f in fields(leaf)][-1] == "mixed_step_split"
    for op in _mla_op_graphs():
        assert not record_operands(op), mlir.StringAttr(
            op.attributes["symbol"]
        ).value
    # These two builders take operands this test does not build, so their
    # source stands in: neither names the record.
    for builder in (mla_prefill_graph, flare_mla_decode_ragged_scaled):
        assert "mixed_step_split" not in inspect.getsource(builder), (
            builder.__name__
        )


def _mla_op_graphs() -> list[mlir.Operation]:
    """Builds the MLA ops the leaf can feed and returns their custom ops."""
    params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        num_q_heads=1,
        devices=[DeviceRef.GPU(0)],
        page_size=64,
    )
    gpu = DeviceRef.GPU(0)
    leaf_types = tree.leaves(params.get_symbolic_inputs())
    q_type = TensorType(DType.bfloat16, ["total", 1, 576], gpu)
    offsets_type = TensorType(DType.uint32, ["rows"], gpu)
    scalar_type = TensorType(DType.int64, ["scalars"], gpu)
    found: list[mlir.Operation] = []
    with Graph(
        "mla_ops",
        input_types=[
            *leaf_types,
            q_type,
            offsets_type,
            scalar_type,
        ],
    ) as graph:
        n = len(leaf_types)
        kv = params.unflatten_kv_inputs(iter(graph.inputs[:n]))[0]
        q = graph.inputs[n].tensor
        offsets = graph.inputs[n + 1].tensor
        layer = ops.constant(0, dtype=DType.uint32, device=DeviceRef.CPU())
        scalars = graph.inputs[n + 2].tensor
        flare_mla_decode_ragged(
            params,
            q,
            offsets,
            kv,
            layer,
            MHAMaskVariant.CAUSAL_MASK,
            1.0,
            scalars,
        )
        flare_mla_prefill_ragged(
            params,
            q,
            q,
            q,
            offsets,
            offsets,
            offsets,
            kv,
            layer,
            MHAMaskVariant.CAUSAL_MASK,
            1.0,
        )
        flare_mla_prefill_plan(params, offsets, kv, layer, 4096)
    for op in graph._mlir_op.regions[0].blocks[0].operations:
        if op.name == "mo.custom" and mlir.StringAttr(
            op.attributes["symbol"]
        ).value.startswith(("attention.mla.decode", "attention.mla.prefill")):
            found.append(op)
    assert len(found) == 3, [
        mlir.StringAttr(op.attributes["symbol"]).value for op in found
    ]
    return found


class _BlockProposer:
    """A proposer that stops the driver where the draft reads its batch."""

    block_size = 2
    mask_token_id = 0
    samples_from_anchor = False
    supports_zero_draft_rows = False

    def materialize(self, batch: Any, *_: object) -> None:
        raise _ProposerCalled(batch)

    def embed_block(self, *_: object) -> object:
        raise AssertionError("materialize stops the driver")


def test_the_block_driver_draft_sees_no_record() -> None:
    """The block draft runs over the committed token, not the verify rows,
    so the batch the proposer reads must carry no split record."""
    target, draft = _unified_leaves()
    proposer = _BlockProposer()
    driver = BlockDriver.__new__(BlockDriver)
    vars(driver).update(
        _proposer=proposer,
        _verify=lambda batch: Verified(
            logits=cast("TensorValue", None),
            hidden=cast("TensorValue", None),
        ),
        # The stub accept hands back real-shaped commit lengths, so the
        # caches build; only the draft forward itself is stopped.
        _accept=lambda batch, logits, **_: Accepted(
            num_accepted=ops.constant(
                np.zeros((1,), dtype=np.int32),
                dtype=DType.int32,
                device=DeviceRef.CPU(),
            ),
            next_tokens=ops.constant(
                np.zeros((1,), dtype=np.int64),
                dtype=DType.int64,
                device=DeviceRef.CPU(),
            ),
            commit_lengths=ops.constant(
                np.zeros((1,), dtype=np.uint32),
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            ),
            is_prefill=ops.constant(
                np.zeros((1,), dtype=np.bool_),
                dtype=DType.bool,
                device=DeviceRef.CPU(),
            ),
        ),
        _bitmask_side_stream=False,
        _skips_draft_rows=False,
        _sampled=False,
        _replica_probs=False,
        _ctx_at_draft_cache_length=False,
        block_size=proposer.block_size,
        merger=RaggedTokenMerger(DeviceRef.CPU()),
        num_speculative_tokens=proposer.block_size - 1,
        _vocab_size=None,
        data_parallel_degree=1,
        devices=[DeviceRef.CPU()],
    )
    # A spec without skippable draft keeps _draft_rows off the decoded graph.
    vars(driver).update(
        _input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU()], distributed=False
        )
    )
    with Graph("block", input_types=[]):
        tokens = ops.constant(
            np.zeros((1,), dtype=np.int64),
            dtype=DType.int64,
            device=DeviceRef.CPU(),
        )
        offsets = ops.constant(
            np.array([0, 1], dtype=np.uint32),
            dtype=DType.uint32,
            device=DeviceRef.CPU(),
        )
        draft_tokens = ops.constant(
            np.zeros((1, 1), dtype=np.int64),
            dtype=DType.int64,
            device=DeviceRef.CPU(),
        )
        with pytest.raises(_ProposerCalled) as called:
            driver._run(
                tokens,
                offsets,
                draft_tokens,
                kv_collections=target,
                draft_kv_collections=draft,
                passthrough_kv={"indexer": target},
                return_n_logits=cast("TensorValue", None),
                seed=cast("TensorValue", None),
                temperature=cast("TensorValue", None),
                top_k=cast("TensorValue", None),
                max_k=cast("TensorValue", None),
                top_p=cast("TensorValue", None),
                min_top_p=cast("TensorValue", None),
            )
    batch = called.value.batch
    leaves = [
        *batch.kv_collections,
        *batch.draft_kv_collections,
        *batch.passthrough_kv["indexer"],
    ]
    assert all(leaf.mixed_step_split is None for leaf in leaves)


@pytest.mark.parametrize(
    "size", [MIXED_STEP_SPLIT_SIZE - 1, MIXED_STEP_SPLIT_SIZE + 1]
)
def test_a_record_of_another_size_is_refused(size: int) -> None:
    record = TensorType(DType.uint32, [size], device=DeviceRef.CPU())
    with Graph("mixed_step_split", input_types=[record]) as graph:
        with pytest.raises(
            ValueError, match=f"{MIXED_STEP_SPLIT_SIZE} elements"
        ):
            _mixed_step_split_value(graph.inputs[0].tensor)


@pytest.mark.parametrize(
    "record, field",
    [
        (
            TensorType(DType.int32, [MIXED_STEP_SPLIT_SIZE], DeviceRef.CPU()),
            "dtype",
        ),
        (
            TensorType(
                DType.uint32, [1, MIXED_STEP_SPLIT_SIZE], DeviceRef.CPU()
            ),
            "rank",
        ),
        (
            TensorType(DType.uint32, [MIXED_STEP_SPLIT_SIZE], DeviceRef.GPU()),
            "device",
        ),
    ],
    ids=["int32", "rank_2", "on_the_gpu"],
)
def test_a_record_off_the_host_layout_is_refused(
    record: TensorType, field: str
) -> None:
    with Graph("mixed_step_split", input_types=[record]) as graph:
        with pytest.raises(ValueError, match=f"mixed_step_split.{field}"):
            _mixed_step_split_value(graph.inputs[0].tensor)
