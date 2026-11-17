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
"""The DSpark drafter's FFN and attention split over the tensor-parallel mesh.

The drafter is built on one mesh axis -- ``tp`` beside the target's tensor
parallelism, ``dp`` when the target runs data parallel -- and its projections
used to be whole on every rank: 2.72 GB of bf16 weight traffic per rank per
step for the FFN against the 340 MB a tensor-parallel split reads, and 672 MB
against 121 MB for the attention. Its FFN now takes the target's own dense-FFN
layout (gate and up column-parallel, down row-parallel plus one all-reduce),
which is also what vLLM's Mistral Large 4 drafter does with its
``MergedColumnParallelLinear`` gate/up and its ``RowParallelLinear`` down; its
attention takes the target's head-axis split (``wq`` and ``wkv_b`` on their
head-major rows, ``wo`` on its head-major columns, plus one all-reduce), while
the latent projection and its norm stay whole.

What is pinned here, at eight ranks on virtual B200s (staging only: no compile,
no execution, no GPU):

* the FFN's and attention's head-major projections take the target's splits,
  and the latent projection and norm stay whole;
* the staged intermediate -- the width the down projection contracts -- is one
  rank's slice rather than the whole inner dimension;
* the block lands two all-reduces per layer -- attention's out-projection and
  the FFN's down projection -- so the residual stream never sees a partial
  sum;
* a one-device mesh stays whole even under a ``tp`` name, so single-device
  serving carries neither a one-way shard nor its all-reduce;
* a data-parallel mesh leaves every projection whole, since its ranks are
  replicas and nothing would complete a row-parallel partial.
"""

from __future__ import annotations

import pytest
from dspark_paged_fixtures import PagedStep
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.mistral_large4_modulev3.dspark import (
    DSparkDrafter,
    DSparkDrafterConfig,
    build_dspark_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4Yarn,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    TensorLayout,
)

DEVICES = 8
"""The deployment's rank count: one node of B200s."""

HIDDEN = 8
HEADS = 8
NOPE = 2
ROPE = 2
V_HEAD = 4
LATENT = 6
MARKOV_RANK = 4
VOCAB = 12
DEPTH = 3
WINDOW = 4
"""A miniature drafter: the trace reads shapes, not numbers, so it stays small."""

INNER = 2 * HIDDEN
"""The gated inner width, which the tensor-parallel split divides eight ways."""

ALL_REDUCE = "mo.distributed.allreduce.sum"
"""The residual collectives: the FFN's inner width and attention's heads."""

HEADS_PER_RANK = HEADS // DEVICES
"""The query heads one rank holds, which every attention projection follows."""


def _yarn() -> MistralLarge4Yarn:
    """The drafter's rope geometry, in a shape small enough to trace fast."""
    return MistralLarge4Yarn(
        rope_theta=10000.0,
        factor=2048.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        attention_factor=1.0,
        lerope=False,
        lerope_alpha_dim=ROPE // 2,
    )


def _config() -> DSparkDrafterConfig:
    """A miniature drafter: one layer, three rows, a four-slot window."""
    return DSparkDrafterConfig(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        kv_lora_rank=LATENT,
        intermediate_size=INNER,
        num_layers=1,
        depth=DEPTH,
        window=WINDOW,
        target_layers=(1,),
        markov_rank=MARKOV_RANK,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        softcap=30.0,
        yarn=_yarn(),
    )


def _mesh(axis: str, devices: int = DEVICES) -> DeviceMesh:
    """``devices`` virtual B200s on one mesh axis, named for the parallelism."""
    return DeviceMesh(
        tuple(Accelerator(rank) for rank in range(devices)),
        (devices,),
        (axis,),
    )


def _stage(axis: str, devices: int = DEVICES) -> tuple[DSparkDrafter, str]:
    """Builds the drafter on an ``axis``-named mesh and traces its paged block."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(devices)
    try:
        mesh = _mesh(axis, devices)
        config = _config()
        kv_params = build_dspark_kv_params(
            KVCacheConfig(kv_cache_page_size=128),
            devices=[DeviceRef.GPU(rank) for rank in range(devices)],
            config=config,
            speculative=SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=DEPTH
            ),
        )
        mapping = DeviceMapping(mesh, (Replicated(),))
        borrowed = TensorLayout(DType.bfloat16, [VOCAB, HIDDEN], mapping)
        input_types = [
            TensorLayout(DType.bfloat16, ["rows", config.tap_width], mapping),
            TensorLayout(DType.int32, ["rows"], mapping),
            TensorLayout(DType.uint32, ["offsets_len"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            borrowed,
            borrowed,
            *kv_params.flattened_kv_inputs(),
        ]
        with default_device(mesh), default_dtype(DType.bfloat16), F.lazy():
            step = PagedStep(DSparkDrafter(config), kv_params, mesh)
            return step.drafter, str(step.trace(*input_types))
    finally:
        set_virtual_device_count(0)


def _stage_dense(axis: str) -> tuple[DSparkDrafter, str]:
    """Builds the drafter on an ``axis``-named mesh and traces its dense entry."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(DEVICES)
    try:
        mesh = _mesh(axis)
        config = _config()
        mapping = DeviceMapping(mesh, (Replicated(),))
        borrowed = TensorLayout(DType.bfloat16, [VOCAB, HIDDEN], mapping)
        input_types = [
            TensorLayout(DType.bfloat16, ["n_ctx", config.tap_width], mapping),
            TensorLayout(DType.int32, ["n_ctx"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            borrowed,
            borrowed,
        ]
        with default_device(mesh), default_dtype(DType.bfloat16), F.lazy():
            drafter = DSparkDrafter(config)
            return drafter, str(drafter.trace(*input_types))
    finally:
        set_virtual_device_count(0)


@pytest.fixture(scope="module")
def tp8() -> tuple[DSparkDrafter, str]:
    """The drafter and its paged block graph, on an eight-rank TP mesh."""
    return _stage("tp")


def test_the_ffn_takes_the_targets_dense_split(
    tp8: tuple[DSparkDrafter, str],
) -> None:
    """Gate and up shard their output rows, down its input columns."""
    drafter, _ = tp8
    ffn = drafter.backbone.layers[0].feed_forward
    assert ffn.tensor_parallel
    for gate_or_up in (ffn.w1, ffn.w3):
        assert gate_or_up.weight.mapping.placements == (Sharded(0),)
    assert ffn.w2.weight.mapping.placements == (Sharded(1),)


def test_the_attention_takes_the_targets_head_split(
    tp8: tuple[DSparkDrafter, str],
) -> None:
    """The head axis splits every projection that has one, and only those.

    ``wq``'s rows and ``wkv_b``'s rows are the head axis, so both shard on
    their output; ``wo`` consumes ``heads * v_head_dim``, so it shards on its
    input and its all-reduce completes the sum. The latent projection and the
    latent norm cannot split -- the 576-wide latent is not a head axis -- so
    both stay whole and every rank derives the same latent key.
    """
    drafter, _ = tp8
    attention = drafter.backbone.layers[0].attention
    assert attention.tensor_parallel
    assert attention.wq.weight.mapping.placements == (Sharded(0),)
    assert attention.wkv_b.weight.mapping.placements == (Sharded(0),)
    assert attention.wo.weight.mapping.placements == (Sharded(1),)
    assert attention.wkv_a_with_mqa.weight.mapping.placements == (Replicated(),)
    assert attention.kv_a_norm.weight.mapping.placements == (Replicated(),)


def test_the_block_lands_two_allreduces_per_layer(
    tp8: tuple[DSparkDrafter, str],
) -> None:
    """The residual stream never sees a partial sum.

    Two projections are row-parallel -- the FFN's down and attention's out -- so
    each layer ends with two collectives and none of them reaches the residual
    add as a partial. The whole-weight tables this fixture borrows add none,
    which makes the count exactly twice the layer count.
    """
    _, text = tp8
    found = text.count(ALL_REDUCE)
    assert found == 2 * _config().num_layers, (found, ALL_REDUCE)


def test_the_attention_contracts_one_rank_of_the_heads(
    tp8: tuple[DSparkDrafter, str],
) -> None:
    """The staged projections are one rank's heads wide, not all of them.

    Per rank ``wq`` is ``[HEADS / 8 * qk_head_dim, HIDDEN]`` and ``wo`` is
    ``[HIDDEN, HEADS / 8 * v_head_dim]``; whole they would carry the full head
    count on both sides. The contraction ``wo`` performs is where the bytes
    show: ``HEADS / 8 * v_head_dim`` terms instead of ``HEADS * v_head_dim``.
    """
    _, text = tp8
    qk_head_dim = NOPE + ROPE
    staged = [
        f"!mo.tensor<[{HEADS_PER_RANK * qk_head_dim}, {HIDDEN}], bf16, gpu:0>",
        f"!mo.tensor<[{HIDDEN}, {HEADS_PER_RANK * V_HEAD}], bf16, gpu:0>",
    ]
    for shape in staged:
        assert shape in text, (
            f"no {shape} in the graph, so the attention's heads are whole: "
            + "\n".join(
                line.strip()
                for line in text.splitlines()
                if "rmo.matmul" in line and "gpu:0>" in line
            )
        )
    for shape in (
        f"!mo.tensor<[{HEADS * qk_head_dim}, {HIDDEN}], bf16, gpu:0>",
        f"!mo.tensor<[{HIDDEN}, {HEADS * V_HEAD}], bf16, gpu:0>",
    ):
        assert shape not in text, (
            f"{shape} survived the split: a rank holds every head"
        )


def test_the_down_projection_contracts_one_rank_of_the_inner_width(
    tp8: tuple[DSparkDrafter, str],
) -> None:
    """The staged intermediate is ``INNER / 8`` wide, not ``INNER``.

    Column-sharding the gate and up projections is what buys the bytes, and the
    down projection's contraction is where it shows: per rank the gate and up
    weights are ``[HIDDEN, INNER / 8]`` and the down weight is ``[INNER / 8,
    HIDDEN]``, so the multiply that reads the intermediate reduces ``INNER / 8``
    terms. Whole, the three would be ``[HIDDEN, INNER]`` and ``[INNER, HIDDEN]``
    on every rank.
    """
    _, text = tp8
    inner_per_rank = INNER // DEVICES
    staged = [
        f"!mo.tensor<[{HIDDEN}, {inner_per_rank}], bf16, gpu:0>",
        f"!mo.tensor<[{inner_per_rank}, {HIDDEN}], bf16, gpu:0>",
    ]
    for shape in staged:
        assert shape in text, (
            f"no {shape} in the graph, so the FFN's columns are whole: "
            + "\n".join(
                line.strip()
                for line in text.splitlines()
                if "rmo.matmul" in line and "gpu:0>" in line
            )
        )
    for shape in (
        f"!mo.tensor<[{HIDDEN}, {INNER}], bf16, gpu:0>",
        f"!mo.tensor<[{INNER}, {HIDDEN}], bf16, gpu:0>",
    ):
        assert shape not in text, (
            f"{shape} survived the split: a rank holds the whole inner width"
        )


def test_the_dense_entry_stages_over_the_same_split() -> None:
    """The reference entry carries the split too, not only the paged one.

    The parity gate and a caller holding a whole prompt both enter the drafter
    through :meth:`DSparkDrafter.forward`, and a tensor-parallel deployment
    builds the same tree the serving graph does. Staging it pins that the dense
    entry's mask and value folds are still expressible over a head shard -- one
    rank's heads and inner width wide -- and that its two row-parallel sums are
    completed inside the graph.
    """
    _, text = _stage_dense("tp")
    qk_head_dim = NOPE + ROPE
    for shape in (
        f"!mo.tensor<[{HEADS_PER_RANK * qk_head_dim}, {HIDDEN}], bf16, gpu:0>",
        f"!mo.tensor<[{HIDDEN}, {HEADS_PER_RANK * V_HEAD}], bf16, gpu:0>",
        f"!mo.tensor<[{HIDDEN}, {INNER // DEVICES}], bf16, gpu:0>",
    ):
        assert shape in text, (
            f"no {shape} in the dense graph, so it is built whole: "
            + "\n".join(
                line.strip()
                for line in text.splitlines()
                if "rmo.matmul" in line and "gpu:0>" in line
            )
        )
    assert text.count(ALL_REDUCE) == 2 * _config().num_layers, text.count(
        ALL_REDUCE
    )


def test_a_one_device_mesh_takes_no_split() -> None:
    """A one-way ``tp`` shard is not a split: nothing is partial, so no collective.

    Single-device serving builds a one-device mesh by the same axis name a
    tensor-parallel deployment uses, so the rank count -- not the name -- is
    what keeps its projections whole. Without that term every single-device
    graph would carry a one-way shard and an all-reduce the residual add could
    do without.
    """
    drafter, text = _stage("tp", devices=1)
    layer = drafter.backbone.layers[0]
    assert not layer.attention.tensor_parallel
    assert not layer.feed_forward.tensor_parallel
    for projection in (
        layer.attention.wq,
        layer.attention.wkv_b,
        layer.attention.wo,
        layer.feed_forward.w1,
        layer.feed_forward.w3,
        layer.feed_forward.w2,
    ):
        assert projection.weight.mapping.placements == (Replicated(),)
    assert ALL_REDUCE not in text


def test_a_data_parallel_mesh_replicates_every_projection() -> None:
    """A replica holds its own rows, so no collective could complete a sum."""
    drafter, text = _stage("dp")
    ffn = drafter.backbone.layers[0].feed_forward
    assert not ffn.tensor_parallel
    for projection in (ffn.w1, ffn.w3, ffn.w2):
        assert projection.weight.mapping.placements == (Replicated(),)
    attention = drafter.backbone.layers[0].attention
    assert not attention.tensor_parallel
    for projection in (attention.wq, attention.wkv_b, attention.wo):
        assert projection.weight.mapping.placements == (Replicated(),)
    assert ALL_REDUCE not in text
