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
"""The ``TP_TP`` layout: every expert's FFN width split across the ranks.

An export without an ``ep_size`` puts Mistral Large 4 on this layout, where
each rank holds a ``1 / n_devices`` slice of all 128 experts rather than 16
whole ones. Nothing moves between devices in the MoE: the routed experts and
the shared expert each return a partial sum, which the block's ``post_mlp``
all-reduces, so the layer still ends with one collective.

The overlap tests stage each served arm twice, once as this branch leaves it
and once with the overlap's four changes and the fused op's override undone,
to hold the ``MODULAR_OVERLAP_SHARED_EXPERT`` opt-in to its contract: unset,
the graphs are the ones the unmodified tree staged, and set, the shared expert
is forked onto its own stream and joined again with nothing else moving. The
``MODULAR_MEGA_MOE_TP`` tests stage each rank's routed experts as one fused
node, and the last two hold the flags to composing: with both set, the shared
expert forks around the fused node exactly as it forks around the chain.

Graph staging only: virtual B200s, no compile, no execution, no GPU.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_moe import (
    ExpertParallelMoE,
    QuantizedMoE,
    TensorParallelMoE,
)
from max.nn.common_layers.quant_tensor import FP8BlockTensor, NVFP4Tensor
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.quant_config import QuantConfig
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    MistralLarge4Config,
    build_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers import (
    MistralLarge4ExpertParallelMoE,
    MistralLarge4MLP,
    MistralLarge4TensorParallelMoE,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.mega_moe_tp import (
    MAX_FUSED_TOKENS,
    MEGA_MOE_TP_ENV,
)
from max.pipelines.architectures.mistral_large4_modulev3.mistral_large4 import (
    MistralLarge4,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    ML4_DSPARK_VERIFY_ROWS,
    MistralLarge4NormalizedConfig,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)

TESTDATA = Path("max/tests/integration/architectures/mistral_large4/testdata")
"""Where the checkpoint's ``params.json`` snapshots live, relative to runfiles."""

DEVICES = 8
"""The deployment's rank count, so ``moe_dim / 8`` is the production 768."""

LAYERS = 2
"""Layer 0 dense, layer 1 sparse: both FFN paths, at the checkpoint's widths."""

MAX_SEQ_LEN = 256
"""Only the rope table's row count scales with it; keep the trace cheap."""

TAPS = (1,)
"""The MoE layer, whose post-layer residual the DSpark drafter reads.

The whole-point taps are 1, 18, 36, 54 and 71; a two-layer graph carries the
first only, which is enough to trace the combined prefill/decode graph.
"""

CROSS_DEVICE = re.compile(
    r"mo\.transfer\[[^\n]*, gpu:(\d+)> to <\"gpu\", (\d+)>"
)
"""A copy from one GPU to another, which no captured decode graph may hold."""


@contextmanager
def _virtual_devices(n_devices: int) -> Iterator[None]:
    """SM100 virtual devices: graph staging with no GPU attached.

    Must be entered before the process creates its first device.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _params() -> dict[str, object]:
    """The checkpoint's own geometry, cut down to two layers.

    The scalars stay the checkpoint's -- they are what the expert widths and
    the router's geometry are built from -- while the per-layer patterns shrink
    to a rope layer 0 with the dense FFN and a rope layer 1 with the experts.
    """
    params = copy.deepcopy(
        json.loads((TESTDATA / "ml4_nvfp4_params.json").read_text())
    )
    params["n_layers"] = LAYERS
    params["nope"] = "rope,rope"
    params["dsa"]["sharing"] = "full,full"
    params["kv_sharing"]["source_layers_str"] = "0,1"
    params["moe"]["first_k_dense_replace"] = 1
    return params


def _ep_config(
    config: MistralLarge4Config, dispatch_quant_config: QuantConfig | None
) -> EPConfig:
    """The EP arm the deployment exports: allreduce, one node, one rank each."""
    return EPConfig(
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=dispatch_quant_config,
        combine_dtype=DType.bfloat16,
        hidden_size=config.hidden_size,
        top_k=int(config.num_experts_per_tok),
        n_experts=config.n_routed_experts,
        max_tokens_per_rank=128,
        n_gpus_per_node=len(config.devices),
        n_nodes=1,
        moe_dim=int(config.moe_intermediate_size),
        fused_shared_expert=False,
        use_allreduce=True,
    )


def _build(
    devices: list[DeviceRef],
    *,
    taps: tuple[int, ...] = (),
    graph_mode: str = "decode",
    expert_parallel: bool = False,
) -> tuple[MistralLarge4, list[BufferType | TensorType]]:
    """The module and the decode graph's symbolic inputs.

    ``ep_config`` stays ``None`` unless ``expert_parallel`` is set, which is
    what an export without ``--ep-size`` leaves behind, so every sparse layer
    resolves ``TP_TP``.

    Args:
        devices: The ranks the graph is staged for.
        taps: Target layers whose post-layer residuals the drafter reads. Empty
            builds the non-speculative graph, which emits none.
        graph_mode: ``"decode"`` for a decode-only worker's graph, ``"auto"``
            for the combined prefill/decode graph Mach serves.
        expert_parallel: Whether to build the EP8 arm instead, for the input
            list its dispatch and combine add.
    """
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())

    def cache_tree() -> MultiKVCacheParams:
        return build_kv_params(
            rope_layers=list(normalized.layers.rope),
            kv_source_layers=list(normalized.layers.kv_source),
            indexer_types=[
                "full" if owns else "shared"
                for owns in normalized.layers.owns_indexer
            ],
            num_attention_heads=normalized.n_heads,
            index_head_dim=normalized.index_head_dim,
            index_topk=normalized.index_topk,
            devices=devices,
            kv_cache_config=KVCacheConfig(kv_cache_page_size=128),
            cache_dtype=DType.float8_e4m3fn,
        )

    kv_params = cache_tree()
    config = MistralLarge4Config.from_normalized(
        normalized,
        kv_params=cache_tree(),
        devices=devices,
        max_seq_len=MAX_SEQ_LEN,
        graph_mode=graph_mode,
    )
    config.target_layer_ids = list(taps)
    n_devices = len(devices)
    config.mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)), (n_devices,), ("tp",)
    )
    assert config.ep_config is None
    ep_manager = None
    if expert_parallel:
        # The served EP arm: the same graph plus the dispatch and combine tail.
        config.ep_config = _ep_config(config, config.nvfp4_config)
        ep_manager = EPBatchManager(config.ep_config)
    with F.lazy(), default_device(config.mesh):
        module = MistralLarge4(config, kv_params, ep_manager)
    if taps:
        module.language_model.return_hidden_states = (
            ReturnHiddenStates.SELECTED_LAYERS
        )
    inputs: list[BufferType | TensorType] = list(
        modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params, device_refs=devices
        )
    )
    # `MistralLarge4.forward` peels one host `batch_context_length` off its
    # variadic arguments before the cache tree.
    inputs.insert(3, TensorType(DType.int32, shape=[1], device=DeviceRef.CPU()))
    if ep_manager is not None:
        inputs.extend(ep_manager.input_types())
    return module, inputs


@pytest.fixture(scope="module")
def tp8() -> tuple[MistralLarge4, str, list[BufferType | TensorType]]:
    """The TP8 module, its staged decode graph and that graph's inputs."""
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(False),
    ):
        module, inputs = _build([DeviceRef.GPU(i) for i in range(DEVICES)])
        return module, str(module.trace(*inputs)), inputs


@pytest.fixture(scope="module")
def tp8_dspark() -> str:
    """The combined prefill/decode graph with the DSpark drafter's taps.

    Staged separately because the row-invariant decode pins are only on a
    graph that can also prefill (``sparse_mla.py`` takes them when
    ``graph_mode != "decode"``), so a decode-only graph cannot pin them.
    """
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(False),
    ):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)],
            taps=TAPS,
            graph_mode="auto",
        )
        return str(module.trace(*inputs))


def _vocab_size() -> int:
    """The checkpoint's vocabulary width, which the LM head shards by rank."""
    vocab = _params()["vocab_size"]
    assert isinstance(vocab, int)
    return vocab


def _moe(module: MistralLarge4) -> MistralLarge4TensorParallelMoE:
    """Layer 1's MoE, typed."""
    moe = module.language_model.layers[1].mlp
    assert isinstance(moe, MistralLarge4TensorParallelMoE)
    return moe


def test_tp_tp_builds_the_tensor_parallel_moe(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """A multi-device layer without an ep_size splits the experts' width."""
    moe = _moe(tp8[0])
    assert len(moe.experts) == 128
    assert moe.moe_dim == 768
    assert isinstance(moe.shared_experts, MistralLarge4MLP)


def test_the_shared_expert_splits_by_weight_mapping(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """The shared expert shards its projections like a dense TP MLP."""
    moe = _moe(tp8[0])
    shared = moe.shared_experts
    assert isinstance(shared, MistralLarge4MLP)
    for projection in (shared.w1.weight, shared.w3.weight):
        assert isinstance(projection, FP8BlockTensor)
        assert projection.weight.mapping.placements == (Sharded(0),)
    down = shared.w2.weight
    assert isinstance(down, FP8BlockTensor)
    # ``row_parallel`` names the last axis as ``None, TP``; the mapping keeps
    # the positive index.
    assert down.weight.mapping.placements == (Sharded(1),)
    # Every rank scores all 128 experts on the whole hidden state, so the
    # router stays replicated and needs no fold onto local experts.
    assert moe.gate.gate_score.weight.mapping.placements == (Replicated(),)


def test_the_routed_experts_stay_whole_until_the_stack(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """A routed expert's weights are unsharded on the host, sliced in the graph.

    ``TensorParallelMoE.gate_up_proj``/``down_proj`` stack all 128 experts per
    rank through ``shard_and_stack``, which does the slicing; declaring the
    expert weights sharded as well would split them twice.
    """
    moe = _moe(tp8[0])
    for expert in moe.experts:
        assert isinstance(expert, MistralLarge4MLP)
        for projection in (
            expert.w1.weight,
            expert.w3.weight,
            expert.w2.weight,
        ):
            assert isinstance(projection, NVFP4Tensor)
            assert projection.weight.mapping.placements == (Replicated(),)


def test_the_layer_still_lands_one_allreduce(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """Each layer keeps its two residual collectives, and the MoE adds none.

    One ``mo.distributed.allreduce.sum`` carries every rank's tensor, so the
    count is per collective: attention's and the FFN's in each layer, plus the
    vocabulary-parallel embedding's own sum ahead of them. The MoE's partial
    sum is folded into the FFN one rather than adding a third.
    """
    found = tp8[1].count("distributed.allreduce.sum")
    assert found == 2 * LAYERS + 1, (
        f"{found} all-reduces; expected two per layer plus the embedding's"
    )


def test_the_graph_holds_no_expert_parallel_collectives(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """TP moves no tokens: the dispatch and combine kernels are absent."""
    offenders = sorted(
        set(re.findall(r"ep\.(?:dispatch|combine)[a-z_.]*", tp8[1]))
    )
    assert not offenders, offenders


def test_the_row_invariant_pins_survive_the_layout(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """The decode pins are attention-side and do not read the MoE layout."""
    layers = MistralLarge4NormalizedConfig.from_params_json(_params()).layers
    per_rank = (
        len(set(layers.kv_owner_layers) | set(layers.indexer_owner_layers)) + 1
    )
    found = tp8[1].count('"mo.matmul.row_invariant"')
    assert found == per_rank * DEVICES, (
        f"{found} row-invariant matmuls; expected {per_rank} per rank"
    )


def test_the_split_rows_pin_survives_the_layout(tp8_dspark: str) -> None:
    """The DSpark verify's split-K count is still pinned under ``TP_TP``.

    The sparse MLA op takes ``split_rows_per_request`` only on a graph that can
    also prefill (``sparse_mla.py``), which is the combined graph Mach serves,
    so this arm is staged rather than the decode-only one. Layers that own
    both a latent and an indexer project them in one row-invariant launch, and
    no MoE op is row-dependent.
    """
    sparse = [
        line
        for line in tp8_dspark.splitlines()
        if '"attention.mla.graph.prefill.decode.paged.sparse"' in line
    ]
    assert sparse, "no combined sparse MLA op in the graph"
    pin = re.compile(rf"split_rows_per_request = {ML4_DSPARK_VERIFY_ROWS}\b")
    unpinned = [line.strip() for line in sparse if not pin.search(line)]
    assert not unpinned, "sparse MLA ops without the split pin:\n" + "\n".join(
        unpinned
    )
    layers = MistralLarge4NormalizedConfig.from_params_json(_params()).layers
    per_rank = (
        len(set(layers.kv_owner_layers) | set(layers.indexer_owner_layers)) + 1
    )
    found = tp8_dspark.count('"mo.matmul.row_invariant"')
    assert found == per_rank * DEVICES, (
        f"{found} row-invariant matmuls; expected {per_rank} per rank"
    )


def test_the_layout_stages_only_the_head_allgather(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """No FFN path gathers: the graph's one all-gather is the LM head's.

    A column-parallel MLP keeps its shards local: the shared expert's forward
    concatenates the two projections' shards, runs one matmul and splits the
    result, and every one of those ops is per rank. Its down projection is
    row-parallel, so it returns a partial that joins the routed experts' in the
    layer's single all-reduce. The remaining ``mo.distributed.allgather`` is
    the vocabulary-parallel head's logits, one ``vocab / n_devices`` shard per
    rank.
    """
    gathers = [
        line
        for line in tp8[1].splitlines()
        if "mo.distributed.allgather" in line
    ]
    assert len(gathers) == 1, [line.strip()[:120] for line in gathers]
    vocab_shard = _vocab_size() // DEVICES
    assert f", {vocab_shard}], bf16" in gathers[0], gathers[0].strip()[:300]


def test_the_export_declares_no_expert_parallel_tail(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """The TP export is the EP one minus exactly the 22 dispatch/combine slots.

    Mach's registry entry for this arm declares no extra inputs and appends the
    same 8 signal buffers, so its slot count is the EP entry's 228 against 250
    for the same 72-layer checkpoint the export is made from. The 22 slots are
    ``ep_atomic_counters`` and ``ep_pointer_tensors``, which the export only
    creates when ``ep_size > 1``; the fixture's own counts scale with its two
    layers, so the anchor is the difference rather than either total.
    """
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(False),
    ):
        _, ep_inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)], expert_parallel=True
        )
    tp_inputs = tp8[2]
    assert len(ep_inputs) - len(tp_inputs) == 22, (
        f"EP declares {len(ep_inputs)} inputs against TP's {len(tp_inputs)}; "
        "expected the 22-slot EP tail"
    )
    assert [str(t) for t in ep_inputs[: len(tp_inputs)]] == [
        str(t) for t in tp_inputs
    ], "the EP arm's inputs are not the TP arm's plus the tail"


def test_the_decode_step_issues_no_cross_device_transfer(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """Every rank builds its own copy of what it needs; nothing is shipped."""
    offenders = CROSS_DEVICE.findall(tp8[1])
    assert not offenders, offenders


def test_the_shared_expert_runs_inline(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """No side-stream region is staged for the shared expert."""
    assert "mo.sequence" not in tp8[1]


def test_the_routed_experts_keep_the_unfused_chain(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """The fused SwiGLU epilogue belongs to the EP dispatch, not this layout.

    ``grouped_matmul_silu`` selects the fused NVFP4 kernel only for activations
    the EP dispatch pre-quantized, so a rank here keeps its gate/up bundle in
    checkpoint order and the chain keeps its separate SiLU.
    """
    assert "grouped_matmul_swiglu_nvfp4" not in tp8[1]
    assert "fused_silu" in tp8[1]


def test_the_staged_expert_stacks_are_per_rank_slices(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """Each rank holds all 128 experts at 1/8 width, not 16 of them whole.

    The gate/up stack is ``[experts, 2 * 768, 6144]`` of packed NVFP4, so its
    data leaf keeps ``6144 / 2`` columns; the down stack is
    ``[experts, 6144, 768]``.
    """
    gate_up = re.search(r"!mo\.tensor<\[128, 1536, 3072\], ui8", tp8[1])
    down = re.search(r"!mo\.tensor<\[128, 6144, 384\], ui8", tp8[1])
    assert gate_up, "no per-rank [128, 1536, 3072] gate/up stack in the graph"
    assert down, "no per-rank [128, 6144, 384] down stack in the graph"
    whole_experts = re.search(r"!mo\.tensor<\[16, 12288, 3072\], ui8", tp8[1])
    assert not whole_experts, "the graph holds whole-expert (EP) stacks"


OVERLAP_SHARED_EXPERT = "MODULAR_OVERLAP_SHARED_EXPERT"
"""The overlap's opt-in knob, read where the graph is built
(``layers/moe.py:_OVERLAP_SHARED_EXPERT_ENV``). Unset is this model's default,
so an arm stages the inline graph unless a test says otherwise."""

REGION = re.compile(r"^\s*%(\d+):(\d+) = mo\.sequence\[(\d+)\]", re.MULTILINE)
"""A side-stream region: the first result's number, the result count and the
stream. One region carries every device's body, so the count is the rank count
and the results are read as ``%<base>#<rank>``."""

ROUTER_OP = "moe.single.group.router"
"""The MoE gate. The fork has to land after it and before the routed chain."""

ROUTED_CHAIN_OP = "grouped_matmul"
"""The routed experts' first launch, which the region has to precede for the
two to overlap at all."""

SHARED_EXPERT_STREAM_ID = 1
"""The stream the shared expert's region runs on, as
``_SHARED_EXPERT_STREAM_ID`` sets it."""

ARMS = ("decode", "DSpark", "EP8")
"""The three graphs a flag-off export has to leave exactly as it was: the
decode-only worker's, the combined prefill/decode graph carrying the drafter's
taps, and the expert-parallel arm, whose shared expert is the other caller of
the helpers this branch moves."""


@contextmanager
def _overlap_arm(armed: bool) -> Iterator[None]:
    """Sets (``armed``) or clears the overlap knob across a staging call."""
    saved = os.environ.pop(OVERLAP_SHARED_EXPERT, None)
    if armed:
        os.environ[OVERLAP_SHARED_EXPERT] = "1"
    try:
        yield
    finally:
        os.environ.pop(OVERLAP_SHARED_EXPERT, None)
        if saved is not None:
            os.environ[OVERLAP_SHARED_EXPERT] = saved


@contextmanager
def _mega_moe_tp_flag(on: bool) -> Iterator[None]:
    """``MODULAR_MEGA_MOE_TP`` as an export sets it: read when the MoE is
    built, so it brackets the module's construction."""
    old = os.environ.get(MEGA_MOE_TP_ENV)
    if on:
        os.environ[MEGA_MOE_TP_ENV] = "1"
    else:
        os.environ.pop(MEGA_MOE_TP_ENV, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(MEGA_MOE_TP_ENV, None)
        else:
            os.environ[MEGA_MOE_TP_ENV] = old


@pytest.fixture(scope="module", autouse=True)
def _both_flags_armed() -> Iterator[None]:
    """Sets both flags to ``1`` for the module, before any graph is staged.

    Every staging site here says which flags it wants, through
    :func:`_overlap_arm` and :func:`_mega_moe_tp_flag`. Under bazel both
    variables are otherwise unset, which is also what "off" means, so a site
    that forgot to clear one would stage the graph it meant to anyway and its
    test could not fail. Armed here, the forgotten flag stages armed instead.
    """
    names = (OVERLAP_SHARED_EXPERT, MEGA_MOE_TP_ENV)
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _pre_change_mlp_forward(self: MistralLarge4MLP, x: Tensor) -> Tensor:
    """``MistralLarge4MLP.forward`` as the tree before this branch had it.

    The gate/up split took the module's width instead of the fused output's own.
    The two agree wherever the weights are a mesh-mapped tensor and disagree on
    the per-device shard the side stream hands this layer.
    """
    gate_up = quant_ops.matmul(
        x, quant_ops.concat_weights(self.w1.weight, self.w3.weight, axis=0)
    )
    gate, up = gate_up.split(
        [self.feed_forward_length, self.feed_forward_length], axis=-1
    )
    return self.w2(F.silu(gate) * up)


def _pre_change_tensor_parallel_forward(
    self: TensorParallelMoE, x: Tensor
) -> Tensor:
    """``TensorParallelMoE.forward`` as the tree before this branch had it.

    The routed chain was written out here rather than taken from
    :meth:`_routed_experts_out`, which the branch factors out of it so the
    tensor-parallel override can fork the shared expert ahead of the same ops.
    """
    router_idx, router_weight = self.gate(x)
    gate_up = quant_ops.stack_device_shards(
        self.gate_up_proj, axis=1, mesh=self.mesh
    )
    down = quant_ops.stack_device_shards(self.down_proj, axis=2, mesh=self.mesh)
    routed_expert_out = F.call_on_mesh(
        self._routed_experts,
        self.mesh,
        out_specs=DeviceMapping(self.mesh, (Partial(),)),
    )(x, router_idx, router_weight, gate_up, down)

    if self.shared_experts is not None:
        shared_out = self.shared_experts(x)
        if self.combine_accum_dtype is not None:
            shared_out = shared_out.cast(self.combine_accum_dtype)
        routed_expert_out += shared_out
    if self.combine_accum_dtype is not None:
        routed_expert_out = routed_expert_out.cast(x.dtype)
    return routed_expert_out


_SHARED_EXPERT_HELPERS = ("_shared_experts_inline", "_compute_shared_experts")


@contextmanager
def _without_the_change() -> Iterator[None]:
    """Puts the tree back to the code this branch replaces.

    Four things moved: ``MistralLarge4TensorParallelMoE`` gained a ``forward``
    override, ``MistralLarge4MLP`` took its split off the fused output's width,
    the two shared-expert helpers moved from ``ExpertParallelMoE`` up to
    ``QuantizedMoE``, and ``TensorParallelMoE.forward`` had its routed chain
    factored into ``_routed_experts_out``. Undoing exactly those four is the
    tree before the branch, so a graph staged here is the graph the flag has to
    leave alone, including the one the factored chain stages.

    The fused routed experts (``MODULAR_MEGA_MOE_TP``) came after, and their
    ``_routed_experts`` override is undone here too, so the comparison is also
    the tree before that flag. Their other change, the gate/up bundle's
    permute, sits inside a property whose unflagged path returns the bundle it
    returned before; a swap cannot see that, so the comparison with a graph
    staged on the tree before the fused op was made once, out of tree.

    What this cannot check is the helpers' *bodies*: it re-homes the branch's
    own function objects, so an edit inside one of them moves both arms
    together. Their bodies were compared against ``892ce9e0e38`` when this
    landed and differ only in a docstring, and both ``_pre_change_*_forward``
    above are that commit's functions verbatim.

    The swaps go through ``setattr``/``delattr``, with the attribute in a
    variable: mypy rejects assigning to a method (``method-assign``), and a
    ``type: ignore`` is not an option here.
    """
    tensor_parallel_forward = MistralLarge4TensorParallelMoE.__dict__["forward"]
    routed_experts = MistralLarge4TensorParallelMoE.__dict__["_routed_experts"]
    mlp_forward = MistralLarge4MLP.forward
    base_forward = TensorParallelMoE.forward
    helpers = {
        name: QuantizedMoE.__dict__[name] for name in _SHARED_EXPERT_HELPERS
    }
    # ``(class, attribute, while undone, afterwards)``; ``None`` means the
    # attribute does not exist in that state. The entries are disjoint -- no two
    # touch the same ``(class, attribute)`` -- so restoring them in any order
    # works, which is why ``apply(before=False)`` need not walk the list back.
    swaps: list[tuple[type, str, object | None, object | None]] = [
        (
            MistralLarge4TensorParallelMoE,
            "forward",
            None,
            tensor_parallel_forward,
        ),
        (
            MistralLarge4TensorParallelMoE,
            "_routed_experts",
            None,
            routed_experts,
        ),
        (MistralLarge4MLP, "forward", _pre_change_mlp_forward, mlp_forward),
        (
            TensorParallelMoE,
            "forward",
            _pre_change_tensor_parallel_forward,
            base_forward,
        ),
    ]
    for name, helper in helpers.items():
        swaps.append((ExpertParallelMoE, name, helper, None))
        swaps.append((QuantizedMoE, name, None, helper))

    def apply(before: bool) -> None:
        for cls, attribute, undone, restored in swaps:
            value = undone if before else restored
            if value is None:
                delattr(cls, attribute)
            else:
                setattr(cls, attribute, value)

    apply(before=True)
    try:
        yield
    finally:
        apply(before=False)


def _stage(arm: str) -> tuple[str, int]:
    """Stages one arm and returns its text and its distinct MoE block count.

    The block count is how many side-stream regions the overlap has to produce:
    one per distinct MoE block subgraph, however many layers share it.
    """
    devices = [DeviceRef.GPU(i) for i in range(DEVICES)]
    if arm == "decode":
        module, inputs = _build(devices)
    elif arm == "DSpark":
        module, inputs = _build(devices, taps=TAPS, graph_mode="auto")
    elif arm == "EP8":
        module, inputs = _build(
            devices, graph_mode="auto", expert_parallel=True
        )
    else:
        raise AssertionError(f"unknown arm {arm}")
    graph = str(module.trace(*inputs))
    blocks = {
        module.language_model._block_name(layer)
        for layer in module.language_model.layers
        if isinstance(
            layer.mlp,
            (MistralLarge4TensorParallelMoE, MistralLarge4ExpertParallelMoE),
        )
    }
    assert blocks, f"no MoE block in the {arm} graph"
    return graph, len(blocks)


SSA = re.compile(r"%[A-Za-z_]*\d+(?:#\d+)?")

RNG_SEED = re.compile(r"#M\.dense_array<\d+> : tensor<ui64>")
"""A random draw's seed. One process hands them out from a counter, so two
builds of the same layer in one process draw different ones and the graphs would
differ in a place the layer has no say over.

No arm here draws one, so the byte-for-byte comparisons below are literally
that; the test asserts it, so a draw added to one of these layers later fails
where the reason is obvious. ``test_quant_moe_parallel._SEED`` normalizes the
same attribute, because its comparison arm does draw.
"""

TRIVIA = re.compile(r"=\s*(?:rmo\.)?mo\.(?:constant|broadcast_to)\b")
"""Ops with nothing to compute: a scalar seed and its broadcast. The two forms
of the shared-expert call materialize a different number of them, and neither
reaches a kernel."""

SYMBOL = re.compile(r'symbol = "([\w.]+)"')
"""A kernel launch's name, for ``mo.custom`` ops and the region's copies."""

COMPOSITE = re.compile(r"\b(?:rmo\.)?mo\.composite\.([\w.]+)\(")
"""A fused launch's name; ``mo.custom`` and the composites are the kernels."""


def _flatten(line: str) -> str | None:
    """One op line with its values flattened, or ``None`` for bookkeeping.

    Inserting a region renumbers every value after it, so the names cannot take
    part in a comparison. ``rmo.`` is how the printer marks an op it placed
    inside a region; the same op on the main stream prints as ``mo.``, and both
    are the op the graph runs. A region's own header, yield and closing brace
    describe the region rather than the computation, as does the graph's closing
    brace, so they take no part either.
    """
    if line.strip() == "}" or REGION.match(line):
        return None
    flat = SSA.sub("%", line.strip()).replace("rmo.", "mo.")
    if TRIVIA.search(flat) or flat.startswith("mo.yield"):
        return None
    return flat


def _ops(graph: str) -> list[str]:
    """The graph's ops, in the order the printer emits them, values flattened."""
    return [flat for line in graph.splitlines() if (flat := _flatten(line))]


def _routed_chain(graph: str) -> list[str]:
    """The routed experts' launches, in the order the graph runs them.

    Nothing else in the graph launches ``grouped_matmul``, so this is the
    routed chain on its own: the one part of the layer whose order the region
    must not disturb.
    """
    return [op for op in _ops(graph) if ROUTED_CHAIN_OP in op]


def _first_difference(candidate: list[str], expected: list[str]) -> str:
    """Describes where two op sequences part ways, for an assertion message."""
    for index, (got, want) in enumerate(zip(candidate, expected, strict=False)):
        if got != want:
            return f"line {index} is {got[:200]!r}, expected {want[:200]!r}"
    return (
        f"the first {min(len(candidate), len(expected))} lines match and the"
        f" lengths differ: {len(candidate)} against {len(expected)}"
    )


def _launches(graph: str) -> Counter[str]:
    """The graph's kernel launches, by name, one entry per device run."""
    launched: Counter[str] = Counter()
    for line in graph.splitlines():
        if symbol := SYMBOL.search(line):
            launched[symbol.group(1)] += 1
        if composite := COMPOSITE.search(line):
            launched[f"composite.{composite.group(1)}"] += 1
    return launched


@pytest.fixture(scope="module")
def staged_arms() -> dict[str, tuple[str, int]]:
    """Each arm as this branch stages it, with both flags unset."""
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(False),
    ):
        return {arm: _stage(arm) for arm in ARMS}


@pytest.fixture(scope="module")
def staged_arms_without_the_change() -> dict[str, tuple[str, int]]:
    """The same three arms, staged with the branch's changes undone."""
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(False),
        _without_the_change(),
    ):
        return {arm: _stage(arm) for arm in ARMS}


@pytest.fixture(scope="module")
def staged_overlap_arms() -> dict[str, tuple[str, int]]:
    """The TP8 arms with the overlap opted into, as an A/B export stages them."""
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(True),
        _mega_moe_tp_flag(False),
    ):
        return {arm: _stage(arm) for arm in ("decode", "DSpark")}


@pytest.mark.parametrize("arm", ARMS)
def test_the_flag_off_graphs_are_the_unmodified_trees_graphs(
    staged_arms: dict[str, tuple[str, int]],
    staged_arms_without_the_change: dict[str, tuple[str, int]],
    arm: str,
) -> None:
    """Unset, the flag reaches nothing: every arm stages the graph it staged.

    The comparison arm has the overlap's four changes and the fused op's
    override undone one by one, so a difference is one of them reaching a
    graph the flags were meant to leave alone -- the tensor-parallel override,
    the base class's factored routed chain, the gate/up split, the helpers
    that now sit on the shared base class, or the fused op's
    ``_routed_experts``. The expert-parallel arm is the one that reaches a moved helper: its
    own override sends the shared expert inline, so it stages no region either.

    Both graphs are built in one process, so a random draw at trace time would
    differ between them for a reason that has nothing to do with the flag; the
    seed assertion below keeps that from reading as a change here.
    """
    graph = staged_arms[arm][0]
    before = staged_arms_without_the_change[arm][0]
    assert RNG_SEED.search(graph) is None, f"the {arm} graph draws a seed"
    assert RNG_SEED.search(before) is None, (
        f"the comparison {arm} graph draws a seed, so the two builds differ in"
        " a value neither the flag nor the branch decides; normalize the seeds"
        " the way test_quant_moe_parallel._SEED does before comparing"
    )
    assert REGION.search(graph) is None, (
        f"the {arm} graph stages a side-stream region with {OVERLAP_SHARED_EXPERT}"
        " unset"
    )
    assert graph == before, (
        f"the {arm} graph changed with {OVERLAP_SHARED_EXPERT} unset"
    )


def test_the_comparison_above_can_fail(
    staged_arms_without_the_change: dict[str, tuple[str, int]],
    staged_overlap_arms: dict[str, tuple[str, int]],
) -> None:
    """Control: the same comparison discriminates once the flag asks for the
    overlap.

    Two graphs that agree for a reason other than this branch -- the knob read
    where nothing reaches it, a fixture that stages one tree twice -- would make
    the test above pass while the flag did nothing at all.
    """
    for arm, (graph, _) in staged_overlap_arms.items():
        assert graph != staged_arms_without_the_change[arm][0], (
            f"the {arm} graph is unchanged with the overlap on, so comparing"
            " it with the flag off is not evidence of anything"
        )


@pytest.mark.parametrize("arm", ("decode", "DSpark"))
def test_the_overlap_forks_and_joins_the_shared_expert(
    staged_arms: dict[str, tuple[str, int]],
    staged_overlap_arms: dict[str, tuple[str, int]],
    arm: str,
) -> None:
    """The flag moves the shared expert into a region, and nothing else.

    One region per MoE block, forked after the router and before the routed
    chain's first launch, so the two run concurrent; one result per rank, each
    read exactly once, by the add that sums it into that rank's routed output.
    The compiler synchronizes a region at its boundary, so a result cannot reach
    a consumer unjoined; what the staging can show is that every result does
    reach the add exactly once.

    Everything else is compared between the two arms: the launch census, which
    is every kernel the shared expert and the routed chain run and how many
    times; the routed chain's own launches, in order, since the region must not
    disturb the sequence it overlaps; and every non-trivial op in the graph.

    The ops are compared as a multiset, not as a sequence, because the two forms
    of the shared-expert call emit them in different orders: one module call on
    a distributed tensor emits op by op across the ranks, where the region runs
    each rank's chain whole. Both emit the same ops, so a difference is a change
    and not an artifact -- but only the routed chain is compared by position,
    which is also the one part of the layer both arms reach through the base
    class's own ``_routed_experts_out``. The scalar constants and broadcasts the
    two forms materialize differently are left out of the comparison; neither
    reaches a kernel.
    """
    graph, blocks = staged_overlap_arms[arm]
    inline = staged_arms[arm][0]
    assert REGION.search(inline) is None, (
        f"the {arm} graph stages a side-stream region with the flag unset"
    )

    regions = REGION.findall(graph)
    assert len(regions) == blocks, (
        f"the {arm} graph stages {len(regions)} side-stream regions for"
        f" {blocks} MoE blocks"
    )
    stream_id = SHARED_EXPERT_STREAM_ID
    lines = graph.splitlines()
    for base, count, stream in regions:
        assert int(stream) == stream_id, (
            f"the {arm} region runs on stream {stream}, not the shared"
            f" expert's {stream_id}"
        )
        assert int(count) == DEVICES, (
            f"the {arm} region carries {count} results for {DEVICES} ranks"
        )
        index = next(
            i for i, line in enumerate(lines) if f"%{base}:{count}" in line
        )
        router = max(
            i for i, line in enumerate(lines[:index]) if ROUTER_OP in line
        )
        chain = min(
            i
            for i, line in enumerate(lines[index:], start=index)
            if ROUTED_CHAIN_OP in line
        )
        assert router < index < chain, (
            f"the {arm} region at line {index} is not forked between the"
            f" router ({router}) and the routed chain ({chain})"
        )
        for rank in range(int(count)):
            used = [
                line for line in lines if re.search(rf"%{base}#{rank}\b", line)
            ]
            assert len(used) == 1, (
                f"the {arm} region's rank-{rank} result is read"
                f" {len(used)} times, not once: {used[:2]}"
            )
            assert ".add(" in used[0], (
                f"the {arm} region's rank-{rank} result does not reach the"
                f" add: {used[0].strip()[:200]}"
            )

    assert _launches(graph) == _launches(inline), (
        f"the overlap changes the {arm} graph's kernel census:"
        f" {_launches(graph) - _launches(inline)} added,"
        f" {_launches(inline) - _launches(graph)} dropped"
    )
    assert _routed_chain(graph) == _routed_chain(inline), (
        f"the overlap reorders the {arm} graph's routed chain:"
        f" {_first_difference(_routed_chain(graph), _routed_chain(inline))}"
    )
    added = Counter(_ops(graph)) - Counter(_ops(inline))
    removed = Counter(_ops(inline)) - Counter(_ops(graph))
    assert not added, (
        f"the overlap gives the {arm} graph ops it did not have:"
        f" {sum(added.values())} of them, {sorted(added)[:2]}"
    )
    assert not removed, (
        f"the overlap drops {sum(removed.values())} of the {arm} graph's ops:"
        f" {sorted(removed)[:2]}"
    )


def _stage_fused(
    graph_mode: str, *, fused: bool, expert_parallel: bool = False
) -> str:
    """One staged graph at the TP8 (or EP8) layout, the flag on or off and
    the shared-expert overlap off."""
    with (
        _mega_moe_tp_flag(fused),
        _overlap_arm(False),
        _virtual_devices(DEVICES),
    ):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)],
            graph_mode=graph_mode,
            expert_parallel=expert_parallel,
        )
        return str(module.trace(*inputs))


@pytest.fixture(scope="module")
def tp8_fused() -> dict[str, str]:
    """The flagged TP8 graph in each mode the model builds."""
    return {
        mode: _stage_fused(mode, fused=True)
        for mode in ("decode", "auto", "prefill")
    }


FUSED_OP = re.compile(r'"mega_moe\.tp"')
FUSED_TOKENS = re.compile(r"max_fused_tokens = (\d+)")


def test_the_flag_off_stages_no_fused_op(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """Without ``MODULAR_MEGA_MOE_TP`` the routed experts keep today's chain."""
    assert not FUSED_OP.search(tp8[1])
    assert "moe.finalize" in tp8[1]
    assert "moe.create.indices" in tp8[1]


@pytest.mark.parametrize("mode", ["decode", "auto"])
def test_the_flag_selects_the_fused_op(
    tp8_fused: dict[str, str], mode: str
) -> None:
    """Each rank's sparse layer is one ``mega_moe.tp`` node that may take the
    fused launch, and the routed chain's own ops are gone from the graph,
    the routing among them: the op lays it out from the top-k ids."""
    graph = tp8_fused[mode]
    sparse = DEVICES * (LAYERS - 1)
    assert len(FUSED_OP.findall(graph)) == sparse
    assert FUSED_TOKENS.findall(graph) == [str(MAX_FUSED_TOKENS)] * sparse
    assert "moe.finalize" not in graph
    assert "fused_silu" not in graph
    assert "moe.create.indices" not in graph


@pytest.mark.parametrize("mode", ["decode", "auto"])
def test_the_fused_op_keeps_the_decode_step_capture_safe(
    tp8_fused: dict[str, str], mode: str
) -> None:
    """The op's scalars and counter buffer are built on each rank, so the
    flagged graph holds neither capture breaker: no copy between GPUs and no
    readback to the host. (The counters are minted once at load.)"""
    graph = tp8_fused[mode]
    assert not CROSS_DEVICE.findall(graph)
    to_host = [
        line
        for line in graph.splitlines()
        if "mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip()[:6] == '<"cpu"'
    ]
    assert not to_host, to_host


def test_a_prefill_graph_never_takes_the_fused_launch(
    tp8_fused: dict[str, str],
) -> None:
    """A prefill-only worker's op runs the chain inside it at every step."""
    assert set(FUSED_TOKENS.findall(tp8_fused["prefill"])) == {"0"}


def test_the_flag_permutes_the_gate_up_bundle(
    tp8_fused: dict[str, str],
) -> None:
    """The op reads W13 gate/up row-interleaved: each rank's bundle is split
    into its gate and up halves and the halves interleaved."""
    assert re.search(
        r"!mo\.tensor<\[128, 2, 768, 3072\], ui8", tp8_fused["auto"]
    )


def test_the_flag_keeps_the_exports_inputs(
    tp8: tuple[MistralLarge4, str, list[BufferType | TensorType]],
) -> None:
    """The op's counters are minted inside the graph, so the flagged export
    declares exactly today's inputs and Mach's ``-tp`` registry entry (228
    slots) serves it unchanged."""
    with (
        _mega_moe_tp_flag(True),
        _overlap_arm(False),
        _virtual_devices(DEVICES),
    ):
        _, fused_inputs = _build([DeviceRef.GPU(i) for i in range(DEVICES)])
    assert [str(t) for t in fused_inputs] == [str(t) for t in tp8[2]]


def test_the_flag_leaves_the_expert_parallel_graph_alone() -> None:
    """EP8 stages byte-identically with the flag on: only the TP layout reads
    it, so the default EP export (and its MEF) cannot move."""
    assert _stage_fused(
        "auto", fused=True, expert_parallel=True
    ) == _stage_fused("auto", fused=False, expert_parallel=True)


@pytest.fixture(scope="module")
def staged_fused_arms() -> dict[str, tuple[str, int]]:
    """The TP8 arms with only the fused routed experts opted into."""
    with (
        _virtual_devices(DEVICES),
        _overlap_arm(False),
        _mega_moe_tp_flag(True),
    ):
        return {arm: _stage(arm) for arm in ("decode", "DSpark")}


@pytest.fixture(scope="module")
def staged_both_arms() -> dict[str, tuple[str, int]]:
    """The TP8 arms with both the fused routed experts and the overlap."""
    with _virtual_devices(DEVICES), _overlap_arm(True), _mega_moe_tp_flag(True):
        return {arm: _stage(arm) for arm in ("decode", "DSpark")}


@pytest.mark.parametrize("arm", ("decode", "DSpark"))
def test_both_flags_fork_the_shared_expert_around_the_fused_op(
    staged_fused_arms: dict[str, tuple[str, int]],
    staged_both_arms: dict[str, tuple[str, int]],
    arm: str,
) -> None:
    """With both flags, the routed segment is the fused op and the shared
    expert is on the side stream, joined once.

    One region per MoE block on the shared expert's stream, forked after the
    router and before the block's ``mega_moe.tp`` node, with each rank's
    result read exactly once, by the add. The fused nodes stay on the main
    stream, the unfused routed chain stays out of the graph, and against the
    fused op alone the overlap adds no launch and no op: the kernel census,
    the fused nodes in order and the graph's op multiset all match.
    """
    graph, blocks = staged_both_arms[arm]
    fused_alone = staged_fused_arms[arm][0]
    assert REGION.search(fused_alone) is None, (
        f"the fused {arm} graph stages a side-stream region with the overlap"
        " off"
    )

    assert any("rmo." in line for line in graph.splitlines()), (
        f"the {arm} graph prints no op as rmo., so the printer no longer marks"
        " an op inside a region and the side-stream check below is blind"
    )
    nodes = [line for line in graph.splitlines() if FUSED_OP.search(line)]
    assert len(nodes) == len(FUSED_OP.findall(fused_alone)) > 0, (
        f"the {arm} graph stages {len(nodes)} mega_moe.tp nodes with both"
        f" flags, {len(FUSED_OP.findall(fused_alone))} with the fused op alone"
    )
    in_region = [line for line in nodes if "rmo." in line]
    assert not in_region, (
        f"the {arm} graph runs mega_moe.tp on the side stream:"
        f" {in_region[0].strip()[:200]}"
    )
    assert ROUTED_CHAIN_OP not in graph, (
        f"the {arm} graph stages the unfused routed chain with both flags"
    )

    regions = REGION.findall(graph)
    assert len(regions) == blocks, (
        f"the {arm} graph stages {len(regions)} side-stream regions for"
        f" {blocks} MoE blocks"
    )
    lines = graph.splitlines()
    for base, count, stream in regions:
        assert int(stream) == SHARED_EXPERT_STREAM_ID
        assert int(count) == DEVICES
        index = next(
            i for i, line in enumerate(lines) if f"%{base}:{count}" in line
        )
        router = max(
            (i for i, line in enumerate(lines[:index]) if ROUTER_OP in line),
            default=None,
        )
        assert router is not None, (
            f"the {arm} region at line {index} is forked before any router"
        )
        # The block's own fused node is the first one after the fork with no
        # router in between; past a router it would be the next block's.
        own = None
        for i in range(index + 1, len(lines)):
            if ROUTER_OP in lines[i]:
                break
            if FUSED_OP.search(lines[i]):
                own = i
                break
        assert own is not None, (
            f"the {arm} region at line {index} (after the router at line"
            f" {router}) has no mega_moe.tp node of its own block after it:"
            " the fork is not ahead of the block's fused op"
        )
        for rank in range(int(count)):
            used = [
                line for line in lines if re.search(rf"%{base}#{rank}\b", line)
            ]
            assert len(used) == 1 and ".add(" in used[0], (
                f"the {arm} region's rank-{rank} result is read"
                f" {len(used)} times, not once by the add: {used[:2]}"
            )

    assert _launches(graph) == _launches(fused_alone), (
        f"the overlap changes the fused {arm} graph's kernel census:"
        f" {_launches(graph) - _launches(fused_alone)} added,"
        f" {_launches(fused_alone) - _launches(graph)} dropped"
    )
    fused_nodes = [op for op in _ops(graph) if FUSED_OP.search(op)]
    alone_nodes = [op for op in _ops(fused_alone) if FUSED_OP.search(op)]
    assert fused_nodes == alone_nodes, (
        f"the overlap changes the fused {arm} graph's mega_moe.tp nodes:"
        f" {_first_difference(fused_nodes, alone_nodes)}"
    )
    added = Counter(_ops(graph)) - Counter(_ops(fused_alone))
    removed = Counter(_ops(fused_alone)) - Counter(_ops(graph))
    assert not added and not removed, (
        f"the overlap changes the fused {arm} graph's ops:"
        f" {sorted(added)[:2]} added, {sorted(removed)[:2]} dropped"
    )


def test_each_flag_alone_stages_only_its_own_change(
    staged_fused_arms: dict[str, tuple[str, int]],
    staged_overlap_arms: dict[str, tuple[str, int]],
) -> None:
    """The overlap alone stages no fused node, and the fused op alone no
    region, so neither flag's tests above see the other's change."""
    for arm, (graph, _) in staged_overlap_arms.items():
        assert not FUSED_OP.search(graph), (
            f"the overlap-only {arm} graph stages mega_moe.tp"
        )
    for arm, (graph, _) in staged_fused_arms.items():
        assert REGION.search(graph) is None, (
            f"the fused-only {arm} graph stages a side-stream region"
        )
