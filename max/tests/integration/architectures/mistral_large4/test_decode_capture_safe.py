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
"""The decode step must not move a tensor from one device to another.

Device-graph capture records one graph per device stream. A cross-device
``mo.transfer`` lowers to ``mgp.buffer.device_to_device`` wrapped in
``mgp.device_wait`` on both sides
(``GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp``), and that wait is a
host-enqueued ``cuStreamWaitEvent`` on another device's stream -- which CUDA
refuses to merge into a capture sequence
(``CUDA_ERROR_STREAM_CAPTURE_MERGE``). One such copy anywhere in the decode
graph costs the whole model its capture, and Mistral Large 4's decode graph
carried 539 of them per step: an all-zero router bias built on device 0 and
replicated per MoE layer, the LeRoPE table built on device 0 and replicated
every step, and the per-row metadata two framework helpers left on one device.

Everything that has to be on every rank is built *at* the replicated mapping
instead, so each device computes its own copy and no copy op exists. Real
collectives -- the embedding's broadcast, the EP dispatch/combine, the
all-reduces -- synchronize through signal buffers inside their kernels and are
capture-legal, so they are allowed to stay.

Graph staging only: virtual B200s, no compile, no execution, no GPU.
"""

from __future__ import annotations

import copy
import json
import logging
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
from max.nn import Module
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.quant_config import QuantConfig
from max.nn.legacy.transformer import ReturnHiddenStates
from max.nn.quant_config import DISABLE_FUSED_SWIGLU_NVFP4_ENV
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    MistralLarge4Config,
    build_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers import (
    MistralLarge4ExpertParallelMoE,
    _dispatch,
    transformer_block,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.sparse_mla import (
    sparse_selection_width,
)
from max.pipelines.architectures.mistral_large4_modulev3.mistral_large4 import (
    MistralLarge4,
    _log_row_invariance_off,
    concat_taps,
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
from max.tensor.sharding import DeviceMesh

TESTDATA = Path("max/tests/integration/architectures/mistral_large4/testdata")
"""Where the checkpoint's ``params.json`` snapshots live, relative to runfiles."""

DEVICES = 8
"""The deployment's rank count: TP8 attention beside EP8 routed experts."""

LAYERS = 4
"""Layer 0 dense, layers 1-3 sparse: both FFN paths, and every block shape.

The schedule mirrors the checkpoint's variety rather than the two-layer
fixture's uniform one -- a rope owner, a NoPE owner, and a rope layer that
reads another layer's latent and carries its top-k forward -- because those are
what the export emits as distinct block subgraphs, and a transfer inside one of
them appears only when that block is traced.
"""

MAX_SEQ_LEN = 256
"""Only the rope table's row count scales with it; keep the trace cheap."""

TAPS = (1, 3)
"""Two of the four staged layers, for the drafter's tap arm.

The full schedule's five taps are 1, 18, 36, 54 and 71; a four-layer graph
carries the first two only, which is enough to trace the capture and the
concatenation's column order.
"""

CROSS_DEVICE = re.compile(
    r"mo\.transfer\[[^\n]*, gpu:(\d+)> to <\"gpu\", (\d+)>"
)
"""A transfer whose source and destination are both GPUs.

Both devices are captured with their own names, so a copy between two of them
is exactly the ``mgp.device_wait``-wrapped pair this file exists to forbid. A
host-to-device transfer -- the weight placement at load -- does not match.
"""

MO_SEQUENCE = re.compile(r"\bmo\.sequence\[\d+\]")
"""A side-stream region.

``mo.sequence`` carries one result per body output, so the bracket count is the
result count rather than the stream; the module-level print drops the
``streamId`` attribute the graph-level one shows, and there is one stream for
every such region anyway.
"""

KEY_ROPE = re.compile(r"mo\.composite\.rope\.ragged\(.*\[[^]]*, 1, 64\],")
"""The indexer key's unfused rope: the unsqueezed ``[rows, 1, 64]`` half.

Scoped to that shape rather than to the op, so another layer's rope -- the MLA's
is inside its own fused op today -- cannot turn this file's key-tail guard into
a complaint about the query or a future block.
"""

KEY_COL_OFFSET = re.compile(r"key_col_offset = (\d+) : index")
"""The column the fused key op was told its key starts at.

The op reads its 128 columns out of the merged projection at that offset, so a
zero is a packed operand staged in front of it -- a copy launch per indexer
layer.
"""

EXPECTED_KEY_COL_OFFSETS = Counter({576: 2, 512: 1})
"""The merged projection's latent width on each indexer-owner layer, per rank.

The fixture's three owners are two rope layers, whose latent carries the 64-wide
rope slice after ``kv_lora_rank``, and one no-rope layer, whose latent is
``kv_lora_rank`` alone; the key's own 128 columns sit after each. Counts are per
rank, so the graph stages ``* DEVICES`` of each. Pinned literally rather than
recomputed from the config: a checkpoint whose geometry moved would otherwise
move both sides of this check together and stay green.
"""

OVERLAP_SHARED_EXPERT = "MODULAR_OVERLAP_SHARED_EXPERT"
"""The A/B escape hatch, read where the graph is built (``layers/moe.py``).

Mistral Large 4 runs its shared expert inline by default and only stages the
side stream when this is ``1``, so an exported ``=1`` in the shell decides the
arm for every graph staged here too.
"""


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


FUSED_SWIGLU_OP = "mo.composite.grouped_matmul_swiglu_nvfp4"
"""The routed experts' gate/up matmul with SwiGLU and NVFP4 requantize fused.

On SM100 the graph compiler pairs it with the down matmul it feeds and fuses
the two into one MegaFFN launch, so staging it is what puts the model on the
one-launch path.
"""

UNFUSED_SILU_OP = "ep.fused_silu.nvfp4"
"""The standalone SiLU and requantize launch of the unfused chain."""

GATE_UP_INTERLEAVE = re.compile(
    r"mo\.transpose\([^)]*\) : \(!mo\.tensor<\[(\d+), 2, (\d+), (\d+)\],"
    r" (ui8|f8e4m3fn), gpu:\d+>, [^)]*\) -> !mo\.tensor<\[\1, \2, 2, \3\]"
)
"""The swap that interleaves a stacked ``[gate; up]`` bundle row by row.

``sigma_permute_gate_up_nvfp4`` views ``[E, 2 * moe_dim, K]`` as
``[E, 2, moe_dim, K]`` and swaps the middle axes, so row ``2i`` is gate row
``i`` and row ``2i + 1`` is up row ``i``: the layout the fused SwiGLU kernel
reads. The packed weights stage it as ``ui8`` and their block scales as
``f8e4m3fn``; group 2 is ``moe_dim``.
"""


@pytest.fixture(scope="module", autouse=True)
def _no_arm_override() -> Iterator[None]:
    """Stages the default arm whatever the shell exported.

    The overrides are read where the graph is built, so leaving one set would
    put its arm into the fixtures the default-arm tests assert on and fail them
    for a reason that has nothing to do with the code. The opt-in fixtures set
    them for their own graphs and restore them.
    """
    saved = {
        name: os.environ.pop(name, None)
        for name in (OVERLAP_SHARED_EXPERT, DISABLE_FUSED_SWIGLU_NVFP4_ENV)
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _params() -> dict[str, object]:
    """The checkpoint's own geometry, cut down to two layers.

    The scalars stay the checkpoint's, because they are what the module's
    shapes and the router's geometry are built from, but the per-layer patterns
    shrink to the two layers this file traces: a rope layer 0 with the dense
    FFN, and a rope layer 1 with the routed experts.
    """
    params = copy.deepcopy(
        json.loads((TESTDATA / "ml4_nvfp4_params.json").read_text())
    )
    params["n_layers"] = LAYERS
    params["nope"] = "rope,rope,nope,rope"
    params["dsa"]["sharing"] = "full,full,full,shared"
    params["kv_sharing"]["source_layers_str"] = "0,1,2,1"
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
    cache_dtype: DType = DType.float8_e4m3fn,
    config_cache_dtype: DType | None = None,
    speculative: bool = False,
) -> tuple[MistralLarge4, list[BufferType | TensorType]]:
    """The module, its config and the decode graph's symbolic inputs.

    Args:
        devices: The ranks the graph is staged for.
        taps: Target layers whose post-layer residuals the drafter reads. Empty
            builds the non-speculative graph, which emits none.
        graph_mode: ``"decode"`` for a decode-only worker's graph, ``"auto"``
            for the combined prefill/decode graph Mach serves.
        cache_dtype: The MLA latent cache's dtype.
        config_cache_dtype: The dtype the config's own cache tree declares,
            when it differs from the cache the graph takes, as
            ``MistralLarge4Config.initialize``'s FP8 tree does under a
            pipeline with a BF16 cache.
        speculative: Whether the cache is sized for a DSpark verify.
    """
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())

    def cache_tree(dtype: DType) -> MultiKVCacheParams:
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
            cache_dtype=dtype,
            speculative_method="dspark" if speculative else None,
            num_draft_tokens=ML4_DSPARK_VERIFY_ROWS - 1 if speculative else 0,
        )

    kv_params = cache_tree(cache_dtype)
    config = MistralLarge4Config.from_normalized(
        normalized,
        kv_params=cache_tree(config_cache_dtype or cache_dtype),
        devices=devices,
        max_seq_len=MAX_SEQ_LEN,
        graph_mode=graph_mode,
    )
    config.target_layer_ids = list(taps)
    n_devices = len(devices)
    mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)), (n_devices,), ("tp",)
    )
    config.mesh = mesh
    # The routed experts do not fit one device, so a multi-device decode graph
    # is always expert parallel; the allreduce backend is the export's arm.
    ep_manager = None
    if n_devices > 1:
        config.ep_config = _ep_config(config, config.nvfp4_config)
        ep_manager = EPBatchManager(config.ep_config)

    with F.lazy(), default_device(mesh):
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
    # variadic arguments before the cache tree, as the ModuleV3 batch
    # processor arranges it.
    inputs.insert(3, TensorType(DType.int32, shape=[1], device=DeviceRef.CPU()))
    if ep_manager is not None:
        inputs.extend(ep_manager.input_types())
    return module, inputs


@pytest.fixture(scope="module")
def staged_decode_graph() -> str:
    """The TP8 decode graph's staged text, traced once for this module."""
    with _virtual_devices(DEVICES):
        module, inputs = _build([DeviceRef.GPU(i) for i in range(DEVICES)])
        return str(module.trace(*inputs))


@pytest.fixture(scope="module")
def staged_tap_graph() -> str:
    """The same decode graph with two taps captured for the drafter.

    The tap arm is a different graph -- its outputs carry the concatenated
    residuals -- so its capture safety is staged separately.
    """
    with _virtual_devices(DEVICES):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)], taps=TAPS
        )
        return str(module.trace(*inputs))


@pytest.fixture(scope="module")
def staged_combined_graphs() -> dict[str, str]:
    """The combined prefill/decode graph Mach serves, base and DSpark arms.

    The DSpark target is the tap arm: the same layers, with the drafter's taps
    captured.
    """
    devices = [DeviceRef.GPU(i) for i in range(DEVICES)]
    graphs = {}
    with _virtual_devices(DEVICES):
        for name, taps in (("base", ()), ("spec", TAPS)):
            module, inputs = _build(devices, taps=taps, graph_mode="auto")
            graphs[name] = str(module.trace(*inputs))
    return graphs


@pytest.fixture(scope="module")
def staged_overlap_graphs() -> dict[str, tuple[str, int]]:
    """The decode-only and both served graphs with the overlap opted into.

    Staged exactly as the default-arm fixtures are, with
    ``MODULAR_OVERLAP_SHARED_EXPERT=1`` set across the traces, since that is
    where the model reads it. Each entry carries the number of distinct MoE
    block subgraphs the staged module has, which is how many regions the escape
    hatch has to produce.
    """
    devices = [DeviceRef.GPU(i) for i in range(DEVICES)]
    saved = os.environ.get(OVERLAP_SHARED_EXPERT)
    os.environ[OVERLAP_SHARED_EXPERT] = "1"
    graphs: dict[str, tuple[str, int]] = {}
    try:
        with _virtual_devices(DEVICES):
            for name, taps, graph_mode in (
                ("decode", (), "decode"),
                ("base", (), "auto"),
                ("DSpark", TAPS, "auto"),
            ):
                module, inputs = _build(
                    devices, taps=taps, graph_mode=graph_mode
                )
                moe_blocks = {
                    module.language_model._block_name(layer)
                    for layer in module.language_model.layers
                    if isinstance(layer.mlp, MistralLarge4ExpertParallelMoE)
                }
                graphs[name] = (str(module.trace(*inputs)), len(moe_blocks))
    finally:
        if saved is None:
            del os.environ[OVERLAP_SHARED_EXPERT]
        else:
            os.environ[OVERLAP_SHARED_EXPERT] = saved
    return graphs


@pytest.mark.parametrize("arm", ["base", "spec"])
def test_decode_launches_take_the_verify_kernels(
    staged_combined_graphs: dict[str, str], arm: str
) -> None:
    """Both arms opt every row-count-dependent decode launch in.

    Each sparse MLA op takes the DSpark verify's split-K count, and each BF16
    projection (an owner's latent, an indexer's key, the LM head shard) runs as
    ``mo.matmul.row_invariant`` on every rank, so a one-row decode step
    computes its row as the four-row verify does. A layer that owns both its
    latent and its indexer projects the two in one launch.
    """
    graph = staged_combined_graphs[arm]
    sparse = [
        line
        for line in graph.splitlines()
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
    found = graph.count('"mo.matmul.row_invariant"')
    assert found == per_rank * DEVICES, (
        f"{found} row-invariant matmuls; expected {per_rank} per rank"
    )


def _graph_cache_leaves(module: MistralLarge4) -> dict[str, MLAKVCacheParams]:
    """The latent leaves of the cache tree the graph's inputs flatten from.

    The pipeline resolves the capture key from the tree the graph's inputs are
    flattened from, so that is the tree these assertions have to read.
    """
    assert isinstance(module.kv_params, MultiKVCacheParams)
    leaves: dict[str, MLAKVCacheParams] = {}
    for group in module.language_model.kv_slots:
        leaf = module.kv_params.children[group]
        assert isinstance(leaf, MLAKVCacheParams)
        leaves[group] = leaf
    return leaves


def _verify_split_pins(module: MistralLarge4) -> dict[str, bool]:
    """The row-invariant pin on each latent leaf of the graph's tree."""
    return {
        group: leaf.sparse_row_invariant_verify
        for group, leaf in _graph_cache_leaves(module).items()
    }


@pytest.mark.parametrize("arm", ["base", "spec"])
def test_the_capture_key_takes_the_split_the_launch_takes(arm: str) -> None:
    """The dispatch key is pinned exactly where the sparse op is pinned.

    One pin, two views: the op sizes the launch from the verify's row count and
    the key has to take the same count, or a captured verify replays a split
    the step does not take and is captured again at every cache length. The
    rest of the key's inputs are the op's too -- the selection's width as the
    index stride, and rows already mapped to storage -- so the two views
    differ in nothing the cache length can move.
    """
    with _virtual_devices(DEVICES):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)],
            graph_mode="auto",
            speculative=arm == "spec",
        )
        graph = str(module.trace(*inputs))
    assert re.search(
        rf"split_rows_per_request = {ML4_DSPARK_VERIFY_ROWS}\b", graph
    )
    assert set(_verify_split_pins(module).values()) == {True}
    index_topk = MistralLarge4NormalizedConfig.from_params_json(
        _params()
    ).index_topk
    for leaf in _graph_cache_leaves(module).values():
        assert leaf.sparse_indices_stride == sparse_selection_width(
            leaf, index_topk
        )
        assert leaf.sparse_physical_indices


def test_a_decode_only_worker_keys_what_it_launches() -> None:
    """A decode-only graph takes no split pin, so its key takes none either.

    ``mla_decode_graph`` has no split pin yet (``TODO(DISTINF-832)`` in
    ``sparse_mla.py``): it launches the dispatcher's own count, so a key that
    pinned the verify's would capture that launch under a count it never takes.
    """
    with _virtual_devices(DEVICES):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(DEVICES)], graph_mode="decode"
        )
        graph = str(module.trace(*inputs))
    assert module.language_model.row_invariant_decode
    assert "split_rows_per_request" not in graph
    assert set(_verify_split_pins(module).values()) == {False}


def _stage_combined(
    n_devices: int,
    *,
    cache_dtype: DType = DType.float8_e4m3fn,
    config_cache_dtype: DType | None = None,
    speculative: bool = False,
) -> str:
    """Stages the combined graph at ``n_devices`` ranks and returns its text."""
    with _virtual_devices(n_devices):
        module, inputs = _build(
            [DeviceRef.GPU(i) for i in range(n_devices)],
            graph_mode="auto",
            cache_dtype=cache_dtype,
            config_cache_dtype=config_cache_dtype,
            speculative=speculative,
        )
        graph = str(module.trace(*inputs))
    assert '"attention.mla.graph.prefill.decode.paged.sparse"' in graph
    return graph


def test_tp4_is_the_largest_head_group_the_pins_cover() -> None:
    """At TP4 each rank holds 16 heads, and 16 x 4 verify rows fill one group."""
    graph = _stage_combined(4)
    assert re.search(
        rf"split_rows_per_request = {ML4_DSPARK_VERIFY_ROWS}\b", graph
    )
    assert '"mo.matmul.row_invariant"' in graph


@pytest.mark.parametrize(
    ("n_devices", "arch", "cache_dtype", "config_cache_dtype"),
    [
        (2, "sm_100a", DType.float8_e4m3fn, None),
        (4, "sm_90a", DType.float8_e4m3fn, None),
        (4, "sm_100a", DType.bfloat16, None),
        (4, "sm_100a", DType.bfloat16, DType.float8_e4m3fn),
    ],
    ids=["tp2_fp8", "tp4_fp8_sm90", "tp4_bf16", "tp4_bf16_graph_fp8_config"],
)
def test_outside_the_regime_the_model_declines_the_pins(
    monkeypatch: pytest.MonkeyPatch,
    n_devices: int,
    arch: str,
    cache_dtype: DType,
    config_cache_dtype: DType | None,
) -> None:
    """A configuration the pins cannot cover keeps the default kernels.

    At TP2 the 32 heads times a 4-row verify overflow one head group, a BF16
    cache has no FP8 verify split, and only SM100 honors the row-invariant
    matmul, so the kernels' compile-time checks would reject the pins. The
    decision reads the cache the graph takes, not the config's own tree. The
    virtual devices stay SM100 (the process creates them once), so the target
    the decision reads is patched instead. Staging a declined graph is
    checked once, at TP1, by the warning test below.
    """
    monkeypatch.setattr(
        transformer_block, "accelerator_architecture_name", lambda: arch
    )
    with _virtual_devices(n_devices):
        module, _ = _build(
            [DeviceRef.GPU(i) for i in range(n_devices)],
            graph_mode="auto",
            cache_dtype=cache_dtype,
            config_cache_dtype=config_cache_dtype,
        )
    assert not module.language_model.row_invariant_decode
    # The key follows the same decision: nothing left a leaf pinned while the
    # launch took the dispatcher's own count.
    assert set(_verify_split_pins(module).values()) == {False}


def test_losing_the_identity_under_speculation_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A DSpark cache outside the regime stages no pins and logs a warning.

    At TP1 the 64 heads times a 4-row verify overflow one head group.
    """
    _log_row_invariance_off.cache_clear()
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        graph = _stage_combined(1, speculative=True)
    assert "split_rows_per_request" not in graph
    assert '"mo.matmul.row_invariant"' not in graph
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "default kernels" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    assert "head group" in warnings[0], warnings[0]


def test_the_decode_only_graph_takes_no_split_pin(
    staged_decode_graph: str,
) -> None:
    """A decode-only worker's graph still leaves the split to the dispatcher.

    Its op takes no split pin yet (``TODO(DISTINF-832)`` in ``sparse_mla.py``),
    so threading one through must change this test rather than pass silently.
    """
    assert "split_rows_per_request" not in staged_decode_graph


def test_the_decode_step_issues_no_cross_device_transfer(
    staged_decode_graph: str,
) -> None:
    """Every rank builds its own copy of what it needs; nothing is shipped."""
    offenders = CROSS_DEVICE.findall(staged_decode_graph)
    assert not offenders, (
        f"{len(offenders)} cross-device transfers in the decode graph: each one"
        " lowers to mgp.buffer.device_to_device between two mgp.device_wait"
        " calls, which CUDA refuses inside a capture"
        " (CUDA_ERROR_STREAM_CAPTURE_MERGE). Build the tensor at the mapping"
        " every rank holds instead, or move it with a broadcast kernel.\n"
        + "\n".join(
            line.strip()
            for line in staged_decode_graph.splitlines()
            if CROSS_DEVICE.search(line)
        )
    )


def test_the_decode_step_still_stages_the_batch_as_a_collective(
    staged_decode_graph: str,
) -> None:
    """The batch still reaches every rank, through a signal-buffer broadcast.

    Removing the transfers must not have quietly removed the staging: a rank
    that never receives the tokens produces nothing to sample from. The
    broadcast is a kernel that synchronizes inside itself, so it is legal under
    capture where a plain transfer is not.
    """
    assert "mo.distributed.broadcast" in staged_decode_graph


def test_the_decode_step_reads_nothing_back_to_the_host(
    staged_decode_graph: str,
) -> None:
    """A device-to-host readback is the other capture breaker.

    It lowers to ``mgp.buffer.device_to_host`` followed by ``mgp.sync``, a
    blocking synchronize that cannot be recorded into a capturing stream. The
    row count a split needs as a host-side bound comes from the host copy of
    the offsets, which is already a graph input.
    """
    to_host = [
        line.strip()
        for line in staged_decode_graph.splitlines()
        if "mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip()[:6] == '<"cpu"'
    ]
    assert not to_host, (
        "device-to-host readback in the decode step:\n" + "\n".join(to_host)
    )


def test_the_shared_expert_runs_inline(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """No side-stream region is staged for the shared expert by default.

    The expert-parallel MoE puts the shared expert on a side stream, to overlap
    the routed experts' communication, unless the model declines it. Mistral
    Large 4 declines it (``layers/moe.py:_compute_shared_experts``): the
    region's output differs from run to run on the same input, and it is the
    only user of ``mo.sequence`` in these graphs, so a graph carrying one is a
    graph whose logits are not reproducible. The DSpark graph is the tap arm
    and stages the same layers, so it is checked too.
    """
    for name, graph in (
        ("decode", staged_decode_graph),
        ("DSpark", staged_tap_graph),
    ):
        regions = MO_SEQUENCE.findall(graph)
        assert not regions, (
            f"the {name} graph stages {len(regions)} side-stream regions, so"
            " the shared expert is back on a side stream and the graph is no"
            " longer deterministic run to run"
        )


_FP32_ACTIVATION_CAST = re.compile(
    r"mo\.cast\(%\w+(?:#\d+)?\) : "
    r"\(!mo\.tensor<\[total_seq_len, 6144\], bf16, gpu:\d+>\)"
    r" -> !mo\.tensor<\[total_seq_len, 6144\], f32, gpu:\d+>"
)
"""A launch that widens the whole 6144-wide activation to float32.

The operand is a plain value name, or one result of a multi-result op
(``%592#0``) where the fused norm+quantize produced it."""


def test_the_router_moves_the_activation_cast_into_its_gemv(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """Every router widens its activation inside the gate GEMV, in fp32.

    The gate projection is float32 on both sides, as the reference is, so the
    activation's widening used to be a launch of its own in front of the fp32
    GEMV. It now rides the GEMV (``moe.router.gate.mixed.gemv``), which is
    bit-identical to the cast plus the fp32 GEMV; the op is capped at the 64
    rows where an fp32 matmul of this shape would cross to the TF32 tile GEMM,
    and above that cap the model asks the op for ``precise_max_m = 256``, so a
    launch up to a c64 verify step (64 requests at 4 rows) stays IEEE fp32
    instead of losing its low mantissa bits, and a wider one (a prefill
    chunk) keeps the chain's TF32 tile GEMM.

    The indexer's head-weight projection folds the same way, through
    ``mo.mixed.gemv.epilogue`` -- the router's op with the projection's store
    epilogue, which the router's op does not take. The query norm's upcast
    rides the fused norm+block-quantize launch (``test_norm_quant_fusions``)
    and the head weights' rides their GEMV, so no whole-activation float32
    cast is staged anywhere.
    """
    owners = len(
        MistralLarge4NormalizedConfig.from_params_json(
            _params()
        ).layers.indexer_owner_layers
    )
    for name, graph in (
        ("decode", staged_decode_graph),
        ("DSpark", staged_tap_graph),
    ):
        routers = graph.count('"moe.single.group.router"')
        assert routers, f"the {name} graph stages no router"
        fused = graph.count('"moe.router.gate.mixed.gemv"')
        assert fused == routers, (
            f"the {name} graph stages {fused} fused gate GEMVs for its"
            f" {routers} routers"
        )
        # The indexer's head weights fold through the sibling op, which keeps
        # the projection's scale chain in the one launch: the router's op
        # writes its result plainly, so a projection carrying those
        # multiplies would trade the cast for a launch of the same size.
        heads = graph.count('"mo.mixed.gemv.epilogue"')
        assert heads == owners * DEVICES, (
            f"the {name} graph stages {heads} fused head-weight GEMVs for its"
            f" {owners} indexer owners on {DEVICES} ranks"
        )
        # The bound is a decision, not a tuning knob: a c64 verify step is 64
        # requests at 4 rows each, and a prefill chunk is far wider. Both
        # folded GEMVs are capped there -- above it each op stages today's
        # chain, which is the row count an fp32 matmul of either shape
        # crosses to the TF32 tile GEMM.
        capped = graph.count("max_m = 64")
        assert capped == routers + owners * DEVICES, (
            f"the {name} graph caps {capped} of its {routers} fused gate"
            f" GEMVs and {owners * DEVICES} fused head-weight GEMVs at the"
            " dispatch's 64-row precision edge"
        )
        # The bound is a decision, not a tuning knob: a c64 verify step is 64
        # requests at 4 rows each, and a prefill chunk is far wider.
        precise = graph.count("precise_max_m = 256")
        assert precise == routers, (
            f"the {name} graph bounds {precise} of its {routers} gate GEMVs at"
            " the 256-row precise edge"
        )
        casts = _FP32_ACTIVATION_CAST.findall(graph)
        assert not casts, (
            f"the {name} graph stages {len(casts)} whole-activation float32"
            " casts; the query norm's upcast is inside the fused norm+quantize"
            " launch and the indexer's head-weight cast is inside its GEMV"
        )


def test_a_target_without_that_gemv_keeps_the_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused GEMVs are SM100-only, so another target keeps today's chain.

    Their launch tables and the fp32 GEMV they must reproduce are the SM100
    target's; a graph staged anywhere else takes the cast and the fp32 matmul,
    for the router gate and for the indexer's head weights alike.
    """
    monkeypatch.setattr(
        _dispatch, "accelerator_architecture_name", lambda: "sm_90a"
    )
    graph = _stage_combined(2)
    assert '"moe.router.gate.mixed.gemv"' not in graph
    assert '"mo.mixed.gemv.epilogue"' not in graph
    assert _FP32_ACTIVATION_CAST.search(graph)
    assert '"moe.single.group.router"' in graph


def test_the_indexer_takes_the_fused_epilogues(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """Each indexer quantizes its query and stores its key in one launch each.

    The query's SSMax, rope and FP8 quantize are one op, and the key's whole
    tail -- the LayerNorm, the rope, the FP8 quantize and both cache stores --
    is another. The separate key-scale store and the FP8 quantize of the old
    chain are gone from both arms, and so are the key's own LayerNorm and rope
    launches; the key's columns come out of the merged projection in place, so
    no copy of them is staged either. Each rank runs every indexer, so each op
    appears once per indexer owner and rank.
    """
    owners = len(
        MistralLarge4NormalizedConfig.from_params_json(
            _params()
        ).layers.indexer_owner_layers
    )
    for name, graph in (
        ("decode", staged_decode_graph),
        ("DSpark", staged_tap_graph),
    ):
        for op in (
            '"indexer.q.ssmax_rope_quantize.fp8"',
            '"indexer.k.quantize_store.fp8.paged.ragged"',
        ):
            found = graph.count(op)
            assert found == owners * DEVICES, (
                f"the {name} graph stages {found} {op}; expected {owners} per"
                " rank"
            )
        # Only the old chain quantized with E8M0 scales; every projection's
        # activation quantize keeps float32 scales.
        leftovers = [
            line.strip()
            for line in graph.splitlines()
            if '"kv_cache.store_k_scales.paged.ragged"' in line
            or (
                '"mo.composite.quantize_dynamic_scaled_fp8"' in line
                and "f8e8m0fnu" in line
            )
        ]
        assert not leftovers, (
            f"the {name} graph still stages the unfused indexer chain:\n"
            + "\n".join(leftovers[:3])
        )
        # The key's LayerNorm and rope are inside the fused op now. An indexer
        # still stages one layer_norm -- its `hidden_size`-wide query input
        # norm -- so a 128-wide one is the key's coming back, and a rope over
        # the key's `[rows, 1, 64]` half means the key's rope left the fused op.
        key_norms = [
            line.strip()
            for line in graph.splitlines()
            if "mo.reduce.layer_norm" in line and "[128]" in line
        ]
        assert not key_norms, (
            f"the {name} graph stages the key's LayerNorm again:\n"
            + "\n".join(key_norms[:3])
        )
        rope = [
            line.strip() for line in graph.splitlines() if KEY_ROPE.search(line)
        ]
        assert not rope, (
            f"the {name} graph stages the indexer key's rope again:\n"
            + "\n".join(rope[:3])
        )
        # The op reads its key columns out of the merged projection in place:
        # the offset it is given is the projection's latent width, and a zero
        # offset means a packed key was staged in front of the op instead --
        # which the compiler turns into a copy launch per indexer layer, the
        # very launch this fusion removes. The geometry is pinned exactly, not
        # merely checked to be non-zero: the offset is now passed by the layer
        # rather than recovered from the operand, so any other value means the
        # layer's latent width and the indexer's key disagree.
        offsets = KEY_COL_OFFSET.findall(graph)
        assert len(offsets) == owners * DEVICES, (
            f"the {name} graph prints {len(offsets)} key-op offsets; expected"
            f" {owners} per rank"
        )
        packed = [offset for offset in offsets if int(offset) == 0]
        assert not packed, (
            f"the {name} graph hands the fused key op a packed operand"
            f" ({len(packed)} of {len(offsets)}), so the projection's slice is"
            " staged as a copy launch in front of it"
        )
        offset_counts = Counter(int(offset) for offset in offsets)
        expected_offsets = Counter(
            {
                offset: count * DEVICES
                for offset, count in EXPECTED_KEY_COL_OFFSETS.items()
            }
        )
        assert offset_counts == expected_offsets, (
            f"the {name} graph's key-op offsets are {dict(offset_counts)};"
            f" expected exactly {dict(expected_offsets)}"
        )


def _merged_projections(
    graph: str, hidden_size: int
) -> list[tuple[str, str, int]]:
    """The staged latent+key weight merges: both operands and the result width.

    A merge is `rmo.concat` of two ``[rows, hidden]`` BF16 weight tensors on
    the output axis, so its result is the projection's full width.
    """
    pattern = re.compile(
        r"= rmo\.concat\((%\w+), (%\w+)\) \[axis = 0 : index\] : \("
        rf"!mo\.tensor<\[\d+, {hidden_size}\], bf16, gpu:\d+>, "
        rf"!mo\.tensor<\[\d+, {hidden_size}\], bf16, gpu:\d+>\) -> "
        rf"!mo\.tensor<\[(\d+), {hidden_size}\], bf16, gpu:\d+>"
    )
    return [(m[0], m[1], int(m[2])) for m in pattern.findall(graph)]


def _projection_widths(
    normalized: MistralLarge4NormalizedConfig,
) -> Counter[int]:
    """The merged projection's width on each owner layer, per rank.

    ``kv_lora_rank`` plus the layer's rope slice, with the index key appended.
    """
    layers = normalized.layers
    return Counter(
        int(normalized.kv_lora_rank)
        + (int(normalized.qk_rope_head_dim) if layers.rope[i] else 0)
        + int(normalized.index_head_dim)
        for i in layers.kv_owner_layers
        if layers.owns_indexer[i]
    )


def test_the_merged_projection_is_one_weight_per_owner(
    staged_combined_graphs: dict[str, str],
) -> None:
    """The latent and key weights merge once per owner and rank, at the weights.

    Both operands are the checkpoint weights' device transfers, so the merged
    weight is one constant tensor: no layer copies an activation or a slice of
    the weight into place per step, which would silently restore the launch
    the merge exists to remove.
    """
    graph = staged_combined_graphs["base"]
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    hidden_size = int(normalized.hidden_size)
    projections = _merged_projections(graph, hidden_size)
    assert projections, "no latent+key weight merge in the graph"
    from_activations = [
        (left, right)
        for left, right, _ in projections
        if not (left.startswith("%result_") and right.startswith("%result_"))
    ]
    assert not from_activations, (
        "the projection merges something other than the two weight transfers,"
        f" so a copy runs every step: {from_activations[:3]}"
    )
    expected = _projection_widths(normalized)
    found = Counter(width for _, _, width in projections)
    assert found == Counter(
        {width: count * DEVICES for width, count in expected.items()}
    ), f"merged projections per rank: {found}, expected {expected}"


def test_the_merged_projection_waits_for_the_row_invariant_pin() -> None:
    """A declined pin keeps the latent and the key projection separate.

    Merged, the two run as one matmul whose dispatch is chosen for the wider
    output, which is not what the two separate matmuls compute. The merge is
    therefore gated on the row-invariant matmul, whose reduction does not
    depend on its width.
    """
    _log_row_invariance_off.cache_clear()
    graph = _stage_combined(1, speculative=True)
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    widths = set(_projection_widths(normalized))
    merged = [
        projection
        for projection in _merged_projections(
            graph, int(normalized.hidden_size)
        )
        if projection[2] in widths
    ]
    assert not merged, (
        "the latent and key projections are merged outside the row-invariant"
        f" regime: {merged[:3]}"
    )


def test_the_shared_expert_stages_a_side_stream_when_opted_in(
    staged_overlap_graphs: dict[str, tuple[str, int]],
) -> None:
    """``MODULAR_OVERLAP_SHARED_EXPERT=1`` puts the overlap back, for an A/B.

    One region per distinct MoE block subgraph, in the graphs Mach serves as
    well as in the decode-only one: the escape hatch has to reach every layer,
    or a measurement against the inline default compares two graphs that differ
    in more than the overlap.
    """
    for name, (graph, moe_blocks) in staged_overlap_graphs.items():
        assert moe_blocks, f"no expert-parallel MoE block in the {name} graph"
        regions = MO_SEQUENCE.findall(graph)
        assert len(regions) == moe_blocks, (
            f"the {name} graph stages {len(regions)} side-stream regions for"
            f" {moe_blocks} MoE block subgraphs"
        )


@pytest.fixture(scope="module")
def staged_chain_graphs() -> dict[str, str]:
    """The decode and tap graphs with the fused SwiGLU opted out.

    ``MAX_DISABLE_FUSED_SWIGLU_NVFP4=1`` is read where the routed experts'
    quant config is built, so it is set across the traces and restored.
    """
    devices = [DeviceRef.GPU(i) for i in range(DEVICES)]
    saved = os.environ.get(DISABLE_FUSED_SWIGLU_NVFP4_ENV)
    os.environ[DISABLE_FUSED_SWIGLU_NVFP4_ENV] = "1"
    graphs: dict[str, str] = {}
    try:
        with _virtual_devices(DEVICES):
            for name, taps in (("decode", ()), ("DSpark", TAPS)):
                module, inputs = _build(devices, taps=taps)
                graphs[name] = str(module.trace(*inputs))
    finally:
        if saved is None:
            del os.environ[DISABLE_FUSED_SWIGLU_NVFP4_ENV]
        else:
            os.environ[DISABLE_FUSED_SWIGLU_NVFP4_ENV] = saved
    return graphs


def test_the_routed_experts_take_the_fused_swiglu(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """Every rank's routed experts stage the fused gate/up and no SiLU launch.

    The fused gate/up op is what the graph compiler fuses with the down matmul
    into one MegaFFN launch, so a graph without it runs the three-launch chain
    instead. Each MoE block stages one per rank.
    """
    for name, graph in (
        ("decode", staged_decode_graph),
        ("DSpark", staged_tap_graph),
    ):
        assert graph.count(FUSED_SWIGLU_OP) >= DEVICES, name
        assert UNFUSED_SILU_OP not in graph, name


def test_opting_out_of_the_fused_swiglu_restores_the_chain(
    staged_chain_graphs: dict[str, str],
) -> None:
    """``MAX_DISABLE_FUSED_SWIGLU_NVFP4=1`` stages the unfused chain, for an
    A/B: the separate SiLU launch is back and no fused gate/up is staged."""
    for name, graph in staged_chain_graphs.items():
        assert FUSED_SWIGLU_OP not in graph, name
        assert graph.count(UNFUSED_SILU_OP) >= DEVICES, name


def test_the_fused_swiglu_reads_interleaved_gate_up_rows(
    staged_decode_graph: str,
    staged_tap_graph: str,
    staged_chain_graphs: dict[str, str],
) -> None:
    """The fused gate/up op gets its weights and scales row-interleaved.

    The op and the permutation are both chosen off ``can_use_fused_swiglu``,
    so staging the op alone would pass with the weights left in their stacked
    ``[gate; up]`` order, which the fused kernel would read as a different
    matmul. Each fused op must come with one interleaving swap of the packed
    weights and one of their scales over the expert width, and the chain must
    stage neither.
    """
    moe = _params()["moe"]
    assert isinstance(moe, dict)
    moe_dim = int(moe["expert_hidden_dim"])
    for name, graph in (
        ("decode", staged_decode_graph),
        ("DSpark", staged_tap_graph),
    ):
        swaps = Counter(
            (int(width), dtype)
            for _, width, _, dtype in GATE_UP_INTERLEAVE.findall(graph)
        )
        fused = graph.count(FUSED_SWIGLU_OP + "(")
        assert fused >= DEVICES, name
        assert swaps == {
            (moe_dim, "ui8"): fused,
            (moe_dim, "f8e4m3fn"): fused,
        }, (
            f"the {name} graph stages {fused} fused gate/up ops but these"
            f" interleaving swaps: {dict(swaps)}"
        )
    for name, graph in staged_chain_graphs.items():
        assert not GATE_UP_INTERLEAVE.search(graph), name


def test_the_tap_arm_is_capture_safe_too(staged_tap_graph: str) -> None:
    """Capturing the drafter's taps adds no transfer and no readback.

    The taps are read off the residual stream inside the graph, so the arm that
    emits them must stage exactly like the arm that does not: a copy here would
    cost the speculative graph its capture just as surely.
    """
    offenders = CROSS_DEVICE.findall(staged_tap_graph)
    assert not offenders, (
        f"{len(offenders)} cross-device transfers in the tap arm of the decode"
        " graph"
    )
    to_host = [
        line.strip()
        for line in staged_tap_graph.splitlines()
        if "mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip()[:6] == '<"cpu"'
    ]
    assert not to_host, "device-to-host readback in the tap arm"


def test_the_tap_arm_emits_the_concatenated_taps(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """The tap arm really carries the taps out; it is not an empty arm.

    The checks above are absences, which a graph that emitted nothing would
    also satisfy. The taps are the block driver's whole input, so this reads
    the staged signature: the tap arm returns the non-tap arm's outputs first
    and then one tap tensor per rank, in rank order, each as wide as the
    concatenation of the declared taps -- and each of those is the result of a
    tap concat in this graph, not some other live tensor read off the block.
    """
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    tap_width = len(TAPS) * normalized.hidden_size
    base = _output_types(staged_decode_graph)
    tapped = _output_types(staged_tap_graph)
    # The logits come first, unchanged, and the logits are the LM head's.
    assert tapped[: len(base)] == base
    assert str(normalized.vocab_size) in base[0], base[0]
    taps = tapped[len(base) :]
    assert len(taps) == DEVICES, (
        f"the tap arm emits {len(taps)} tap tensors for {DEVICES} ranks"
    )
    for rank, tap in enumerate(taps):
        assert f"[total_seq_len, {tap_width}]" in tap, tap
        assert tap.endswith(f"gpu:{rank}>"), tap
    # Each tap operand is the value the tap concat produced, so the output is
    # tied to the capture rather than merely being some wide tensor.
    concats = {
        result
        for result, line in _tap_concat_results(
            staged_decode_graph, staged_tap_graph
        )
    }
    base_operands = _output_operands(staged_decode_graph)
    tapped_operands = _output_operands(staged_tap_graph)
    assert tapped_operands[: len(base_operands)] == base_operands, (
        "the tap arm returns different logits operands: "
        f"{tapped_operands[: len(base_operands)]} vs {base_operands}"
    )
    assert set(tapped_operands[len(base_operands) :]) == concats, (
        "the tap outputs are not the tap concats: "
        f"{tapped_operands[len(base_operands) :]} vs {sorted(concats)}"
    )


class _TapConcat(Module[..., Tensor]):
    """Traces one tap concat, so its column order is read off the graph."""

    def __init__(self, order: tuple[int, ...]) -> None:
        """Records the declared order to concatenate in."""
        super().__init__()
        self.order = order

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        """Concatenates the two taps in the declared order."""
        # Keyed in the opposite order on purpose: the columns follow ``order``,
        # not the mapping's insertion order.
        return concat_taps({3: second, 1: first}, self.order)


def test_the_taps_are_concatenated_in_the_declared_order() -> None:
    """Tap ``1``'s columns come before tap ``3``'s, as the checkpoint names them.

    The drafter's projection columns are order-dependent, so this reads the
    staged concat's operand order rather than trusting the list literal.
    """
    cpu = DeviceRef.CPU()
    with F.lazy():
        module = _TapConcat((1, 3))
    graph = str(
        module.trace(
            TensorType(DType.bfloat16, ["rows", 8], device=cpu),
            TensorType(DType.bfloat16, ["rows", 8], device=cpu),
        )
    )
    concat = [line.strip() for line in graph.splitlines() if "concat" in line]
    assert len(concat) == 1, graph
    # The operands are the op's parenthesized list; the trace names the first
    # input ``%arg0`` and the second ``%arg1``, so the declared order is the one
    # that lists ``%arg0`` first.
    operands = re.search(r"concat\(([^)]*)\)", concat[0])
    assert operands is not None, concat[0]
    first, second = [
        operand.strip() for operand in operands.group(1).split(",")
    ]
    assert (first, second) == ("%arg0", "%arg1"), concat[0]


def test_capturing_taps_leaves_the_logits_subgraph_alone(
    staged_decode_graph: str, staged_tap_graph: str
) -> None:
    """The tap arm's ops are the non-tap arm's, plus the concat itself.

    The capture reads the residual stream and changes nothing about it, so the
    graph that computes the logits must be the same graph. Only the tap
    concat, the extra outputs and the signature may differ -- and a difference
    is a difference per op line, so a changed operand or a changed count in
    any op, concats included, is caught: every line of the non-tap arm must
    still be there, and what is added must be the tap concats alone.
    """
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    tap_width = len(TAPS) * normalized.hidden_size
    missing = _op_lines(staged_decode_graph) - _op_lines(staged_tap_graph)
    assert not missing, (
        "capturing taps changed or dropped ops the logits are built from:\n"
        + "\n".join(sorted(missing)[:10])
    )
    extra = _op_lines(staged_tap_graph) - _op_lines(staged_decode_graph)
    # One concat per rank: the taps are sharded like every other tensor here.
    assert len(extra) == DEVICES, (
        f"capturing taps added {len(extra)} op lines, not {DEVICES}:\n"
        + "\n".join(sorted(extra)[:10])
    )
    for line in extra:
        assert "concat" in line, line
        assert f") -> !mo.tensor<[total_seq_len, {tap_width}]," in line, line
    # The logits are still the graph's first output, at the same type.
    assert (
        _output_types(staged_tap_graph)[0]
        == _output_types(staged_decode_graph)[0]
    )


def _op_lines(graph: str) -> Counter[str]:
    """The graph's operation lines, without its signature or its outputs.

    The two arms differ in their signature, in their ``mo.output`` terminator
    and in one concat per rank, so those are what this leaves out. Everything
    else is compared as a multiset: an op whose operands changed appears in
    both graphs, which a set difference would hide.
    """
    return Counter(
        line.strip()
        for line in graph.splitlines()
        if line.strip()
        and not line.lstrip().startswith(
            ("mo.graph", "func.func", "mo.output", "mo.return", "}")
        )
    )


def _tap_concat_results(
    base_graph: str, tap_graph: str
) -> list[tuple[str, str]]:
    """The ``(result name, line)`` of every op the tap arm adds.

    The tap arm's added lines are its tap concats, one per rank, so their
    result names are what the extra outputs have to be.
    """
    extra = _op_lines(tap_graph) - _op_lines(base_graph)
    results: list[tuple[str, str]] = []
    for line in extra:
        match = re.match(r"(%[\w.$]+) = ", line)
        assert match is not None, line
        results.append((match.group(1), line))
    return results


def _output_operands(graph: str) -> list[str]:
    """The SSA names the graph's ``mo.output`` returns, one per output."""
    line = _output_line(graph)
    operands, _, _ = line.partition(" : ")
    return [part.strip() for part in operands.split(" ", 1)[1].split(",")]


def _output_types(graph: str) -> list[str]:
    """The types of the graph's ``mo.output`` operands, one per output.

    A type contains commas of its own (a shape's symbolic dims, a device
    tuple), so the operand list is split at depth zero of the angle brackets
    that wrap each type rather than at every comma.
    """
    line = _output_line(graph)
    _, _, types = line.partition(" : ")
    depth = 0
    parts: list[str] = []
    current = ""
    for char in types:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return [part for part in parts if part]


def _output_line(graph: str) -> str:
    """The graph's ``mo.output`` terminator, as one line."""
    return next(
        entry.strip()
        for entry in graph.splitlines()
        if re.search(r"\bmo\.output\b", entry)
    )
