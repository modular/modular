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
"""The DP4+EP4 graph stages and compiles on virtual devices, no all-reduce.

Exercises the data-parallel attention + expert-parallel MoE wiring end to
end at the IR level (the deepseekV3 ModuleV3 ``DP_EP`` pattern): the batch
splits per rank from ``data_parallel_splits``, the attention weights stay
whole per rank, and the routed experts dispatch/combine through the EP
buffers. Staging runs on virtual SM100 devices; the compile leg then runs
the staged graph through ``InferenceSession.compile_reusing_mefs`` -- the
entry point the MEF export compiles through -- still without a GPU, so a
graph that builds but whose operands disagree with a kernel's signature
(a dtype drift the CSA2 and Engram kernels reject) fails here in CI rather
than twenty minutes into an export. The step's collective inventory is also
pinned, so no unplanned cross-device transfer can slip into the captured
step.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
    extra_inputs,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import default_device
from max.tensor.sharding import DeviceMesh
from runtime_config import text_config

# The rig's DP width: four ranks, each running the full attention heads on
# its own request shard.
_DP = 4


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


def _dp_text_config(
    replay: bool = False, lookback_history: bool = False
) -> SimpleNamespace:
    """The synthetic text config, plus its two Engram layers.

    The Engram layers make the DP graph cover the request-sliced lookback
    fields and the per-rank shape ties the engram kernel's checks read.
    The Engram ops' table layout is fixed when their kernels are compiled
    (two layers, n-gram width 4, 8 heads, 256-byte rows), so the config
    carries that geometry and the table row counts the hash layout
    derives for it.

    The kernel-facing geometry is the checkpoint's, not the staging
    fixture's: the compile leg instantiates the real kernels, whose
    comptime constraints the real config satisfies and the tiny staging
    values do not -- the MXFP4 indexer's SM100 scorer takes only 64, 32,
    8 or 4-head queries (mla_index_fp8.mojo), the CSA2 attention tiles
    cover whole eight-row atoms (tensor_core_async.mojo), the expert
    FFN's K dims must be multiples of its BK (mega_ffn_kernel.mojo),
    the EP combine chunks the hidden size (ep_comm.mojo), and a
    candidate block must fit one warp's key span (mla_index_fp8.mojo
    again). Everything with no kernel constraint -- layer count,
    vocabulary, expert count, table rows -- stays tiny.

    ``replay`` exports the decoder's host-planned kept rows, which needs
    layers past the last KV source: two SWA-only layers are appended.
    ``lookback_history`` exports the Engram window's device history (the
    overlap mechanism).
    """
    text = text_config()
    if replay:
        text.num_hidden_layers = 4
        text.compress_ratios = [0, 1, 0, 0]
        text.decoder_replay = True
    if lookback_history:
        text.engram_lookback_history = True
    text.index_n_heads = 32
    text.num_attention_heads = 64
    text.hidden_size = 5120
    text.moe_intermediate_size = 2304
    text.q_lora_rank = 1280
    text.o_lora_rank = 1024
    text.o_groups = 8
    text.candidate_block_size = 8
    layout = EngramLayout(
        layer_ids=(0, 1),
        num_embeddings=(0, 0),
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        compressed_vocab_size=100,
        pad_token_id=2,
        vocab_size=20_000,
    )
    text.engram_layer_ids = [0, 1]
    text.engram_num_embeddings = [int(p.sum()) for p in layout.primes]
    text.engram_max_ngram_size = 4
    text.engram_n_heads = 8
    text.engram_head_dim = 256
    text.engram_compressed_vocab_size = 100
    text.engram_vocab_size = 20_000
    text.engram_pad_token_id = 2
    return text


def _nvfp4_dispatch_config(n_layers: int) -> QuantConfig:
    """The routed experts' modelopt NVFP4 storage quantization."""
    return QuantConfig(
        format=QuantFormat.NVFP4,
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(range(n_layers)),
        attn_quantized_layers=set(),
        can_use_fused_swiglu=True,
    )


def _ep_config(config: DeepseekV41Config, n_devices: int) -> EPConfig:
    """The production EP config, as the pipeline model builds it."""
    return EPConfig(
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=_nvfp4_dispatch_config(
            int(config.num_hidden_layers)
        ),
        combine_dtype=DType.bfloat16,
        hidden_size=config.hidden_size,
        top_k=int(config.num_experts_per_tok),
        n_experts=config.n_routed_experts,
        max_tokens_per_rank=4096,
        n_gpus_per_node=n_devices,
        n_nodes=1,
        moe_dim=int(config.moe_intermediate_size),
        fused_shared_expert=False,
        use_allreduce=False,
    )


def _dp_ep_compile_inputs(
    config: V41Config,
    kv_params: MultiKVCacheParams,
    device_refs: list[DeviceRef],
    ep_manager: EPBatchManager,
    dp: int,
) -> list[TensorType | BufferType]:
    """The forward's input types in DP ABI order, before signal buffers.

    The CPU DP split pair follows ``input_row_offsets`` and precedes the
    KV leaves, matching ``DeepseekV41.forward``'s variadic order.
    """
    gpu0 = device_refs[0]
    inputs: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["total_seq_len"], gpu0),
        TensorType(DType.int64, ["return_n_logits"], DeviceRef.CPU()),
        TensorType(DType.uint32, ["input_row_offsets_len"], gpu0),
        TensorType(DType.int64, [dp + 1], DeviceRef.CPU()),
        TensorType(DType.int64, ["input_row_offsets_len"], DeviceRef.CPU()),
    ]
    inputs.extend(kv_params.flattened_kv_inputs())
    inputs.extend(
        t for _, t in extra_inputs(config, device_refs, data_parallel_degree=dp)
    )
    inputs.extend(ep_manager.input_types())
    return inputs


def _build_dp4_ep4(
    replay: bool = False, lookback_history: bool = False
) -> tuple[DeepseekV41, V41Config, list[TensorType | BufferType]]:
    """The DP4+EP4 module, its interface config and its compile inputs."""
    text = _dp_text_config(replay, lookback_history)
    interface = V41Config.from_hf({"text_config": vars(text)})
    device_refs = [DeviceRef.GPU(i) for i in range(_DP)]
    kv = build_kv_params(
        interface,
        KVCacheConfig(kv_cache_page_size=128),
        device_refs,
        data_parallel_degree=_DP,
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=device_refs,
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(_DP)),
        (_DP,),
        ("dp",),
    )
    ep_manager = EPBatchManager(_ep_config(config, _DP))
    with F.lazy(), default_device(mesh):
        module = DeepseekV41(
            config,
            kv,
            mesh,
            text_config=text,
            token_map=np.arange(text.vocab_size, dtype=np.int32),
            data_parallel_degree=_DP,
            ep_batch_manager=ep_manager,
        )
    inputs = _dp_ep_compile_inputs(interface, kv, device_refs, ep_manager, _DP)
    return module, interface, inputs


def _stage_dp4_ep4(replay: bool = False, lookback_history: bool = False) -> str:
    """Traces the DP4+EP4 graph and returns its staged text."""
    with _virtual_devices(_DP):
        module, _, inputs = _build_dp4_ep4(replay, lookback_history)
        return str(module.trace(*inputs))


@pytest.fixture(scope="module")
def staged_dp4_ep4() -> str:
    """The DP4+EP4 graph's staged text, traced once for this module.

    Staging builds the whole module and traces the full graph, so both
    tests read one trace through this module-scoped fixture rather than
    paying the staging cost per test.
    """
    return _stage_dp4_ep4()


def test_dp4_ep4_graph_stages_without_allreduce(staged_dp4_ep4: str) -> None:
    text = staged_dp4_ep4
    # The vocab-parallel embedding all-reduces its gather on the full
    # replicated batch before the token split (by design: every rank
    # still holds the same rows there). That is the graph's one
    # all-reduce; the DP trunk after the split has none -- every rank's
    # rows are completed by the EP dispatch/combine.
    assert text.count("mo.distributed.allreduce.sum") == 1, (
        "the DP graph must all-reduce only the embedding gather"
    )
    for needle in (
        "ep.dispatch",  # the P2P dispatch custom op
        "input_split_",  # the per-rank token split dims
    ):
        assert needle in text, f"missing {needle!r} in the staged graph"


def test_dp4_ep4_step_collectives_are_only_the_designed_ones(
    staged_dp4_ep4: str,
) -> None:
    """The step crosses devices only where the DP/EP design says so.

    Every collective here lowers into the captured step, and an unplanned
    one lowers into the host-enqueued waits (``mgp.device_wait``) and
    readbacks (``mgp.sync``) that device-graph capture rejects -- the
    failure the load-time dispatch scale and the host-resident row count
    exist to avoid. A reintroduced cross-device scalar hub (a per-step
    gather of the dispatch scale, a cross-rank sum of the row count)
    would surface as a collective outside this inventory; the set is
    pinned so it cannot return unnoticed. The lowered-IR gate over the
    checkpoint's real graph is ``serve/test_graph_capture_safe.py``,
    which lands with the capture work.
    """
    text = staged_dp4_ep4
    ops = set(re.findall(r"\bmo\.distributed\.[\w.]+", text))
    assert ops == {
        # The head's vocab gather, before the DP outputs leave the ranks.
        "mo.distributed.allgather",
        # The embedding's gather on the full replicated batch, once.
        "mo.distributed.allreduce.sum",
        # The staged inputs replicated per rank, and constants.
        "mo.distributed.broadcast",
        # The routed experts' P2P dispatch and combine, per MoE layer.
        "mo.distributed.ep.combine",
        "mo.distributed.ep.dispatch.block.scaled.nv",
    }, sorted(ops)


def test_dp4_ep4_off_graph_takes_no_replay_input(staged_dp4_ep4: str) -> None:
    """The default export is the graph the replay change found.

    The digest and op census are the evidence that it is byte-for-byte that
    graph: run this test on the commit before the change and on this one and
    the two ``graph digest`` lines must match. The lookback-history
    mechanism gates the same way, so the digest also covers it.
    """
    census = Counter(re.findall(r"\b(?:mo|mogg|mgp)\.[\w.]+", staged_dp4_ep4))
    print(
        "graph digest:",
        hashlib.sha256(staged_dp4_ep4.encode()).hexdigest(),
        f"ops={sum(census.values())}",
        f"kinds={len(census)}",
    )
    assert "replay.kept_offsets" not in staged_dp4_ep4
    assert "engram_lookback_ids" not in staged_dp4_ep4
    assert "engram_lookback_source" not in staged_dp4_ep4
    assert "engram_lookback_history" not in staged_dp4_ep4


def test_dp4_ep4_replay_graph_stages_with_the_designed_collectives() -> None:
    """The kept-row graph adds its split and no collective of its own."""
    text = _stage_dp4_ep4(replay=True)
    # The kept rows are a ragged batch of their own: each rank cuts its
    # requests' share with the token split, under its own dim labels.
    assert "replay_rows" in text
    assert text.count("mo.distributed.allreduce.sum") == 1
    assert set(re.findall(r"\bmo\.distributed\.[\w.]+", text)) == set(
        re.findall(r"\bmo\.distributed\.[\w.]+", _stage_dp4_ep4())
    )


def test_dp4_ep4_lookback_history_graph_stages_the_v2_mechanism() -> None:
    """The history export reads the window in-graph, with no new collective.

    The trace runs by input type, so what the mechanism leaves in the
    staged text is its history buffer's own dim: the read ran inside the
    trunk (a buffer input the DP wiring has never carried before). The
    read is device-0 compute, so the step's collective inventory is
    exactly the default's -- an unplanned collective would lower into the
    host waits that device-graph capture rejects.
    """
    text = _stage_dp4_ep4(lookback_history=True)
    assert "engram_history_rows" in text
    assert set(re.findall(r"\bmo\.distributed\.[\w.]+", text)) == set(
        re.findall(r"\bmo\.distributed\.[\w.]+", _stage_dp4_ep4())
    )


def test_dp4_ep4_graph_compiles() -> None:
    """The staged DP4+EP4 graph clears the graph compiler, no GPU.

    Staging alone cannot catch an operand whose dtype drifted from its
    kernel's signature: the CSA2, indexer and Engram custom ops
    type-check when the compiler parses the Mojo it generates for them,
    which is where a DP4/EP4 export failed after a seventeen-minute
    compile. Compiling the same staged graph through the export's entry
    point (``InferenceSession.compile_reusing_mefs``) fails on that drift
    in CI; the weights stay external, so no checkpoint is read.
    """
    with _virtual_devices(_DP):
        module, _, inputs = _build_dp4_ep4()
        staged = module.trace(*inputs)
        session = InferenceSession(devices=[Accelerator(i) for i in range(_DP)])
        session.compile_reusing_mefs(staged.graph)
