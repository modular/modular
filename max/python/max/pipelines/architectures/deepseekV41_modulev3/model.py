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
"""Loads and compiles the DeepSeek-V4.1 text ModuleV3 model."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from max import functional as F
from max.driver import Buffer, Device, Usage, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    EPInit,
    calculate_ep_max_tokens_per_rank,
)

# The export's ep_init graph declares its counter buffers by the same word
# count EPCommInitializer allocates at runtime, so the engine introspects
# from the MEF the size it must zero-fill at boot.
from max.nn.legacy.comm.ep.ep_manager import _ep_counter_words
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    KVCacheConfig,
    ModuleV3Outputs,
    ModuleV3PipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.support.host_memory import available_host_memory, host_memory_limit
from max.support.human_readable_formatter import to_human_readable_bytes
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoTokenizer
from typing_extensions import override

if TYPE_CHECKING:
    from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import DeepseekV41BatchProcessor
from .deepseekV41 import DeepseekV41
from .inputs import v41_config
from .interface import extra_inputs
from .layers.engram_token_map import (
    check_map_digest,
    compressed_token_map_array,
)
from .model_config import DeepseekV41Config
from .weight_adapters import AdapterResult, EngramTable, engram_embed_tables

logger = logging.getLogger("max.pipelines")
_TABLE_WEIGHT_COLS = 256
_TABLE_SCALE_COLS = 8
# Host-RAM budget for the load (the pinned Engram tables can be very large).
# Bounded by total RAM as well as what is free now, so an undersized host
# fails fast -- the same bound the V2 loader uses.
_ENGRAM_HOST_RAM_MAX_FRACTION_OF_TOTAL = 0.8


@dataclass
class _PinnedTable:
    """One pinned Engram table under an identity key.

    ``digest`` is the table's content identity for the reload dedupe,
    computed only once a second table asks for the same slot: the first
    load of a shape has nothing to compare against and must not stream
    a table that reaches tens of gigabytes through a hash.
    """

    digest: bytes | None
    buffer: Buffer


# Pinned host tables, process-global by design: one serving process loads one
# checkpoint, so the map holds one model's tables and shares them across
# reloads of the same weights. The ceiling: a process that loads a second
# checkpoint pins both models' tables (production tables run to ~98 GB)
# for its lifetime -- add eviction on model unload before serving two
# checkpoints from one process.
_PINNED_TABLES: dict[tuple[str, int, str, int, int], list[_PinnedTable]] = {}


def _table_digest(codes: np.ndarray) -> bytes:
    """One table's content identity, hashed in place (no copy)."""
    # ``ndarray.data`` is the zero-copy memoryview of the array's storage.
    return hashlib.blake2b(codes.data, digest_size=16).digest()


def _check_engram_host_ram(name: str, byte_count: int) -> None:
    """Fails fast if pinning ``byte_count`` host bytes would exhaust RAM.

    The pinned Engram tables are the one un-reclaimable host allocation
    this loader adds: every module weight is a safetensors view, so the
    bytes about to be pinned, measured against what is free right now,
    is the whole projection. The budget mirrors the V2 loader's -- the
    same fraction of total RAM, bounded by what is actually available
    (the V2 loader also counts its host-staged weights, which it
    materializes).

    Raises:
        RuntimeError: If the pin would exceed the host-RAM budget.
    """
    total_bytes = host_memory_limit()
    available_bytes = available_host_memory()
    if total_bytes is None and available_bytes is None:
        logger.warning(
            "Unable to determine host memory capacity; skipping the Engram "
            "host-RAM preflight check."
        )
        return
    budget_bytes = min(
        b
        for b in (
            int(total_bytes * _ENGRAM_HOST_RAM_MAX_FRACTION_OF_TOTAL)
            if total_bytes is not None
            else None,
            available_bytes,
        )
        if b is not None
    )
    if byte_count > budget_bytes:
        raise RuntimeError(
            f"Pinning {name} needs {to_human_readable_bytes(byte_count)} "
            "of page-locked host RAM, which exceeds the "
            f"{to_human_readable_bytes(budget_bytes)} budget this box can "
            f"safely provide ({_ENGRAM_HOST_RAM_MAX_FRACTION_OF_TOTAL:.0%} "
            "of its "
            f"{to_human_readable_bytes(total_bytes) if total_bytes is not None else 'unknown'} "
            "total RAM, bounded by what is actually available right now). "
            "Provision more host memory, or free what is in use, before "
            "retrying."
        )


def _pinned_table(
    device: Device,
    name: str,
    rows: int,
    width: int,
    codes: np.ndarray,
    dtype: DType,
) -> Buffer:
    """A page-locked host copy of one Engram table, allocated once.

    ``Usage.STAGING`` is page-locked only against the accelerator: the
    driver maps a ``CPU()`` staging request to ordinary pageable memory,
    which the UVA gather must not read. The pinned buffer is still
    accepted for the ABI's CPU-typed ``engram_table_*`` inputs (the
    ``test_engram_ops`` host-table pattern, which runs exactly this
    placement through the compiled model).

    Args:
        device: The accelerator the gather reads the table through.
        name: The ABI input this table feeds.
        rows: Rows in the table.
        width: Bytes per row, the ABI's column count.
        codes: The checkpoint's table bytes, as a uint8 ``[rows, width]``
            view of its mapping.
        dtype: The ABI dtype the bytes relabel to (E4M3 codes or E8M0
            scales).

    Returns:
        The pinned buffer, shared with every earlier caller that asked for
        the same bytes on the same device.
    """
    key = (device.label, device.id, name, rows, width)
    entries = _PINNED_TABLES.get(key, [])
    # Content identity for the reload dedupe, hashed only when a same-shaped
    # table is already pinned to compare against: a first load pays nothing.
    digest = _table_digest(codes) if entries else None
    for entry in entries:
        if entry.digest is None:
            entry.digest = _table_digest(entry.buffer.to_numpy())
        if entry.digest == digest:
            return entry.buffer.view(dtype)
    _check_engram_host_ram(name, rows * width)
    buffer = Buffer(
        shape=(rows, width),
        dtype=DType.uint8,
        device=device,
        usage=Usage.STAGING,
    )
    assert buffer.pinned, (
        f"{name}: staging on {device} is not page-locked; pin against the "
        "accelerator the gather reads the table through"
    )
    # ``to_numpy`` aliases the allocation rather than copying it (the same
    # contract ``nn.parallel.numpy_parallel_ops`` writes staging memory
    # through), so the checkpoint's bytes land in the pinned allocation
    # itself. ``Buffer.from_numpy`` would first materialize a writable
    # copy of its own of the read-only checkpoint view -- a second
    # host-sized copy of a table that reaches tens of gigabytes.
    view = buffer.to_numpy()
    assert not view.flags.owndata, "staging to_numpy must alias the buffer"
    np.copyto(view, codes)
    _PINNED_TABLES.setdefault(key, []).append(_PinnedTable(digest, buffer))
    # The kernel consumes the raw bytes as E4M3 codes / E8M0 scales:
    # a bit-preserving view, no value conversion.
    return buffer.view(dtype)


def build_ep_config(
    pipeline_config: PipelineConfig, text: Any, n_devices: int
) -> EPConfig:
    """Builds the ModuleV3 EP config from the pipeline ``ep_size``.

    A module-level function so the memory planner sizes the same buffers
    the model allocates: ``text`` is any config view carrying
    ``hidden_size``, ``num_hidden_layers``, ``num_experts_per_tok``,
    ``n_routed_experts`` and ``moe_intermediate_size`` (the frozen
    ``DeepseekV41Config`` or the HF ``text_config``).

    NVFP4 dispatch matches both the experts' W4A4 grouped chain (the
    dispatch quantizes with the checkpoint's static per-expert
    ``input_scale``, the same scale the tensor-parallel path quantizes
    with) and vLLM's dispatch cost (about 2.9 KB per routed row). The
    combine returns bf16, as vLLM's does.

    Raises:
        ValueError: If DSpark is on without ``max_batch_size``, which
            bounds the draft rows the dispatch buffers must also hold.
    """
    runtime = pipeline_config.runtime
    ep_size = runtime.ep_size
    dp = pipeline_config.model.data_parallel_degree
    rows = runtime.max_batch_input_tokens
    speculative = pipeline_config.speculative
    if speculative is not None and speculative.speculative_method == "dspark":
        # Verification runs each request's drafts beside its active tokens,
        # and the scheduler's token budget counts only the active tokens.
        if runtime.max_batch_size is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark with expert parallelism needs "
                "max_batch_size, which bounds the draft rows a verification "
                "step adds to max_batch_input_tokens"
            )
        rows += runtime.max_batch_size * (
            speculative.num_speculative_tokens or 0
        )
    n_layers = int(text.num_hidden_layers)
    # The routed experts' storage quantization, which the dispatch
    # token format and the fused SwiGLU kernel read: modelopt NVFP4
    # (16-value E2M1 blocks, E4M3 per-block scales, static fp32
    # input scales).
    dispatch_quant_config = QuantConfig(
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
    return EPConfig(
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=dispatch_quant_config,
        combine_dtype=DType.bfloat16,
        hidden_size=text.hidden_size,
        top_k=int(text.num_experts_per_tok),
        n_experts=text.n_routed_experts,
        max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=rows,
            ep_size=ep_size,
            data_parallel_degree=dp,
            use_allreduce=False,
        ),
        n_gpus_per_node=n_devices,
        n_nodes=max(1, ep_size // n_devices),
        moe_dim=int(text.moe_intermediate_size),
        fused_shared_expert=False,
        use_allreduce=False,
    )


def validate_engram_tables(
    tables: Sequence[EngramTable], config: DeepseekV41Config
) -> None:
    """Checks the adapter's tables against the ABI's own table extents.

    Pure check, no allocation: it runs before the tokenizer is loaded and
    before any table is pinned, so a wrong handoff costs nothing.

    Args:
        tables: The adapter's tables, in ``engram_layer_ids`` order.
        config: The frozen interface view, whose ``engram_num_embeddings``
            is the ABI's per-ordinal row count.

    Raises:
        ValueError: If the adapter handed over a different number of tables
            than the ABI declares, or any table's row count differs from
            the ABI's extent for its ordinal.
    """
    want = tuple(int(rows) for rows in config.engram_num_embeddings)
    if len(tables) != len(want):
        raise ValueError(
            f"the weight adapter handed over {len(tables)} Engram tables, "
            f"but the DeepSeek-V4.1 ABI declares {len(want)} for layers "
            f"{list(config.engram_layer_ids)}"
        )
    for ordinal, table in enumerate(tables):
        if int(table.rows) != want[ordinal]:
            raise ValueError(
                f"engram_table_{ordinal}_weight is [{int(table.rows)}, "
                f"{_TABLE_WEIGHT_COLS}], but the ABI declares "
                f"[{want[ordinal]}, {_TABLE_WEIGHT_COLS}] for layer "
                f"{table.layer}"
            )


def serving_token_map(
    pipeline_config: PipelineConfig, config: DeepseekV41Config
) -> np.ndarray:
    """The serving tokenizer's compressed-id map, digest-checked.

    N-grams hash compressed ids, so a wrong map silently hashes
    unrelated tokens alike -- the placeholder it replaces was all zeros.
    The map is built from the checkpoint's own tokenizer at the
    configured model revision -- the same revision the pipeline
    resolves the config, weights and tokenizer files at, so serving
    from a hub repo id works without any pre-seeded snapshot -- and
    checked against the recorded sha256, so a ``tokenizers`` Unicode
    bump or a swapped checkpoint fails here instead of serving
    meaningless Engram output. The model binds the array once as the
    ``engram_token_map`` graph input the engine feeds at boot; the
    trunk's Engram layers also read it as a constructor constant, so
    the input and the constant carry the same bytes.

    Args:
        pipeline_config: The pipeline whose checkpoint names the
            tokenizer.
        config: The frozen interface view, whose ``vocab_size`` sizes
            the map.

    Returns:
        The map as int32 ``[vocab_size]``, in tokenizer id order.

    Raises:
        ValueError: If the tokenizer's vocabulary is not the model's
            size, or the map's sha256 is not the recorded digest.
    """
    model = pipeline_config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_path,
        revision=model.huggingface_model_revision,
        trust_remote_code=model.trust_remote_code,
    )
    token_map = compressed_token_map_array(tokenizer)
    if len(token_map) != config.vocab_size:
        raise ValueError(
            f"the {model.model_path} tokenizer holds {len(token_map)} "
            "ids, but the DeepSeek-V4.1 Engram token map is "
            f"[{config.vocab_size}]"
        )
    digest = check_map_digest(token_map)
    logger.info(
        "DeepSeek-V4.1 Engram token map sha256 %s (from %s)",
        digest,
        model.model_path,
    )
    return token_map


def engram_static_inputs(
    tables: Sequence[EngramTable], token_map: np.ndarray, device: Device
) -> dict[str, Buffer]:
    """Pins the Engram tables and stages the token map, by ABI input name.

    Pinned against ``device``, the pipeline's first.
    """
    static: dict[str, Buffer] = {}
    for ordinal, table in enumerate(tables):
        weight_name, scale_name = table.table_names(ordinal)
        static[weight_name] = _pinned_table(
            device,
            weight_name,
            table.rows,
            256,
            table.weight_codes,
            DType.float8_e4m3fn,
        )
        static[scale_name] = _pinned_table(
            device,
            scale_name,
            table.rows,
            8,
            table.scale_codes,
            DType.float8_e8m0fnu,
        )
    # The engine feeds the token map through the ABI's ``engram_token_map``
    # input; the trunk's Engram layers also read it as a constructor
    # constant, so the input and the constant carry the same bytes.
    static["engram_token_map"] = Buffer.from_numpy(token_map).to(device)
    return static


def check_parallel_layout(pipeline_config: PipelineConfig) -> None:
    """Raises unless the pipeline asks for a layout this trunk wires.

    Raises:
        ValueError: If more than one device serves other than data-parallel
            attention with expert-parallel MoE over all of them, if
            ``ep_size`` isn't the device count, or if expert parallelism
            runs without chunked prefill.
    """
    ep_size = pipeline_config.runtime.ep_size
    dp = pipeline_config.model.data_parallel_degree
    # The arch registration lets every multi-GPU layout through, but this
    # trunk wires exactly one: DP attention with EP over all the devices.
    # Everything else -- plain tensor parallelism included -- dies deep in
    # graph staging, so refuse it before the graphs compile, naming the
    # layout that works. The config's ``device_specs`` is read rather than
    # a model's devices: an EP1 build (a graph-contract recorder can stage
    # the model without ``__init__``) never touches the device list.
    n_devices = len(pipeline_config.model.device_specs)
    if n_devices > 1 and (dp != n_devices or ep_size != n_devices):
        raise ValueError(
            f"DeepSeek-V4.1 serves {n_devices} devices only as "
            "data-parallel attention with expert-parallel MoE: pass "
            f"--data-parallel-degree {n_devices} --ep-size "
            f"{n_devices} (got data_parallel_degree={dp}, "
            f"ep_size={ep_size}). Plain tensor parallelism is not "
            "wired for this trunk."
        )
    if ep_size <= 1:
        return
    if ep_size != n_devices:
        # Reachable on a single device with ep_size > 1: the check above
        # already refused every multi-device mismatch.
        raise ValueError(
            f"ep_size={ep_size} must equal the device count "
            f"({n_devices}) on the single-node DP-attention shape."
        )
    if not pipeline_config.runtime.enable_chunked_prefill:
        # The dispatch/combine buffers are sized to max_tokens_per_rank
        # (max_batch_input_tokens on this P2P single-node shape), which the
        # batch constructor treats as a soft limit: with chunking off it
        # admits an oversized prompt whole, and the dispatch overflows the
        # fixed buffers.
        raise ValueError(
            "DeepSeek-V4.1 expert parallelism requires chunked prefill: "
            "the EP dispatch buffers are sized to "
            "max_batch_input_tokens, which without chunking is a soft "
            "limit a longer prompt exceeds."
        )


class DeepseekV41Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext, DeepseekV41Config],
):
    """Loads the text trunk and binds its shared pinned Engram tables.

    With ``data_parallel_degree > 1`` (and ``ep_size == n_devices``) the
    graph is DP-attention + EP-MoE: every rank runs the full attention
    heads on its own request shard and the routed experts stay
    expert-sharded with a P2P dispatch/combine (the DeepseekV3 ModuleV3
    ``DP_EP`` pattern). The tokens stay one global batch on device 0; the
    graph splits them per rank from ``data_parallel_splits``.
    """

    _engram_tables: list[EngramTable]
    _token_map: np.ndarray

    # gen_mef's EP backend check: with DP attention each rank's dispatch
    # tokens are its own rows, which only the P2P backend serves. Irrelevant
    # at ep_size == 1 (the check returns before reading it).
    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"

    model_config_cls: ClassVar[type[Any]] = DeepseekV41Config
    batch_processor_cls: ClassVar[type[DeepseekV41BatchProcessor]] = (
        DeepseekV41BatchProcessor
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        # Capture the session so _init_distributed_runtime can run the EP
        # handshake outside virtual-device mode.
        self.session = session
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

    def _build_ep_config(
        self, model_config: DeepseekV41Config, n_devices: int
    ) -> EPConfig:
        """Builds the ModuleV3 EP config from the pipeline ``ep_size``."""
        return build_ep_config(self.pipeline_config, model_config, n_devices)

    @override
    def _init_distributed_runtime(
        self, model_config: DeepseekV41Config
    ) -> None:
        super()._init_distributed_runtime(model_config)
        self._ep_batch_manager: EPBatchManager | None = None
        self.ep_comm_initializer: EPCommInitializer | None = None

        check_parallel_layout(self.pipeline_config)
        if self.pipeline_config.runtime.ep_size <= 1:
            return
        n_devices = len(self.devices)
        # DeepseekV41Config is mutable; the EP config lives on the batch
        # manager the rest of the runtime already reads (graph_specs
        # included).
        ep_config = self._build_ep_config(model_config, n_devices)
        self._ep_batch_manager = EPBatchManager(ep_config)
        self._modulev3_extra_input_types = self._ep_batch_manager.input_types()
        if not is_virtual_device_mode():
            # In-process serving allocates the symmetric heap and the
            # counters now, and the base load_graphs hands the buffers to
            # the batch processor, so every forward's input tail carries
            # them. The export leaves them to the engine's handshake,
            # which owns the counters the compiled forward shares.
            self.ep_comm_initializer = EPCommInitializer(ep_config)
            self.ep_comm_initializer.ep_init(self.session)
            # run_init already wrote node_id through the shared config;
            # restate it like the deepseekV3 pattern does.
            ep_config.node_id = self.ep_comm_initializer.config.node_id

    @override
    def graph_specs(
        self, model_config: DeepseekV41Config
    ) -> dict[str, GraphSpec]:
        """The decoder, plus the EP handshake graph when EP is on.

        The ``ep_init`` graph allocates the engine-side symmetric heap and
        returns its buffer addresses, so the engine runs it once at startup
        with its own atomic counters (``mach/docs/expert-parallelism.md``,
        contract 1). The export never runs it in-process: the engine's
        handshake owns the counters the forward shares.
        """
        specs = super().graph_specs(model_config)
        ep_batch_manager = getattr(self, "_ep_batch_manager", None)
        ep_config = ep_batch_manager.config if ep_batch_manager else None
        if ep_config is not None:
            counter_shape = [
                _ep_counter_words(
                    ep_config.n_experts,
                    ep_config.n_gpus_per_node,
                    ep_config.use_allreduce,
                    ep_config.moe_dim,
                )
            ]
            init = EPInit(ep_config, counter_shape)
            specs[GraphRole.EP_INIT] = GraphSpec(
                init,
                "ep_init",
                init.input_types(),
                weights={},
            )
        return specs

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV41, DeepseekV41Config]:
        """Builds the trunk for a DSpark pairing, without its weights.

        The module compiles with the weights :meth:`checkpoint_weights`
        returns. Its Engram tables are pinned here and reach the pairing's
        batching on the config, as the graph inputs they are, with the
        trunk's extras the graph takes past the canonical tail. The
        pairing's verify gathers the logit rows it scores and taps the
        layers the DSpark stages read, so the trunk keeps the next-token
        outputs that data parallelism allows.

        Raises:
            ValueError: If ``target_layer_ids`` names other layers than the
                DSpark stages read, or the layout isn't one this trunk
                wires (:func:`check_parallel_layout`).
        """
        pipeline_config = context.pipeline_config
        check_parallel_layout(pipeline_config)
        config = DeepseekV41Config.initialize(
            pipeline_config, max_seq_len=context.max_seq_len
        )
        layers = list(context.target_layer_ids)
        if layers and layers != config.dspark_target_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 collects the states at its"
                f" dspark_target_layer_ids {config.dspark_target_layer_ids};"
                f" got target_layer_ids {layers}."
            )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, MultiKVCacheParams)
            config.kv_params = context.kv_params
        assert isinstance(config.kv_params, MultiKVCacheParams)
        devices = [ref.to_device() for ref in config.devices]
        dp = pipeline_config.model.data_parallel_degree
        tables = engram_embed_tables(context.weights, config.engram_layer_ids)
        validate_engram_tables(tables, config)
        token_map = serving_token_map(pipeline_config, config)
        config.engram_static_inputs = engram_static_inputs(
            tables, token_map, devices[0]
        )
        config.dspark_trailing_inputs = extra_inputs(
            v41_config(pipeline_config.model.huggingface_config),
            config.devices,
            dp,
            verify_rows=True,
        )
        ep_batch_manager: EPBatchManager | None = None
        if pipeline_config.runtime.ep_size > 1:
            ep_config = build_ep_config(pipeline_config, config, len(devices))
            context.start_ep(ep_config)
            ep_batch_manager = EPBatchManager(ep_config)
        axis = "dp" if dp > 1 else "tp"
        mesh = DeviceMesh(tuple(devices), (len(devices),), (axis,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            module = DeepseekV41(
                config,
                config.kv_params,
                mesh,
                text_config=DeepseekV41Config.get_text_config(
                    pipeline_config.model.huggingface_config
                ),
                token_map=token_map,
                data_parallel_degree=dp,
                ep_batch_manager=ep_batch_manager,
            )
        return module, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DeepseekV41Config
    ) -> dict[str, Any]:
        """Returns the weights the built trunk compiles with.

        The Engram tables the adapter hands back beside them are not
        weights: :meth:`build_module` pinned them as graph inputs.
        """
        del config
        return dict(
            parse_state_dict_from_weights(
                context.pipeline_config, context.weights, context.adapter
            )
        )

    def _load_state_dict(self) -> dict[str, Any]:
        result = super()._load_state_dict()
        if not isinstance(result, AdapterResult):
            raise ValueError(
                "DeepSeek-V4.1 requires its safetensors weight adapter"
            )
        self._engram_tables = result.engram_tables
        self._checkpoint = result.state_dict
        return result.state_dict

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV41Config:
        del state_dict
        config = DeepseekV41Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        validate_engram_tables(self._engram_tables, config)
        self._token_map = serving_token_map(self.pipeline_config, config)
        return config

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict, model_config
        return DType.bfloat16

    @override
    def _instantiate_module(
        self, model_config: DeepseekV41Config
    ) -> DeepseekV41:
        assert isinstance(self.kv_params, MultiKVCacheParams)
        # The single mesh axis is the DP axis under data parallelism: the
        # vocab-parallel embedding and head shard their weight over it by
        # name, and every other weight replicates.
        dp = self.pipeline_config.model.data_parallel_degree
        axis = "dp" if dp > 1 else "tp"
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), (axis,))
        return DeepseekV41(
            model_config,
            self.kv_params,
            mesh,
            text_config=DeepseekV41Config.get_text_config(
                self.huggingface_config
            ),
            token_map=self._token_map,
            data_parallel_degree=dp,
            ep_batch_manager=getattr(self, "_ep_batch_manager", None),
        )

    def _load_decoder(
        self, session: InferenceSession
    ) -> Callable[..., ModuleV3Outputs]:
        decoder = super()._load_decoder(session)
        self._bind_engram_static_inputs()
        return decoder

    def _bind_engram_static_inputs(self) -> None:
        """Pins the adapter's table views and hands them to the processor.

        Once every table lives in page-locked memory the adapter's views
        are dropped: the loader never keeps a second copy of a table that
        reaches tens of gigabytes.
        """
        processor = self.batch_processor
        assert isinstance(processor, DeepseekV41BatchProcessor)
        processor.bind_engram(
            engram_static_inputs(
                self._engram_tables, self._token_map, self.devices[0]
            )
        )
        self._engram_tables = []
