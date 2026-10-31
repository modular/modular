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
"""Implements the DeepSeek-V4.1-Flash PipelineModel.

Builds the servable graph on one device or tensor-parallel across several,
and loads the weights: routed experts stream straight to their devices, and
Engram tables are pinned in host memory while the graph compiles. Images are
encoded by a separately compiled vision model on device 0, whose span rows
the language graph merges into its token embeddings.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from max import functional as F
from max import tree
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    Usage,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    Weight,
    ops,
)
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    ReturnLogits,
)
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
)
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    compile_and_init_graphs,
)
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    SingleReplicaRaggedBatchProcessor,
    build_single_replica_ragged_token_arrays,
    ragged_batch_inputs,
    ragged_kv_symbolic_inputs,
    single_replica_context_batch,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import (
    CompilationTimer,
    compute_data_parallel_splits,
    parse_state_dict_from_weights,
)
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.profiler import traced
from max.support.host_memory import available_host_memory, host_memory_limit
from max.support.human_readable_formatter import to_human_readable_bytes
from max.tensor import default_dtype
from max.tensor.compilation import CompiledCallable
from transformers import AutoTokenizer

from . import engram_shared_host, kv_leaves, weight_adapters
from .batch_vision_inputs import pack_uncached_images
from .deepseekV41 import DeepseekV41, DeepseekV41Block, VisionMerge
from .engram_lookback import EngramLookbackBatchState
from .layers.attention import CSA2Cache
from .layers.engram_hash import EngramLayout, engram_lookback_token_ids
from .layers.engram_lookup import EngramLookup
from .layers.engram_token_map import compressed_token_map_array
from .layers.moe import DeepseekV41MoE
from .memory_planner import check_planned_max_length
from .model_config import (
    DeepseekV41Config,
    forward_row_bound,
    max_images_per_request,
)
from .replay import (
    ReplayPlan,
    build_replay_plan_with_reason,
    log_replay_not_built,
    replay_min_tokens,
    requested_min_tokens,
    sliding_window_leaf_id,
)
from .replay_writes import DSparkWriteBatch, DSparkWriteWitness
from .vision import (
    DeepseekV41VisionConfig,
    DeepseekV41VisionModel,
    vision_input_types,
)

logger = logging.getLogger("max.pipelines")

# Host-RAM budget for the load (Engram tables can be very large). Bounded by
# total RAM as well as what is free now, so an undersized host fails fast.
_ENGRAM_HOST_RAM_MAX_FRACTION_OF_TOTAL = 0.8


def _weight_data_bytes(weight_data: WeightData) -> int:
    """Returns ``weight_data``'s exact in-memory size, no decoding needed."""
    return (
        math.prod(weight_data.shape.static_dims)
        * weight_data.dtype.size_in_bytes
    )


def _expert_upload_window_bytes(
    state_dict: dict[str, Any], num_devices: int
) -> int:
    """The most host memory :func:`stream_routed_experts` holds at once.

    Every upload thread (``_UPLOAD_THREADS_PER_DEVICE`` per device) holds
    one expert's rows of one :class:`~.weight_adapters.DeferredExpertStack`
    at a time.
    """
    return (
        num_devices
        * _UPLOAD_THREADS_PER_DEVICE
        * max(
            (
                _weight_data_bytes(weight_data) // len(weight_data.parts)
                for weight_data in state_dict.values()
                if isinstance(weight_data, weight_adapters.DeferredExpertStack)
            ),
            default=0,
        )
    )


def _check_engram_host_ram_capacity(
    state_dict: dict[str, Any],
    model_config: DeepseekV41Config,
    num_devices: int = 1,
) -> None:
    """Fails fast if loading this checkpoint would exhaust host RAM.

    The projected peak is the pinned Engram tables (zero if a shared copy
    already exists), every weight staged through host memory, and the
    routed-expert and Engram streaming windows. Compiler and runtime memory
    is not projected.

    Raises:
        RuntimeError: If the projection exceeds the host-RAM budget.
    """
    engram_names = {
        f"trunk.layers.{layer_id}.engram_fn.embed{suffix}"
        for layer_id in model_config.engram_layer_ids
        for suffix in (".weight", ".scale")
    }
    for name in engram_names:
        assert isinstance(state_dict[name], WeightData)

    # Pinned once, shared by every device, at their checkpoint byte size.
    engram_pinned_bytes = sum(
        _weight_data_bytes(state_dict[name]) for name in engram_names
    )
    if engram_names and _shared_engram_copy_exists(state_dict, model_config):
        logger.info(
            "Host-RAM preflight: this host already shares a complete copy of "
            "the Engram tables (%s), which this load attaches to instead of "
            "pinning its own.",
            to_human_readable_bytes(engram_pinned_bytes),
        )
        engram_pinned_bytes = 0
    host_staged_bytes = sum(
        _weight_data_bytes(weight_data)
        for name, weight_data in state_dict.items()
        if isinstance(weight_data, WeightData)
        and not isinstance(weight_data, weight_adapters.DeferredExpertStack)
        and name not in engram_names
    )
    window_bytes = _expert_upload_window_bytes(state_dict, num_devices) + (
        _PIN_THREADS * _PIN_CHUNK_BYTES if engram_names else 0
    )
    projected_bytes = engram_pinned_bytes + host_staged_bytes + window_bytes

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
    logger.info(
        "Host-RAM preflight: projected peak %s (Engram pinned %s, host-staged "
        "weights %s, streaming windows %s) against a %s budget.",
        to_human_readable_bytes(projected_bytes),
        to_human_readable_bytes(engram_pinned_bytes),
        to_human_readable_bytes(host_staged_bytes),
        to_human_readable_bytes(window_bytes),
        to_human_readable_bytes(budget_bytes),
    )
    if projected_bytes > budget_bytes:
        raise RuntimeError(
            "Loading this checkpoint projects to "
            f"{to_human_readable_bytes(projected_bytes)} of host RAM "
            "(host-resident Engram tables plus the rest of the checkpoint's "
            "weight staging), which exceeds the "
            f"{to_human_readable_bytes(budget_bytes)} budget this box can "
            f"safely provide ({_ENGRAM_HOST_RAM_MAX_FRACTION_OF_TOTAL:.0%} "
            f"of its {to_human_readable_bytes(total_bytes) if total_bytes is not None else 'unknown'} "
            "total RAM, bounded by what is actually available right now). "
            "Provision more host memory, or free what is in use, before "
            "retrying."
        )


@dataclass(frozen=True)
class _ExpertUpload:
    """One device's share of one routed-expert tensor to upload."""

    name: str
    """The device-resident weight's registry name."""
    replaces: str
    """The registry name of the host weight it replaces."""
    source: weight_adapters.DeferredExpertStack
    start: int
    stop: int
    device: Device


_ROUTED_EXPERT_TENSORS = (
    ("gate_up_proj", "weight"),
    ("gate_up_proj", "weight_scale"),
    ("down_proj", "weight"),
    ("down_proj", "weight_scale"),
)


def _replace_weight(module: Module, attr: str, weight: Weight) -> None:
    """Sets ``module.attr`` to ``weight``, unregistering the weight it replaces.

    The old weight is re-keyed to its current name first, since
    ``load_state_dict`` may have renamed it after registration.
    """
    old = getattr(module, attr)
    registered = module.layer_weights
    key = next(key for key, value in registered.items() if value is old)
    registered[old.name] = registered.pop(key)
    setattr(module, attr, weight)


def make_routed_experts_device_resident(
    source: DeepseekV41,
    serving: Sequence[DeepseekV41],
    state_dict: Mapping[str, Any],
    devices: Sequence[Device],
    device_refs: Sequence[DeviceRef],
) -> list[_ExpertUpload]:
    """Points ``serving``'s routed-expert weights at device-resident buffers.

    For each deferred routed-expert tensor
    (:class:`~.weight_adapters.DeferredExpertStack`) in ``state_dict``, which
    holds ``source``'s loaded values by their path in it, each serving model
    gets a weight on its rank's device holding that rank's axis-0 run of
    experts, named ``{name}[{rank}]`` (``{name}`` on one device).

    Returns:
        The uploads :func:`stream_routed_experts` must perform.
    """
    n = len(serving)
    uploads: list[_ExpertUpload] = []
    for layer_idx, source_block in enumerate(source.trunk.layers):
        assert isinstance(source_block, DeepseekV41Block)
        assert isinstance(source_block.ffn, DeepseekV41MoE)
        for proj_name, attr in _ROUTED_EXPERT_TENSORS:
            source_weight = getattr(
                getattr(source_block.ffn.experts, proj_name), attr
            )
            assert isinstance(source_weight, Weight)
            deferred = state_dict.get(
                f"trunk.layers.{layer_idx}.ffn.experts.{proj_name}.{attr}"
            )
            if not isinstance(deferred, weight_adapters.DeferredExpertStack):
                continue
            n_experts, *rest = deferred.shape.static_dims
            local = n_experts // n
            for rank, model in enumerate(serving):
                block = model.trunk.layers[layer_idx]
                assert isinstance(block, DeepseekV41Block)
                assert isinstance(block.ffn, DeepseekV41MoE)
                name = (
                    source_weight.name
                    if n == 1
                    else f"{source_weight.name}[{rank}]"
                )
                resident = Weight(
                    name=name,
                    dtype=source_weight.dtype,
                    shape=(local, *rest),
                    device=device_refs[rank],
                )
                # Declared on its own device, so the engine binds the
                # uploaded buffer in place instead of staging it through
                # host memory. It is still an ordinary constant, so
                # computation on it alone is still lifted to the init phase.
                Graph.current.add_weight(
                    resident, force_initial_weight_on_host=False
                )
                _replace_weight(
                    getattr(block.ffn.experts, proj_name), attr, resident
                )
                uploads.append(
                    _ExpertUpload(
                        name,
                        source_weight.name,
                        deferred,
                        rank * local,
                        (rank + 1) * local,
                        devices[rank],
                    )
                )
    return uploads


@dataclass
class ExpertStreaming:
    """Routed experts a speculative pairing streams straight to their devices.

    Its verify makes them device-resident as it makes the serving copies,
    and a load deferred to the compile uploads them.
    """

    deferred: dict[str, weight_adapters.DeferredExpertStack]
    """The deferred routed-expert values, by their path in the model."""
    uploads: list[_ExpertUpload] = field(default_factory=list)
    """The uploads the verify planned."""


_UPLOAD_THREADS_PER_DEVICE = 4


def stream_routed_experts(
    uploads: Sequence[_ExpertUpload], weights_registry: dict[str, Any]
) -> None:
    """Uploads each device's share of the routed experts, one expert at a time.

    Each expert's rows are copied to the device and paged out before the
    next is read, so host memory holds one expert per upload thread rather
    than all routed experts. Each device uses ``_UPLOAD_THREADS_PER_DEVICE``
    threads that prefetch their next share (``will_need``), since checkpoint
    reads bound the upload. The copies hold the GIL, so this should not run
    concurrently with the graph compile.

    The device buffers replace the host weights in ``weights_registry``,
    which in virtual-device mode only drops the host weights.
    """
    for upload in uploads:
        weights_registry.pop(upload.replaces, None)
    if is_virtual_device_mode() or not uploads:
        # Compile-only: no device to upload to.
        return
    start_time = time.perf_counter()
    by_device: dict[int, list[_ExpertUpload]] = {}
    for upload in uploads:
        by_device.setdefault(id(upload.device), []).append(upload)
    queues = [
        device_uploads[i::_UPLOAD_THREADS_PER_DEVICE]
        for device_uploads in by_device.values()
        for i in range(_UPLOAD_THREADS_PER_DEVICE)
        if device_uploads[i::_UPLOAD_THREADS_PER_DEVICE]
    ]

    def upload_all(queue: list[_ExpertUpload]) -> dict[str, Buffer]:
        buffers: dict[str, Buffer] = {}
        queue[0].source.will_need(queue[0].start, queue[0].stop)
        for i, upload in enumerate(queue):
            if i + 1 < len(queue):
                ahead = queue[i + 1]
                ahead.source.will_need(ahead.start, ahead.stop)
            source = upload.source
            rest = source.shape.static_dims[1:]
            share = Buffer(
                shape=(upload.stop - upload.start, *rest),
                dtype=source.dtype,
                device=upload.device,
            )
            trailing = (slice(None),) * len(rest)
            for slot, expert in enumerate(range(upload.start, upload.stop)):
                host = source.materialize(expert, expert + 1)
                share[(slice(slot, slot + 1), *trailing)].inplace_copy_from(
                    Buffer.from_numpy(host).view(source.dtype)
                )
                del host
                source.page_out(expert, expert + 1)
            buffers[upload.name] = share
        return buffers

    with ThreadPoolExecutor(len(queues)) as pool:
        for result in pool.map(upload_all, queues):
            weights_registry.update(result)
    for device_uploads in by_device.values():
        device_uploads[0].device.synchronize()
    elapsed = time.perf_counter() - start_time
    total_bytes = sum(
        _weight_data_bytes(u.source) // len(u.source.parts) * (u.stop - u.start)
        for u in uploads
    )
    logger.info(
        "Uploaded %s of routed experts to %d devices in %.1fs (%.2f GB/s).",
        to_human_readable_bytes(total_bytes),
        len(by_device),
        elapsed,
        total_bytes / max(elapsed, 1e-9) / 1e9,
    )


def csa2_cache(
    specs: Sequence[kv_leaves.V41KVLeafSpec],
    groups: Sequence[Sequence[PagedCacheValues]],
    device_idx: int,
) -> CSA2Cache:
    """Builds device ``device_idx``'s attention cache from the KV leaves.

    ``groups`` holds one list per leaf, in ``specs`` order, with one entry
    per device.
    """
    by_key = {
        spec.key: group[device_idx]
        for spec, group in zip(specs, groups, strict=True)
    }
    # A "comp" leaf's ``owning_layer_ids`` order is its storage slot order.
    main_kv_by_layer: dict[int, PagedCacheValues] = {}
    index_k_by_layer: dict[int, PagedCacheValues] = {}
    comp_layer_index_by_layer_id: dict[int, int] = {}
    state_by_layer: dict[int, PagedCacheValues] = {}
    for spec in specs:
        if spec.kind != "comp":
            continue
        comp_leaf = by_key[spec.key]
        idx_leaf = by_key[f"idx_{spec.key}"]
        # Ratio 1 has no "state" leaf: every token is a complete compression
        # group on its own.
        state_leaf = by_key.get(f"{spec.key}_state")
        for slot, layer_id in enumerate(spec.owning_layer_ids):
            main_kv_by_layer[layer_id] = comp_leaf
            index_k_by_layer[layer_id] = idx_leaf
            comp_layer_index_by_layer_id[layer_id] = slot
            if state_leaf is not None:
                state_by_layer[layer_id] = state_leaf
    return CSA2Cache(
        swa_collection=by_key["swa"],
        main_kv_by_layer=main_kv_by_layer,
        index_k_by_layer=index_k_by_layer,
        comp_layer_index_by_layer_id=comp_layer_index_by_layer_id,
        state_by_layer=state_by_layer,
    )


def _load_engram_host_tables(
    state_dict: dict[str, Any],
    model_config: DeepseekV41Config,
    pin_device: Accelerator,
) -> dict[int, tuple[Buffer, Buffer]]:
    """Pops and pins each Engram layer's table out of ``state_dict``.

    Engram tables live in pinned host memory, never device memory, so they
    have no ``Weight``. The checkpoint's fp8/e8m0 bytes are copied in as-is.
    Pinned memory is visible to every accelerator, so ``pin_device`` is
    nominal and every device's graph input views these same buffers.

    When possible, all processes on the host share one copy; otherwise each
    pins a private copy.

    Returns:
        ``{layer_id: (weight, scale)}`` pinned host buffers.
    """
    codes = _engram_codes(state_dict, model_config, pop=True)
    shared = _open_shared_engram_tables(codes, pin_device)
    tables: dict[int, tuple[Buffer, Buffer]] = {}
    for layer_id in model_config.engram_layer_ids:
        prefix = _engram_prefix(layer_id)
        pair = []
        for suffix, dtype in _ENGRAM_TABLE_DTYPES:
            name = prefix + suffix
            pair.append(
                Buffer.from_numpy(shared.arrays[name]).view(dtype)
                if shared is not None
                else _pin_streamed(codes[name], dtype, pin_device)
            )
        tables[layer_id] = (pair[0], pair[1])
    return tables


_ENGRAM_TABLE_DTYPES = (
    (".weight", DType.float8_e4m3fn),
    (".scale", DType.float8_e8m0fnu),
)


def _engram_prefix(layer_id: int) -> str:
    return f"trunk.layers.{layer_id}.engram_fn.embed"


def _engram_row_ranges(
    model_config: DeepseekV41Config,
    token_map: np.ndarray | None,
    n_dev: int,
) -> list[list[tuple[int, int]]]:
    """Per rank, per Engram layer, the ``(first row, rows)`` it reads.

    One device, or each data-parallel device, reads the whole table. Under
    tensor parallelism each rank reads only the rows of its own hash columns,
    so table traffic splits across ranks.
    """
    layout = EngramLayout.from_config(model_config)
    if layout is None:
        return [[] for _ in range(n_dev)]
    if n_dev == 1 or model_config.data_parallel_degree > 1:
        return [
            [(0, rows) for rows in layout.num_embeddings] for _ in range(n_dev)
        ]
    assert token_map is not None
    lookup = EngramLookup(layout, token_map)
    return [
        [
            lookup.shard_rows(index, rank, n_dev)
            for index in range(len(layout.layer_ids))
        ]
        for rank in range(n_dev)
    ]


def _row_view(buffer: Buffer, start: int, rows: int) -> Buffer:
    """Rows ``[start, start + rows)`` of a pinned table, without a copy."""
    if start == 0 and rows == buffer.shape[0]:
        return buffer
    return buffer[start : start + rows, :]


def _engram_codes(
    state_dict: dict[str, Any], model_config: DeepseekV41Config, *, pop: bool
) -> dict[str, np.ndarray]:
    """Every Engram table's checkpoint bytes, in one fixed order.

    The order fixes the shared host copy's layout, so every process derives
    the same key and offsets from the same checkpoint.
    Views only: nothing is read from the checkpoint here.
    """
    codes: dict[str, np.ndarray] = {}
    for layer_id in model_config.engram_layer_ids:
        for suffix, _ in _ENGRAM_TABLE_DTYPES:
            name = _engram_prefix(layer_id) + suffix
            data = state_dict.pop(name) if pop else state_dict[name]
            assert isinstance(data, WeightData)
            codes[name] = weight_adapters.as_uint8(data.data)
    return codes


def _shared_engram_copy_exists(
    state_dict: dict[str, Any], model_config: DeepseekV41Config
) -> bool:
    """Whether another process already shares these Engram tables here."""
    if not engram_shared_host.enabled():
        return False
    try:
        codes = _engram_codes(state_dict, model_config, pop=False)
        key = engram_shared_host.source_key(codes)
        return key is not None and engram_shared_host.complete_copy_exists(
            key, {name: array.shape for name, array in codes.items()}
        )
    except Exception:
        logger.debug("Shared Engram copy probe failed", exc_info=True)
        return False


def _open_shared_engram_tables(
    codes: dict[str, np.ndarray], pin_device: Accelerator
) -> engram_shared_host.SharedHostTables | None:
    """This host's shared, pinned copy of ``codes``, or ``None``.

    Creates it (copying ``codes`` in, :func:`_copy_streamed`) if no live
    process shares a complete one yet, else attaches to it.
    """
    if not engram_shared_host.enabled() or is_virtual_device_mode():
        return None
    key = engram_shared_host.source_key(codes)
    if key is None:
        logger.info(
            "Engram tables are not mapped from a checkpoint file; pinning a "
            "private copy."
        )
        return None

    def fill(arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        for name, code in codes.items():
            _copy_streamed(code, arrays[name])

    start = time.perf_counter()
    try:
        registrar = engram_shared_host.CudaHostRegistrar(pin_device.id)
        shared = engram_shared_host.open_shared_tables(
            key,
            {name: array.shape for name, array in codes.items()},
            fill,
            register=registrar,
            read_only_attach=registrar.read_only,
        )
    except (engram_shared_host.SharedTablesUnavailable, OSError) as e:
        logger.warning(
            "Shared Engram host tables unavailable (%s); pinning a private "
            "copy.",
            e,
        )
        return None
    logger.info(
        "%s this host's shared Engram tables (%s, key %s) in %.1fs.",
        "Created" if shared.created else "Attached to",
        to_human_readable_bytes(shared.size),
        key,
        time.perf_counter() - start,
    )
    return shared


# Each chunk is paged out once copied, bounding the transient page cache.
_PIN_CHUNK_BYTES = 64 << 20
_PIN_THREADS = 16


def _host_view(buffer: Buffer) -> np.ndarray:
    """A writable uint8 numpy view of a host-accessible (pinned) ``buffer``.

    Built from its address so writes through it need neither a DLPack
    export (which may come back read-only) nor a ``Buffer`` copy, both of
    which would hold the GIL for the whole memcpy.
    """
    nbytes = buffer.num_elements * buffer.dtype.size_in_bytes
    raw = (ctypes.c_uint8 * nbytes).from_address(buffer._data_ptr())
    return np.ctypeslib.as_array(raw)


def _pin_streamed(
    codes: np.ndarray, dtype: DType, pin_device: Accelerator
) -> Buffer:
    """``codes``' bytes in a new pinned ``dtype`` buffer, a chunk at a time.

    ``_PIN_THREADS`` threads copy row chunks and page each out of the
    checkpoint mapping. numpy copies release the GIL, so this can overlap
    the graph compile.
    """
    pinned = Buffer(
        shape=codes.shape, dtype=dtype, device=pin_device, usage=Usage.STAGING
    )
    _copy_streamed(codes, _host_view(pinned).reshape(codes.shape))
    return pinned


def _copy_streamed(codes: np.ndarray, destination: np.ndarray) -> None:
    """Copies ``codes`` into ``destination`` in row chunks on many threads.

    Each chunk is paged out of the checkpoint mapping once copied.
    """
    rows = max(1, _PIN_CHUNK_BYTES // max(1, codes[:1].nbytes))

    def copy(start: int) -> None:
        source = codes[start : start + rows]
        np.copyto(destination[start : start + len(source)], source)
        weight_adapters.page_out(source)

    with ThreadPoolExecutor(_PIN_THREADS) as pool:
        list(pool.map(copy, range(0, len(codes), rows)))


DROP_PAGE_CACHE_ENV = "MAX_DROP_CHECKPOINT_PAGE_CACHE"
"""Set to ``0`` to keep the checkpoint in page cache after the load."""


def _drop_checkpoint_page_cache(weights: Weights) -> None:
    """Drops the checkpoint's page cache once every weight is loaded.

    Page cache counts against the container's memory cgroup. Routed experts
    and Engram are paged out as consumed; this drops the rest. Pages still
    mapped by any process stay.
    """
    if os.environ.get(DROP_PAGE_CACHE_ENV, "1") == "0":
        return
    paths = getattr(weights, "_filepaths", None)
    if paths:
        weight_adapters.drop_page_cache(paths)


def _build_engram_token_map(pipeline_config: PipelineConfig) -> np.ndarray:
    """Loads this checkpoint's tokenizer and derives Engram's token map.

    The tokenizer comes from the configured model path and revision,
    honoring ``trust_remote_code``.
    """
    model = pipeline_config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_path,
        revision=model.huggingface_model_revision,
        trust_remote_code=model.trust_remote_code,
    )
    return compressed_token_map_array(tokenizer)


def vision_config(
    pipeline_config: PipelineConfig, huggingface_config: Any
) -> DeepseekV41VisionConfig | None:
    """``None`` without a tower, or with images capped at 0."""
    config = getattr(huggingface_config, "vision_config", None)
    if config is None or max_images_per_request(pipeline_config) == 0:
        return None
    return DeepseekV41VisionConfig.from_hf(
        config.to_dict(),
        DeepseekV41Config.get_text_config(huggingface_config).hidden_size,
    )


def _vision_tower(
    pipeline_config: PipelineConfig,
    huggingface_config: Any,
    weights: Weights,
    device: Device,
) -> (
    tuple[
        DeepseekV41VisionModel, dict[str, WeightData], DeepseekV41VisionConfig
    ]
    | None
):
    """Builds the vision model on ``device``, with its checkpoint weights.

    ``None`` when :func:`vision_config` is.

    Raises:
        ValueError: If the checkpoint's vision tensors do not load strictly
            into the model.
    """
    config = vision_config(pipeline_config, huggingface_config)
    if config is None:
        return None
    with F.lazy(), default_dtype(DType.bfloat16):
        vision = DeepseekV41VisionModel(config)
        vision.to(device)
    state = weight_adapters.convert_safetensor_vision_state_dict(
        dict(weights.items())
    )
    problems = weight_adapters.vision_load_mismatches(
        state, {n: (t.shape, t.dtype) for n, t in vision.parameters}
    )
    if problems:
        raise ValueError(
            "the checkpoint's vision tensors do not load into the vision "
            "model:\n" + "\n".join(problems)
        )
    logger.info(
        "Loading %d vision tensors: 0 missing, 0 unexpected", len(state)
    )
    return vision, state, config


@dataclass(frozen=True)
class EngramPlan:
    """The pinned Engram tables a graph takes as inputs, before pinning."""

    input_types: list[TensorType]
    """One CPU input pair per (device, Engram layer), device-major."""
    token_map: np.ndarray | None
    """The Engram token map; ``None`` without Engram layers."""
    pin: Callable[[], list[Buffer]]
    """Pins the tables and returns the buffers of :attr:`input_types`. Its
    copies release the GIL, so it can overlap the compile."""


def plan_engram(
    state_dict: dict[str, Any],
    model_config: DeepseekV41Config,
    pipeline_config: PipelineConfig,
    devices: Sequence[Device],
) -> EngramPlan:
    """Pops the Engram tables from ``state_dict`` and plans their inputs.

    With several devices each reads only its rank's rows, viewing the same
    pinned buffers.
    """
    if not is_virtual_device_mode():
        _check_engram_host_ram_capacity(
            state_dict, model_config, num_devices=len(devices)
        )
    engram_layer_ids = model_config.engram_layer_ids
    pin_device = devices[0]
    # Popped now so the strict `load_state_dict` never sees them.
    engram_sources: dict[str, Any] = {
        name: state_dict.pop(name)
        for layer_id in engram_layer_ids
        for name in (
            f"trunk.layers.{layer_id}.engram_fn.embed.weight",
            f"trunk.layers.{layer_id}.engram_fn.embed.scale",
        )
    }
    token_map = (
        _build_engram_token_map(pipeline_config) if engram_layer_ids else None
    )
    n_dev = len(devices)
    row_ranges = _engram_row_ranges(model_config, token_map, n_dev)
    input_types: list[TensorType] = []
    for rank in range(n_dev):
        for index, layer_id in enumerate(engram_layer_ids):
            prefix = _engram_prefix(layer_id)
            _, rows = row_ranges[rank][index]
            for suffix, dtype in _ENGRAM_TABLE_DTYPES:
                shape = list(engram_sources[prefix + suffix].shape)
                input_types.append(
                    TensorType(
                        dtype, [rows, *shape[1:]], device=DeviceRef.CPU()
                    )
                )

    def pin() -> list[Buffer]:
        if is_virtual_device_mode():
            return []
        assert isinstance(pin_device, Accelerator)
        tables = _load_engram_host_tables(
            engram_sources, model_config, pin_device
        )
        return [
            _row_view(buffer, *row_ranges[rank][index])
            for rank in range(n_dev)
            for index, layer_id in enumerate(engram_layer_ids)
            for buffer in tables[layer_id]
        ]

    return EngramPlan(input_types, token_map, pin)


def engram_tables_per_device(
    inputs: Sequence[Value[Any]],
    engram_layer_ids: Sequence[int],
    num_devices: int,
) -> list[dict[int, tuple[TensorValue, TensorValue]]]:
    """Splits the Engram graph inputs into one table dict per device.

    Each dict maps an Engram layer id to its ``(weight, scale)`` inputs.
    """
    tables = iter(inputs)
    return [
        {
            layer_id: (next(tables).tensor, next(tables).tensor)
            for layer_id in engram_layer_ids
        }
        for _ in range(num_devices)
    ]


def engram_lookback_ids_for_contexts(
    contexts: Sequence[TextContext], depth: int
) -> np.ndarray:
    """``[len(contexts), depth]`` int64 Engram lookback window, host side.

    Each row holds the ``depth`` tokens before the request's
    ``processed_length``, so chunked-prefill continuations and decode steps
    see their real predecessors.
    """
    return engram_lookback_token_ids(
        [ctx.tokens.all for ctx in contexts],
        [ctx.tokens.processed_length for ctx in contexts],
        depth,
    )


@runtime_checkable
class EngramTableInputs(Protocol):
    """Model inputs that carry the pinned host Engram tables."""

    engram_host_buffers: Sequence[Buffer]
    """The pinned host Engram tables, bound as graph inputs."""


@dataclass
class DeepseekV41Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4.1-Flash model execution."""

    tokens: Buffer
    """Input token IDs, ragged."""

    input_row_offsets: Buffer
    """Offsets of each sequence in the ragged ``tokens``."""

    return_n_logits: Buffer
    """Number of trailing logits to return."""

    signal_buffers: list[Buffer] = field(default_factory=list)
    """One per device when there is more than one, else empty."""

    engram_lookback_ids: Buffer | None = None
    """Int64 ``[batch, max_ngram_size - 1]`` token ids preceding each chunk.

    On device 0; ``None`` when the config has no Engram layers.
    """

    host_input_row_offsets: Buffer | None = None
    """CPU copy of :attr:`input_row_offsets`; only under data parallelism."""

    data_parallel_splits: Buffer | None = None
    """Int64 ``[replicas + 1]`` request boundaries of each replica, on CPU.

    Only under data parallelism, where the batch is replica-major.
    """

    ep_buffers: Sequence[Buffer] = ()
    """The routed experts' expert-parallel communication buffers; only
    under data parallelism."""

    engram_host_buffers: Sequence[Buffer] = ()
    """Pinned Engram tables, one (weight, scale) pair per device and layer.

    :attr:`buffers` places them last, so it is the full input list for
    device-graph capture and replay.
    """

    engram_lookback_source: Buffer | None = None
    """Int64 ``[batch]`` row of :attr:`engram_lookback_history` to read each
    request's window from, or -1 for :attr:`engram_lookback_ids`.

    Only under speculative decoding with Engram layers, as is the history.
    """

    engram_lookback_history: Buffer | None = None
    """``[rows, max_ngram_size - 1]`` device-0 windows the previous step
    recorded for this one, one row per row of the previous batch.

    Rows follow the replica-major batch, as the lookback window does, so one
    buffer serves every data-parallel replica.
    """

    merged_rows: Buffer | None = None
    """A host int8 placeholder as long as the step's prompt tokens plus all
    its drafts, a dimension the speculative graph declares; never read.

    Only under speculative decoding.
    """

    batch_context_lengths: list[Buffer] = field(default_factory=list)
    """Per replica, its summed context length, int32 ``[1]`` on CPU.

    Only under speculative decoding with data parallelism.
    """

    replay_plan: ReplayPlan | None = None
    """The rows the replay graph keeps; ``None`` runs the full graph.

    Only under speculative decoding with replay enabled.
    """

    replay_buffers: tuple[Buffer, ...] = ()
    """The replay graph's extra inputs, which the full graph does not take:
    kept indices, offsets and skips, the longest kept row per replica and,
    under data parallelism, the host kept offsets."""

    replay_write_batch: DSparkWriteBatch | None = None
    """The KV writes this batch's graph will cover, published once it is
    submitted. Only under speculative decoding with replay enabled."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The graph inputs; the vision ones are empty for a text-only model."""
        assert self.kv_cache_inputs is not None
        lookback = (
            (self.engram_lookback_ids,)
            if self.engram_lookback_ids is not None
            else ()
        )
        data_parallel = (
            (self.host_input_row_offsets, self.data_parallel_splits)
            if self.host_input_row_offsets is not None
            and self.data_parallel_splits is not None
            else ()
        )
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *lookback,
            *data_parallel,
            *self.ep_buffers,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *self.engram_host_buffers,
        )


class DeepseekV41BatchProcessor(
    SingleReplicaRaggedBatchProcessor[
        TextContext, DeepseekV41Inputs, DeepseekV41Config
    ]
):
    """Batches requests into ragged inputs for the model's replicas.

    Inputs follow the standard ragged KV order. With Engram layers,
    ``engram_lookback_ids`` is inserted right after ``return_n_logits``.
    Under data parallelism the replicas' requests are staged replica-major
    into one batch, and ``host_input_row_offsets`` and
    ``data_parallel_splits`` follow (on CPU), then the expert-parallel
    communication buffers.

    It encodes a batch's images for the pipeline's vision cache, and under
    speculative decoding it prepares every batch of the DSpark pairing,
    which also reads each request's Engram lookback from the history the
    previous step recorded.
    """

    _ep_buffers: tuple[Buffer, ...] = ()
    _vision_model: Model | None = None
    _empty_vision_embeddings: list[Buffer] | None = None
    _served_kv_params: KVCacheParamInterface | None = None
    _submitted_replay_writes: DSparkWriteBatch | None = None
    _previous_replay_writes: tuple[DSparkWriteWitness, ...] = ()

    def __init__(
        self, config: DeepseekV41Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._lookback = EngramLookbackBatchState()

    @functools.cached_property
    def replay_min_tokens(self) -> int:
        """The replay threshold under speculative decoding; 0 when off."""
        if requested_min_tokens() == 0:
            return 0
        threshold, disabled = replay_min_tokens(
            self.config,
            self.runtime.pipeline_config.runtime.device_graph_capture,
        )
        if disabled is not None:
            logger.warning(
                "DSV41 decoder replay requested but disabled: %s", disabled
            )
        return threshold

    def bind_kv_params(self, kv_params: KVCacheParamInterface) -> None:
        """Binds the served KV tree, whose leaf ids the block manager keys on.

        The target's own params name the window leaf ``swa.*``, but a
        speculative pairing registers it under its child, ``target.swa.*``.
        """
        self._served_kv_params = kv_params

    @functools.cached_property
    def _replay_leaf_id(self) -> str:
        served = self._served_kv_params
        return sliding_window_leaf_id(
            self.config.kv_params if served is None else served
        )

    def bind_ep_buffers(self, buffers: Sequence[Buffer]) -> None:
        """Binds the expert-parallel buffers created at model load."""
        self._ep_buffers = tuple(buffers)

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Binds the buffers of the runtime a speculative pairing started."""
        if initializer is not None:
            self.bind_ep_buffers(initializer.model_inputs())

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession
    ) -> None:
        """Wires the compiled vision model."""
        del session
        self._vision_model = vision_model

    @functools.cached_property
    def _vision_config(self) -> DeepseekV41VisionConfig | None:
        pipeline_config = self.runtime.pipeline_config
        return vision_config(
            pipeline_config, pipeline_config.model.huggingface_config
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the pipeline-selected uncached images to device 0."""
        if self._vision_config is None:
            return None
        return pack_uncached_images(selection, devices[0], self._vision_config)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Encodes the packed images into their span rows on every device."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        if self._vision_model is None:
            raise ValueError("this checkpoint has no vision model")
        (rows,) = self._vision_model.execute(*packed)
        assert isinstance(rows, Buffer)
        return VisionEncodeResult(embeddings=[rows.to(d) for d in devices])

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row span embeddings, cached across steps."""
        if self._empty_vision_embeddings is None:
            self._empty_vision_embeddings = create_empty_image_embeddings(
                devices, self.config.hidden_size
            )
        return self._empty_vision_embeddings

    @property
    def _engram_lookback_depth(self) -> int:
        """``max_ngram_size - 1``, or 0 when the config has no Engram."""
        if not self.config.engram_layer_ids:
            return 0
        return self.config.engram_max_ngram_size - 1

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[Any]:
        """The symbolic inputs the batch processor builds."""
        # Signal buffers only with more than one device.
        inputs: list[Any] = ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=len(device_refs) > 1,
        )
        depth = self._engram_lookback_depth
        # Right after `return_n_logits`; must match the order of
        # `DeepseekV41Inputs.buffers`.
        extra: list[TensorType | BufferType] = []
        if depth:
            extra.append(
                TensorType(
                    DType.int64,
                    shape=["engram_lookback_batch", depth],
                    device=device_refs[0],
                )
            )
        dp = self.config.data_parallel_degree
        if dp > 1:
            extra.extend(
                tree.leaves(
                    ragged_batch_inputs(
                        device_refs[0],
                        ("host_row_offsets", "data_parallel_splits"),
                        data_parallel_degree=dp,
                    ),
                    leaf=TensorType,
                )
            )
        if self.config.ep_config is not None:
            extra.extend(EPBatchManager(self.config.ep_config).input_types())
        inputs[3:3] = extra
        return inputs

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV41Inputs:
        """Builds the first token inputs from the request batch."""
        self._submitted_replay_writes = None
        dp = self.config.data_parallel_degree
        if dp > 1:
            if len(replica_batches) != dp:
                raise ValueError(
                    f"expected {dp} replica batches, got {len(replica_batches)}"
                )
            context_batch = [ctx for batch in replica_batches for ctx in batch]
        else:
            context_batch = list(
                single_replica_context_batch(
                    replica_batches, processor_name=type(self).__qualname__
                )
            )
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            context_batch
        )
        device0 = self.runtime.devices[0]
        depth = self._engram_lookback_depth
        engram_lookback_ids = None
        if depth:
            engram_lookback_ids = Buffer.from_numpy(
                engram_lookback_ids_for_contexts(context_batch, depth)
            ).to(device0)
        inputs = DeepseekV41Inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            signal_buffers=list(self.runtime.signal_buffers),
            engram_lookback_ids=engram_lookback_ids,
            host_input_row_offsets=(
                Buffer.from_numpy(offsets_np) if dp > 1 else None
            ),
            data_parallel_splits=(
                Buffer.from_numpy(compute_data_parallel_splits(replica_batches))
                if dp > 1
                else None
            ),
            ep_buffers=self._ep_buffers,
            engram_host_buffers=self.config.engram_host_buffers,
        )
        if self.runtime.pipeline_config.speculative is not None:
            self._prepare_spec_decode(
                inputs, replica_batches, context_batch, return_n_logits
            )
        return inputs

    def _prepare_spec_decode(
        self,
        inputs: DeepseekV41Inputs,
        replica_batches: Sequence[Sequence[TextContext]],
        contexts: Sequence[TextContext],
        return_n_logits: int,
    ) -> None:
        """Adds what the DSpark pairing's graph takes beyond the model's.

        ``return_n_logits`` counts the drafts each request verifies, and the
        sampled token.
        """
        drafts = len(contexts) * max(return_n_logits - 1, 0)
        inputs.merged_rows = Buffer(
            shape=(inputs.tokens.shape[0] + drafts,),
            dtype=DType.int8,
            device=CPU(),
        )
        if len(replica_batches) > 1:
            inputs.batch_context_lengths = [
                Buffer.from_numpy(
                    np.array(
                        [sum(ctx.tokens.current_position for ctx in batch)],
                        dtype=np.int32,
                    )
                )
                for batch in replica_batches
            ]
        self._prepare_replay(inputs, replica_batches, contexts, return_n_logits)
        depth = self._engram_lookback_depth
        if not depth:
            return
        device0 = self.runtime.devices[0]
        inputs.engram_lookback_history = self._lookback.ensure_history(
            capacity=self.runtime.max_global_batch_size,
            batch=len(contexts),
            depth=depth,
            device=device0,
        )
        inputs.engram_lookback_source = Buffer.from_numpy(
            self._lookback.source(contexts)
        ).to(device0)

    def _prepare_replay(
        self,
        inputs: DeepseekV41Inputs,
        replica_batches: Sequence[Sequence[TextContext]],
        contexts: Sequence[TextContext],
        return_n_logits: int,
    ) -> None:
        """Plans the batch's replay rows and snapshots its KV writes.

        The plan's merged rows are each request's prompt rows followed by
        its ``return_n_logits - 1`` verify draft rows. The write snapshot is
        taken whenever replay is enabled, trimmed or not, so the pipeline can
        publish what this batch's graph writes.
        """
        min_tokens = self.replay_min_tokens
        if not min_tokens:
            self._previous_replay_writes = ()
            return
        verify_width = max(0, return_n_logits - 1)
        plan, skip_reason = build_replay_plan_with_reason(
            contexts,
            window_size=self.config.sliding_window,
            min_tokens=min_tokens,
            verify_width=verify_width,
            return_hidden_states=self.runtime.return_hidden_states
            != ReturnHiddenStates.NONE,
            enable_echo=self.runtime.pipeline_config.model.enable_echo,
        )
        if skip_reason is not None:
            log_replay_not_built(logger, skip_reason, contexts)
        inputs.replay_plan = plan
        inputs.replay_write_batch = DSparkWriteBatch.capture(
            contexts,
            plan,
            leaf_id=self._replay_leaf_id,
            verified_width=verify_width,
            previous=self._previous_replay_writes,
        )
        if plan is None:
            return
        device0 = self.runtime.devices[0]
        dp = len(replica_batches) > 1
        boundaries = (
            compute_data_parallel_splits(replica_batches)
            if dp
            else [0, len(contexts)]
        )
        inputs.replay_buffers = (
            *(
                Buffer.from_numpy(array).to(device0)
                for array in (
                    plan.kept_indices,
                    plan.kept_offsets,
                    plan.kept_skip,
                )
            ),
            Buffer.from_numpy(plan.kept_longest(boundaries)),
            *((Buffer.from_numpy(plan.kept_offsets),) if dp else ()),
        )
        logger.debug(
            "DSV41 decoder replay plan: kept_indices=%s kept_offsets=%s "
            "kept_skip=%s boundaries=%s",
            plan.kept_indices,
            plan.kept_offsets,
            plan.kept_skip,
            boundaries,
        )

    def settle_replay(
        self, inputs: DeepseekV41Inputs, draft_width: int | None
    ) -> bool:
        """Decides at execution whether the replay graph runs this batch.

        Write coverage is claimed only when the draft width is the one the
        snapshot was taken for. A plan that cannot claim coverage is dropped
        and the full graph runs. A plan built for another width than the
        batch's drafts is a hard error: its kept rows address a different
        merged row space.

        Returns:
            Whether to run the replay graph.
        """
        self._submitted_replay_writes = None
        write_batch = inputs.replay_write_batch
        if (
            write_batch is not None
            and write_batch.verified_width != draft_width
        ):
            inputs.replay_write_batch = None
        plan = inputs.replay_plan
        if plan is None:
            return False
        if inputs.replay_write_batch is None:
            logger.debug(
                "DSV41 decoder replay plan dropped: draft_width=%s "
                "plan_verify_width=%d",
                draft_width,
                plan.verify_width,
            )
            inputs.replay_plan = None
            inputs.replay_buffers = ()
            return False
        if plan.verify_width != draft_width:
            raise RuntimeError(
                f"Replay plan verify width {plan.verify_width} differs "
                f"from the batch's draft width {draft_width}"
            )
        logger.debug(
            "DSV41 decoder replay engaged: kept_rows=%d total_rows=%d "
            "requests=%d",
            len(plan.kept_indices),
            inputs.tokens.shape[0],
            len(plan.kept_skip),
        )
        return True

    def record_submission(self, inputs: DeepseekV41Inputs) -> None:
        """Records the writes of the graph that just returned normally."""
        self._submitted_replay_writes = inputs.replay_write_batch

    def begin_captured_submission(self, inputs: DeepseekV41Inputs) -> None:
        """Clears the submitted writes before a captured replay refreshes."""
        self._submitted_replay_writes = None
        if inputs.replay_plan is not None or inputs.replay_buffers:
            raise RuntimeError(
                "Captured DSpark submissions require FULL inputs"
            )

    def on_captured_submission_success(
        self, inputs: DeepseekV41Inputs, draft_width: int | None
    ) -> None:
        """Records the writes a captured submission covers."""
        if inputs.replay_plan is not None or inputs.replay_buffers:
            raise RuntimeError(
                "Captured DSpark submissions require FULL inputs"
            )
        write_batch = inputs.replay_write_batch
        if write_batch is None:
            return
        if write_batch.verified_width != draft_width:
            raise RuntimeError(
                "Captured DSpark verify width differs from its snapshot"
            )
        self._submitted_replay_writes = write_batch

    def finalize_kv_cache_writes(self, contexts: Sequence[TextContext]) -> None:
        """Publishes the submitted writes ahead of the pipeline's cache step."""
        submitted = self._submitted_replay_writes
        self._submitted_replay_writes = None
        self._previous_replay_writes = (
            submitted.publish(contexts) if submitted is not None else ()
        )


class _GraphBody(Module):
    """Traces ``build``, which emits a graph's body from its inputs."""

    def __init__(self, build: Callable[..., DecoderOutputs]) -> None:
        super().__init__()
        self._build = build

    def __call__(self, *inputs: Value[Any]) -> DecoderOutputs:
        return self._build(*inputs)


class DeepseekV41Model(
    LogProbabilitiesMixin,
    PipelineModelWithKVCache[TextContext, DeepseekV41Config, Model],
):
    """A DeepSeek-V4.1-Flash pipeline model for text and image generation.

    :class:`LogProbabilitiesMixin` computes logprobs from the graph's
    float32 logits. As a
    :class:`~max.pipelines.lib.vision_encoder_cache.SupportsVisionEncoding`
    model, the pipeline's vision encoder cache drives :meth:`vision_execute`
    and hands the language graph each step's span rows and their token
    rows.
    """

    model_config_cls: ClassVar[type[Any]] = DeepseekV41Config
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DeepseekV41BatchProcessor
    )

    model: Model
    """The compiled and initialized MAX Engine model."""

    vision_model: CompiledCallable[..., Any] | None
    """The compiled vision model; ``None`` for a checkpoint without one."""

    _engram_host_buffers: list[Buffer]
    """Pinned Engram tables, bound as the last graph inputs.

    Ordered device-major, then by Engram layer id, as the inputs are
    declared.
    """

    _strict_state_dict_loading = True

    _pending_weight_loads: list[tuple[Callable[[], None], bool]]
    """Weight loads queued by :meth:`_defer_weight_load`."""

    _running_weight_loads: list[tuple[Future[None], bool]]
    """Weight loads :meth:`_start_weight_loads` started."""

    _weights_dict: dict[str, Any]
    """The checkpoint weights :meth:`graph_specs` builds the decoder from."""

    _stream_routed_experts: ClassVar[bool] = True
    """Whether routed experts are uploaded straight to their devices.

    When on, :meth:`_load_state_dict` leaves them deferred and the graph
    build makes them device-resident. A subclass whose graph build does not
    make them device-resident must turn this off.
    """

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
        max_batch_size: int = 1,
    ) -> None:
        # Reject a plan clamped below one prefill chunk before loading.
        runtime = pipeline_config.runtime
        requested = pipeline_config.model.max_length
        if requested is not None:
            check_planned_max_length(
                planned=memory_plan.planned_max_length,
                requested=requested,
                forward_rows=forward_row_bound(
                    runtime.max_num_input_tokens,
                    runtime.max_batch_input_tokens,
                    requested,
                ),
            )
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)
        self.vision_model = self._load_vision_model(session)
        if self.vision_model is not None:
            self._batching.bind_vision_encoder(
                vision_model=self.vision_model.engine_model, session=session
            )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV41, DeepseekV41Config]:
        """Builds the trunk for a speculative pairing.

        The DSpark stages the adapter drops are left out.

        A caller that sized the cache passes its KV params. The trunk is built
        without the Engram tables it reads, which the pairing's graph takes as
        inputs (:attr:`~.DeepseekV41Config.engram_input_types`) and its verify
        binds. Pinning them, and streaming the routed experts straight to
        their devices, are weight loads deferred to the compile.

        Raises:
            ValueError: If ``target_layer_ids`` names other layers than the
                DSpark stages read.
        """
        config = DeepseekV41Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        layers = list(context.target_layer_ids)
        if layers and layers != config.dspark_target_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 collects the states at its"
                f" dspark_target_layer_ids {config.dspark_target_layer_ids};"
                f" got target_layer_ids {layers}."
            )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParamInterface)
            config.kv_params = context.kv_params
        config.return_logits = context.return_logits
        config.return_hidden_states = context.return_hidden_states
        context.start_ep(config.ep_config)
        assert context.adapter is not None
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config,
            context.weights,
            functools.partial(context.adapter, defer_routed_experts=True),
        )
        plan = plan_engram(
            state_dict,
            config,
            context.pipeline_config,
            [ref.to_device() for ref in config.devices],
        )
        config.engram_input_types = plan.input_types
        if config.engram_layer_ids:

            def pin(weights_registry: dict[str, Any]) -> None:
                del weights_registry
                config.engram_host_buffers = plan.pin()

            context.defer_weight_load(pin, overlaps_compile=True)
        module = DeepseekV41(
            config, engram_host_tables={}, engram_token_map=plan.token_map
        )
        module.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=cls._strict_state_dict_loading,
        )
        streaming = ExpertStreaming(
            {
                name: value
                for name, value in state_dict.items()
                if isinstance(value, weight_adapters.DeferredExpertStack)
            }
        )
        module.expert_streaming = streaming
        context.defer_weight_load(
            lambda registry: stream_routed_experts(streaming.uploads, registry),
            overlaps_compile=False,
        )
        return module, config

    def _load_vision_model(
        self, session: InferenceSession
    ) -> CompiledCallable[..., Any] | None:
        """Compiles the vision model on device 0 and loads its weights."""
        device = self.devices[0]
        tower = _vision_tower(
            self.pipeline_config, self.huggingface_config, self.weights, device
        )
        if tower is None:
            return None
        vision, weights, config = tower
        return vision.compile(
            *vision_input_types(
                config, DeviceRef.from_device(device), DType.bfloat16
            ),
            weights=weights,
            session=session,
        )

    @classmethod
    def build_vision_graph(
        cls, context: ModelLoadContext
    ) -> tuple[Graph, dict[str, Any]]:
        """Builds the vision model's graph on device 0.

        Raises:
            ValueError: If the checkpoint has no vision model, or serves text
                only.
        """
        spec = context.pipeline_config.model.device_specs[0]
        device_ref = DeviceRef(spec.device_type, spec.id)
        tower = _vision_tower(
            context.pipeline_config,
            context.huggingface_config,
            context.weights,
            device_ref.to_device(),
        )
        if tower is None:
            raise ValueError(
                "This DeepSeek-V4.1 checkpoint serves no images: it has no"
                " vision model, or max_images_per_request is 0."
            )
        vision, weights, config = tower
        graph_spec = GraphSpec(
            vision,
            "deepseekV41_vision",
            vision_input_types(config, device_ref, DType.bfloat16),
            weights=weights,
        )
        return graph_spec.trace(), dict(graph_spec.weights_registry())

    @property
    def _batching(self) -> DeepseekV41BatchProcessor:
        assert isinstance(self._batch_processor, DeepseekV41BatchProcessor)
        return self._batch_processor

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the selected images, as the batch processor does."""
        return self._batching.pack_vision_inputs(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Encodes the selected images, as the batch processor does."""
        return self._batching.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the batch processor's embeddings for a batch with no new images."""
        return self._batching.empty_vision_embeddings(devices)

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV41Config:
        del state_dict
        model_config = self.arch_config_as(DeepseekV41Config)
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states
        return model_config

    def _load_state_dict(self) -> dict[str, Any]:
        """Returns the adapter's state dict, routed experts left deferred.

        With :attr:`_stream_routed_experts`, routed-expert tensors are
        :class:`~.weight_adapters.DeferredExpertStack` values that
        :func:`stream_routed_experts` uploads directly to the devices.
        """
        if self.adapter is None or not self._stream_routed_experts:
            return super()._load_state_dict()
        return parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            functools.partial(self.adapter, defer_routed_experts=True),
            hf_config=self._hf_config_for_weights(),
        )

    def _prepare_engram(
        self, state_dict: dict[str, Any], model_config: DeepseekV41Config
    ) -> tuple[list[TensorType], np.ndarray | None]:
        """Schedules the Engram tables' pinning and declares their graph inputs.

        See :func:`plan_engram`. The pinning is deferred
        (:meth:`_defer_weight_load`) to overlap the compile.

        Returns:
            The Engram graph input types (placed after all other inputs),
            and the Engram token map (``None`` without Engram layers).
        """
        plan = plan_engram(
            state_dict, model_config, self.pipeline_config, self.devices
        )
        self._engram_host_buffers = []

        def pin() -> None:
            self._engram_host_buffers = plan.pin()

        if model_config.engram_layer_ids:
            self._defer_weight_load(pin, overlaps_compile=True)
        return plan.input_types, plan.token_map

    def _defer_weight_load(
        self, load: Callable[[], None], *, overlaps_compile: bool
    ) -> None:
        """Queues host-side weight loading for :meth:`load_model` to run.

        Each load runs on its own thread once the graph is built.
        ``overlaps_compile`` loads keep running during the compile and must
        be GIL-free, or they starve it.
        """
        self._pending_weight_loads.append((load, overlaps_compile))

    ep_comm_initializer: EPCommInitializer | None = None

    def _init_distributed_runtime(
        self, session: InferenceSession, model_config: Any
    ) -> None:
        """Sets up the routed experts' expert-parallel communication.

        Only under data parallelism (``model_config.ep_config``), and not on
        virtual devices, which compile the graph without running it.
        """
        assert isinstance(model_config, DeepseekV41Config)
        ep_config = model_config.ep_config
        if ep_config is None or is_virtual_device_mode():
            return
        initializer = EPCommInitializer(ep_config)
        init = initializer.init_module()
        specs: dict[str, GraphSpec] = {
            GraphRole.EP_INIT: GraphSpec(init, "ep_init", init.input_types())
        }
        modules = compile_and_init_graphs(session, specs)
        initializer.run_init(modules[GraphRole.EP_INIT].engine_model)
        self.ep_comm_initializer = initializer
        # Every V4.1 batch processor takes the buffers as graph inputs.
        bind = getattr(self.batch_processor, "bind_ep_buffers", None)
        if bind is None:
            raise TypeError(
                f"{type(self.batch_processor).__name__} cannot take the "
                "expert-parallel buffers"
            )
        bind(initializer.model_inputs())

    def _start_weight_loads(self) -> None:
        """Starts the deferred weight loads, each on its own thread.

        Returns once the loads that must not overlap the compile are done;
        :meth:`_finish_weight_loads` waits for the rest.
        """
        pending, self._pending_weight_loads = self._pending_weight_loads, []
        pool = ThreadPoolExecutor(max(1, len(pending)))
        self._running_weight_loads = [
            (pool.submit(load), overlaps_compile)
            for load, overlaps_compile in pending
        ]
        pool.shutdown(wait=False)
        for load, overlaps_compile in self._running_weight_loads:
            if not overlaps_compile:
                load.result()

    def _finish_weight_loads(self) -> None:
        """Waits for the weight loads :meth:`_start_weight_loads` started."""
        running, self._running_weight_loads = self._running_weight_loads, []
        for load, _ in running:
            load.result()

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Loads the decoder :meth:`graph_specs` returns."""
        return self.load_graphs(session)[GraphRole.DECODER]

    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns.

        The decoder's trace starts the deferred weight loads, so the compile
        waits for the routed-expert upload and overlaps the Engram pinning,
        which finishes before the host weights can be released.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        self._pending_weight_loads = []
        self._weights_dict = self._load_state_dict()
        model_config = self._create_model_config(self._weights_dict)
        self._init_distributed_runtime(session, model_config)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(model_config)
            timer.mark_build_complete()
            self.compiled_modules = compile_and_init_graphs(session, specs)
            self._finish_weight_loads()
        models = {
            role: compiled.engine_model
            for role, compiled in self.compiled_modules.items()
        }
        self._maybe_release_host_weights(*models.values())
        _drop_checkpoint_page_cache(self.weights)
        self._wire_batch_processor(models[GraphRole.DECODER], model_config)
        return models

    def graph_specs(
        self, model_config: DeepseekV41Config
    ) -> dict[str, GraphSpec]:
        """Returns the decoder graph, tensor or data parallel across devices.

        The model is built as the graph traces, since its Engram tables are
        graph inputs, and the trace ends by starting the deferred weight
        loads (:meth:`_start_weight_loads`).
        """
        state_dict, self._weights_dict = self._weights_dict, {}
        engram_layer_ids = model_config.engram_layer_ids
        engram_input_types, engram_token_map = self._prepare_engram(
            state_dict, model_config
        )

        assert self.batch_processor is not None
        input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params, device_refs=self.device_refs
        )
        hidden_size = model_config.hidden_size
        vision_types = [
            *(
                TensorType(
                    DType.bfloat16, ["vision_rows", hidden_size], device=ref
                )
                for ref in self.device_refs
            ),
            *(
                TensorType(DType.int32, ["vision_scatter_rows"], device=ref)
                for ref in self.device_refs
            ),
        ]
        input_types = [*input_types, *vision_types, *engram_input_types]

        n_dev = len(self.device_refs)
        n_trailing = len(vision_types) + len(engram_input_types)
        weights_registry: dict[str, Any] = {}

        def build(*graph_inputs: Value[Any]) -> DecoderOutputs:
            all_inputs = list(graph_inputs)
            tokens, input_row_offsets, return_n_logits = all_inputs[:3]
            # The Engram lookback window follows `return_n_logits`, then the
            # data-parallel host offsets and splits.
            n_leading = 4 if engram_layer_ids else 3
            engram_lookback_ids = (
                all_inputs[3].tensor if engram_layer_ids else None
            )
            data_parallel = model_config.data_parallel_degree > 1
            if data_parallel:
                host_row_offsets, data_parallel_splits = (
                    v.tensor for v in all_inputs[n_leading : n_leading + 2]
                )
                n_leading += 2
            n_ep = (
                len(EPBatchManager(model_config.ep_config).input_types())
                if model_config.ep_config is not None
                else 0
            )
            ep_inputs = all_inputs[n_leading : n_leading + n_ep]
            n_leading += n_ep
            variadic_args = all_inputs[n_leading:-n_trailing]
            vision_inputs = all_inputs[-n_trailing:][: 2 * n_dev]
            engram_graph_inputs = all_inputs[-n_trailing:][2 * n_dev :]
            visions = [
                VisionMerge(embeddings.tensor, indices.tensor)
                for embeddings, indices in zip(
                    vision_inputs[:n_dev], vision_inputs[n_dev:], strict=True
                )
            ]

            per_device_tables = engram_tables_per_device(
                engram_graph_inputs, engram_layer_ids, n_dev
            )

            nn_model = DeepseekV41(
                model_config,
                engram_host_tables=per_device_tables[0],
                engram_token_map=engram_token_map,
            )
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )
            weights_registry.update(nn_model.state_dict(auto_initialize=False))
            if n_dev == 1:
                serving = [nn_model]
            elif data_parallel:
                serving = nn_model.data_parallel_replicas(
                    self.device_refs, per_device_tables
                )
                if serving[0].ep_batch_manager is not None:
                    serving[0].ep_batch_manager.fetch_buffers(ep_inputs)
            else:
                serving = nn_model.tensor_parallel_replicas(
                    self.device_refs, per_device_tables
                )
            expert_uploads = make_routed_experts_device_resident(
                nn_model, serving, state_dict, self.devices, self.device_refs
            )

            assert isinstance(self.kv_params, MultiKVCacheParams)
            specs = model_config.kv_leaf_specs
            if n_dev == 1:
                groups = self.kv_params.unflatten_basic_kv_tree(
                    iter(variadic_args)
                )
                outputs = nn_model.serve(
                    tokens.tensor,
                    input_row_offsets.tensor,
                    return_n_logits.tensor,
                    csa2_cache(specs, groups, 0),
                    engram_lookback_ids=engram_lookback_ids,
                    vision=visions[0],
                )
            else:
                signal_buffers = [v.buffer for v in variadic_args[:n_dev]]
                groups = self.kv_params.unflatten_basic_kv_tree(
                    iter(variadic_args[n_dev:])
                )
                caches = [csa2_cache(specs, groups, i) for i in range(n_dev)]
                tokens_per_dev = ops.distributed_broadcast(
                    tokens.tensor, signal_buffers
                )
                offsets_per_dev = ops.distributed_broadcast(
                    input_row_offsets.tensor, signal_buffers
                )
                lookback_per_dev = (
                    ops.distributed_broadcast(
                        engram_lookback_ids, signal_buffers
                    )
                    if engram_lookback_ids is not None
                    else None
                )
                if data_parallel:
                    outputs = DeepseekV41.serve_data_parallel(
                        serving,
                        tokens_per_dev,
                        offsets_per_dev,
                        host_row_offsets,
                        data_parallel_splits,
                        return_n_logits.tensor,
                        caches,
                        signal_buffers,
                        engram_lookback_ids=lookback_per_dev,
                        visions=visions,
                    )
                else:
                    outputs = DeepseekV41.serve_tensor_parallel(
                        serving,
                        tokens_per_dev,
                        offsets_per_dev,
                        return_n_logits.tensor,
                        caches,
                        signal_buffers,
                        engram_lookback_ids=lookback_per_dev,
                        visions=visions,
                    )
            if expert_uploads:
                self._defer_weight_load(
                    lambda: stream_routed_experts(
                        expert_uploads, weights_registry
                    ),
                    overlaps_compile=False,
                )
            # The end of the trace is the last step before the compile.
            self._start_weight_loads()
            return outputs

        return {
            GraphRole.DECODER: GraphSpec(
                _GraphBody(build),
                "deepseekV41",
                input_types,
                weights=weights_registry,
            )
        }

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Returns the batch processor's inputs plus the pinned Engram tables.

        Keeps ``inputs.buffers`` the full input list for device-graph
        capture and replay.
        """
        inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        assert isinstance(inputs, EngramTableInputs)
        inputs.engram_host_buffers = self._engram_host_buffers
        return inputs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the model on one batch of inputs."""
        assert isinstance(model_inputs, DeepseekV41Inputs)
        # Inputs built directly by the batch processor carry no tables.
        model_inputs.engram_host_buffers = self._engram_host_buffers
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)
