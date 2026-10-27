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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer, Device, Usage
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3Outputs, ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.support.host_memory import available_host_memory, host_memory_limit
from max.support.human_readable_formatter import to_human_readable_bytes
from max.tensor.sharding import DeviceMesh
from transformers import AutoTokenizer

from .batch_processor import DeepseekV41BatchProcessor
from .deepseekV41 import DeepseekV41
from .layers.engram_token_map import (
    check_map_digest,
    compressed_token_map_array,
)
from .model_config import DeepseekV41Config
from .weight_adapters import AdapterResult, EngramTable

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


class DeepseekV41Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext, DeepseekV41Config],
):
    """Loads the text trunk and binds its shared pinned Engram tables."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV41Config
    batch_processor_cls: ClassVar[type[DeepseekV41BatchProcessor]] = (
        DeepseekV41BatchProcessor
    )
    _engram_tables: list[EngramTable]
    _token_map: np.ndarray

    def _load_state_dict(self) -> dict[str, Any]:
        result = super()._load_state_dict()
        if not isinstance(result, AdapterResult):
            raise ValueError(
                "DeepSeek-V4.1 requires its safetensors weight adapter"
            )
        self._engram_tables = result.engram_tables
        self._checkpoint = result.state_dict
        return result.state_dict

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV41Config:
        del state_dict
        if self.pipeline_config.runtime.ep_size > 1:
            raise ValueError(
                "DeepSeek-V4.1 ModuleV3 currently supports tensor parallelism"
            )
        config = DeepseekV41Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        self._validate_engram_tables(config)
        self._token_map = self._serving_token_map(config)
        return config

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict, model_config
        return DType.bfloat16

    def _instantiate_module(
        self, model_config: DeepseekV41Config
    ) -> DeepseekV41:
        assert isinstance(self.kv_params, MultiKVCacheParams)
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), ("tp",))
        return DeepseekV41(
            model_config,
            self.kv_params,
            mesh,
            text_config=DeepseekV41Config.get_text_config(
                self.huggingface_config
            ),
            token_map=self._token_map,
        )

    def _load_decoder(
        self, session: InferenceSession
    ) -> Callable[..., ModuleV3Outputs]:
        decoder = super()._load_decoder(session)
        self._bind_engram_static_inputs()
        return decoder

    def _bind_engram_static_inputs(self) -> None:
        """Pins the adapter's table views and hands them to the processor.

        Pinned against the pipeline's first device, and once every table
        lives in page-locked memory the adapter's views are dropped: the
        loader never keeps a second copy of a table that reaches tens of
        gigabytes.
        """
        processor = self.batch_processor
        assert isinstance(processor, DeepseekV41BatchProcessor)
        pin_device = self.devices[0]
        static: dict[str, Buffer] = {}
        for ordinal, table in enumerate(self._engram_tables):
            weight_name, scale_name = table.table_names(ordinal)
            static[weight_name] = _pinned_table(
                pin_device,
                weight_name,
                table.rows,
                256,
                table.weight_codes,
                DType.float8_e4m3fn,
            )
            static[scale_name] = _pinned_table(
                pin_device,
                scale_name,
                table.rows,
                8,
                table.scale_codes,
                DType.float8_e8m0fnu,
            )
        # The pinned buffers -- and the processor's own reference to them --
        # are the only copies now: dropping the checkpoint views releases a
        # second full mapping of every table for the model's lifetime.
        self._engram_tables = []
        processor.bind_engram(static)

    def _validate_engram_tables(self, config: DeepseekV41Config) -> None:
        """Checks the adapter's tables against the ABI's own table extents.

        Pure check, no allocation: it runs before the tokenizer is loaded and
        before any table is pinned, so a wrong handoff costs nothing.

        Args:
            config: The frozen interface view, whose ``engram_num_embeddings``
                is the ABI's per-ordinal row count.

        Raises:
            ValueError: If the adapter handed over a different number of tables
                than the ABI declares, or any table's row count differs from
                the ABI's extent for its ordinal.
        """
        want = tuple(int(rows) for rows in config.engram_num_embeddings)
        tables = self._engram_tables
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

    def _serving_token_map(self, config: DeepseekV41Config) -> np.ndarray:
        """The serving tokenizer's compressed-id map, digest-checked.

        N-grams hash compressed ids, so a wrong map silently hashes
        unrelated tokens alike -- the placeholder it replaces was all zeros.
        The map is built from the checkpoint's own tokenizer at the
        configured model revision -- the same revision the pipeline
        resolves the config, weights and tokenizer files at, so serving
        from a hub repo id works without any pre-seeded snapshot -- and
        checked against the recorded sha256, so a ``tokenizers`` Unicode
        bump or a swapped checkpoint fails here instead of serving
        meaningless Engram output. The model constructor bakes the array
        into the graph as a constant: no forward input carries it.

        Args:
            config: The frozen interface view, whose ``vocab_size`` sizes
                the map.

        Returns:
            The map as int32 ``[vocab_size]``, in tokenizer id order.

        Raises:
            ValueError: If the tokenizer's vocabulary is not the model's
                size, or the map's sha256 is not the recorded digest.
        """
        model = self.pipeline_config.model
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
