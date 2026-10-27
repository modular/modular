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
"""The DP4/EP4 memory plan must stay admissible at the export's flags.

Two consumers read these estimates, and the DP and EP corrections split
between them. The export-time plan
(``memory_estimation.plan_from_sizes``) charges its statics against
``device_memory_utilization`` of free memory read before anything
loads: the checkpoint counts once there (the engine's own KV sizing
reads device free with the weights already resident, so the DP
replicas are runtime-owned bytes), and the EP dispatch's comm buffers
and receive pool stay out of it because the engine books them itself,
through the sidecar. The engine-facing sidecar
(:meth:`DeepseekV41MemoryPlanner.forward_scratch_bytes_per_device`) is
the exception: the engine sizes and allocates the unified KV arena
before EP init allocates the dispatch heaps, so the sidecar keeps
holding the EP reserve back or an unpinned boot sizes the KV pool into
its room. The tests pin both figures, so neither the export refusal
nor the engine reserve can regress silently.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41.memory_planner import (
    _SCALE_TILE_BYTES,
    _SCALE_TILE_ROWS,
    _SCALE_TILE_SCALES,
    _mm_pages,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.memory_planner import (
    DeepseekV41MemoryPlanner,
    engram_table_bytes,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    build_ep_config,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.kv_cache import kv_cache_memory_size
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.memory_estimation import MemoryEstimator
from runtime_config import text_config

_HF_CONFIG = {"text_config": vars(text_config())}

_GIB = 1024**3
# The frozen export geometry: 4x B200, utilization 0.55, page size 128.
_UTILIZATION = 0.55
_PAGE_SIZE = 128


def _devices(free: int, count: int = 4) -> list[Device]:
    """A fake device list with ``free`` bytes free per device."""
    devices: list[Device] = []
    for _ in range(count):
        device = MagicMock()
        device.stats = {"free_memory": free, "total_memory": free}
        device.max_single_alloc_size = free
        devices.append(device)
    return devices


def _arch_config() -> DeepseekV41Config:
    """The arch config the runtime builds over the synthetic text config."""
    kv = build_kv_params(
        V41Config.from_hf(_HF_CONFIG),
        KVCacheConfig(kv_cache_page_size=_PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text_config()),
    )


def _planner() -> DeepseekV41MemoryPlanner:
    """The planner over the synthetic text config, as the runtime builds it."""
    return DeepseekV41MemoryPlanner(_arch_config())


def _pipeline_config(
    dp: int,
    ep_size: int,
    n_devices: int,
    *,
    weights_size: int = 0,
    max_input_tokens: int | None = 8,
    paths: tuple[Path, ...] = (),
) -> Mock:
    """The frozen export's shape: ``dp == ep_size == the device count``."""
    config = Mock()
    config.model.data_parallel_degree = dp
    config.model.device_specs = [Mock() for _ in range(n_devices)]
    config.model.max_length = 256
    config.model.max_length_is_user_provided = True
    config.model.resolved_weight_paths = Mock(return_value=list(paths))
    config.model.weights_size = Mock(return_value=weights_size)
    config.model.huggingface_config = _HF_CONFIG
    config.model.kv_cache = KVCacheConfig(
        kv_cache_page_size=_PAGE_SIZE,
        device_memory_utilization=_UTILIZATION,
    )
    config.model.model_name = "deepseek-v41-flash"
    config.runtime.ep_size = ep_size
    config.runtime.max_batch_input_tokens = 512
    config.runtime.max_num_input_tokens = max_input_tokens
    config.runtime.max_batch_total_tokens = None
    config.runtime.is_disaggregated = False
    config.runtime.vision_cache_utilization = 0
    config.estimate_signal_buffer_memory = Mock(return_value=0)
    return config


def _expected_ep_reserve(dp4_config: Mock) -> tuple[int, int]:
    """(comm buffers, receive pool) the dispatch allocates per device.

    The independent statement of the EP sizing the sidecar must carry:
    the comm buffers are whatever ``EPConfig.estimate_memory_usage``
    sizes for the config the model itself builds, and the pool is the
    fused chain's receive-side tensors -- the dispatched rows packed
    NVFP4 with their block scales and source map, the packed down input
    the fused gate/up+SwiGLU kernel writes (its epilogue runs the
    activation and the re-quant, so no bf16 intermediate exists), and
    the down projection's bf16 output. A phantom term (or a dropped
    one) moves this hand computation and fails the tests that use it.
    """
    text = text_config()
    ep_config = build_ep_config(dp4_config, text, n_devices=4)
    rows = int(ep_config.get_max_recv_tokens())
    hidden = int(text.hidden_size)
    inter = int(text.moe_intermediate_size)
    n_local = int(text.n_routed_experts) // 4
    tile_rows = -(-(rows + n_local * _SCALE_TILE_ROWS) // _SCALE_TILE_ROWS)

    def scale_bytes(width: int) -> int:
        tile_cols = -(-width // (16 * _SCALE_TILE_SCALES))
        return tile_rows * tile_cols * _SCALE_TILE_BYTES

    dispatched = rows * (hidden // 2)  # NVFP4 packs two fp4 per byte
    source_map = rows * 8  # one [rows, 2] int32 entry per row
    packed_down_input = rows * (inter // 2)
    down_output = rows * hidden * 2
    return int(ep_config.estimate_memory_usage()), _mm_pages(
        dispatched
        + scale_bytes(hidden)
        + source_map
        + packed_down_input
        + scale_bytes(inter)
        + down_output
    )


def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """Writes a safetensors file whose headers weight sizing consumes.

    Each value declares the tensor's BYTE extent (not its element count);
    the dtype string is decorative, since the sizing pass reads only
    ``data_offsets``.
    """
    header: dict[str, Any] = {}
    payload = b""
    for name, (dtype, shape) in tensors.items():
        nbytes = int(np.prod(shape)) if shape else 1
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + nbytes],
        }
        payload += bytes(nbytes)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    return path


def _checkpoint(tmp_path: Path) -> tuple[Path, int]:
    """A checkpoint with one tensor per sizing category, plus its byte total.

    The routed expert, the vocab-parallel embedding and head and the
    skipped DSpark subtree all contribute no extra copy under data
    parallelism; the ``wo_a`` pair lands as its expanded bf16 leaf.
    """
    checkpoint = _safetensors(
        tmp_path / "weights.safetensors",
        {
            "layers.0.attn.wo_a.weight": ("F8_E4M3", [32, 64]),
            "layers.0.attn.wo_a.scale": ("F8_E8M0", [1, 2]),
            "layers.0.ffn.experts.0.w1.weight": ("F8_E4M3", [32, 64]),
            "embed.weight": ("BF16", [64, 64]),
            "head.weight": ("BF16", [64, 64]),
            "mtp.0.head.weight": ("BF16", [64, 64]),
        },
    )
    return checkpoint, 5 * 4096


def test_dp4_weights_count_the_checkpoint_once(tmp_path: Path) -> None:
    """The plan's weights are the single-device estimate, DP or not.

    Free memory is summed over devices, but the DP replicas are runtime
    resident bytes the engine's own KV sizing already sees (the weights
    are loaded before the KV pool is sized): counting them here too
    refused exports the devices fit.
    """
    checkpoint, checkpoint_bytes = _checkpoint(tmp_path)
    paths = (checkpoint,)
    planner = _planner()
    single = planner.estimate_weights_size(
        _pipeline_config(
            dp=1,
            ep_size=1,
            n_devices=1,
            paths=paths,
            weights_size=checkpoint_bytes,
        )
    )
    summed = planner.estimate_weights_size(
        _pipeline_config(
            dp=4,
            ep_size=4,
            n_devices=4,
            paths=paths,
            weights_size=checkpoint_bytes,
        )
    )
    config = V41Config.from_hf(_HF_CONFIG)
    text = text_config()
    woa_elements = (
        int(text.o_groups)
        * int(text.o_lora_rank)
        * (int(text.num_attention_heads) // int(text.o_groups))
        * int(text.head_dim)
    )
    expansion = int(text.num_hidden_layers) * (
        woa_elements - woa_elements // (32 * 32)
    )
    once = max(checkpoint_bytes - engram_table_bytes(config) + expansion, 0)
    assert single == once
    assert summed == single


def test_dp4_activations_exclude_the_ep_reserve() -> None:
    """The summed activation estimate is the per-device scratch alone.

    The EP dispatch's comm buffers and receive pool are not forward
    scratch, and the export plan leaves them to the engine: they
    allocate after the plan has priced the KV budget, and reserving
    them out of it here double-books bytes the engine books itself
    through the sidecar.
    """
    planner = _planner()
    without_ep = planner.estimate_activation_memory(
        _pipeline_config(dp=4, ep_size=1, n_devices=4), _HF_CONFIG
    )
    with_ep = planner.estimate_activation_memory(
        _pipeline_config(dp=4, ep_size=4, n_devices=4), _HF_CONFIG
    )
    assert with_ep == without_ep


def test_dp4_sidecar_scratch_reserves_the_ep_buffers() -> None:
    """The sidecar figure the engine's KV sizing holds back covers EP.

    The engine sizes and allocates the unified KV arena before EP init
    runs its handshake and allocates the dispatch heaps, so the sidecar
    must carry the comm buffers and receive pool or an unpinned boot
    hands their room to the KV pool.
    """
    planner = _planner()
    without_ep = planner.forward_scratch_bytes_per_device(
        _pipeline_config(dp=4, ep_size=1, n_devices=4)
    )
    config = _pipeline_config(dp=4, ep_size=4, n_devices=4)
    with_ep = planner.forward_scratch_bytes_per_device(config)
    comm, pool = _expected_ep_reserve(config)
    assert comm > 0
    assert pool > 0
    assert with_ep - without_ep == comm + pool


def test_unwired_shapes_add_no_ep_reserve() -> None:
    """Only the wired ``dp == ep_size == n_devices`` shape carries EP buffers.

    Every other shape the model refuses before load, or serves without
    EP, must reserve none: the sidecar falls back to the forward
    scratch alone.
    """
    planner = _planner()
    scratch = planner.forward_scratch_bytes_per_device(
        _pipeline_config(dp=4, ep_size=1, n_devices=4)
    )
    for dp, ep_size, n_devices in ((1, 1, 1), (1, 1, 2), (2, 4, 4), (4, 2, 4)):
        sidecar = planner.forward_scratch_bytes_per_device(
            _pipeline_config(dp=dp, ep_size=ep_size, n_devices=n_devices)
        )
        assert sidecar == scratch


def test_the_dp4_ep4_plan_stays_admissible() -> None:
    """The frozen DP4/EP4 geometry plans a KV budget, and refuses nothing.

    Drives the same ``plan_from_sizes`` the exporter runs, at the export's
    utilization and page size over a device list the EP reserve once
    pushed past the budget -- the refusal that blocked every GPU run.
    """
    planner = _planner()
    arch_config = _arch_config()
    config = _pipeline_config(dp=4, ep_size=4, n_devices=4)
    weights = planner.estimate_weights_size(config)
    activations = planner.estimate_activation_memory(config, _HF_CONFIG)
    signals = config.estimate_signal_buffer_memory(arch_config)
    free = 5 * _GIB // 4  # per device: the plan fits, the EP terms did not
    devices = _devices(free)

    plan = MemoryEstimator.plan_from_sizes(
        config,
        config.model,
        arch_config,
        devices,
        weights,
        activations,
        signals,
        max_batch_size=32,
    )

    expected_available = int(
        _UTILIZATION * 4 * free - weights - activations - signals
    )
    assert plan.available_cache_memory == kv_cache_memory_size(
        params=arch_config.get_kv_params(),
        available_cache_memory=expected_available,
        max_batch_size=32,
        max_seq_len=config.model.max_length,
        is_di_enabled=config.runtime.is_disaggregated,
        model_name=config.model.model_name,
    )
    assert plan.available_cache_memory > 0
    assert plan.planned_max_batch_size == 32


def test_dp4_scratch_matches_the_summed_estimate() -> None:
    """The plan and the sidecar agree on the scratch; EP rides engine-side.

    The plan subtracts the summed scratch from the KV budget while the
    engine holds the sidecar back per device, so the two must agree on
    the forward's bytes or the plan and the runtime size the same
    forward differently. The EP reserve rides only with the engine: the
    KV pool is sized before the dispatch heaps allocate.
    """
    planner = _planner()
    config = _pipeline_config(dp=4, ep_size=4, n_devices=4)
    summed = planner.estimate_activation_memory(config, _HF_CONFIG)
    per_device = planner.forward_scratch_bytes_per_device(config)
    comm, pool = _expected_ep_reserve(config)
    assert summed == (per_device - comm - pool) * 4


if __name__ == "__main__":
    raise SystemExit("run via bazel or pytest")
