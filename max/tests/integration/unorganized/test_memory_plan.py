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

"""Tests for FreeMemory and the NVLS multicast pool sizing."""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device, DeviceSpec
from max.nn.legacy.comm import MulticastPool
from max.pipelines.lib import memory_ledger
from max.pipelines.lib.memory_plan import FreeMemory, MemoryPlan
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH

GIB = 1024**3
_CONFIG = "max.pipelines.lib.config.config"


def _device_with_stats(**stats: int | None) -> MagicMock:
    device = MagicMock(spec=Device)
    device.stats = stats
    return device


def test_free_memory__sums_exact_and_unmeasured_devices() -> None:
    exact = _device_with_stats(free_memory=10 * GIB, total_memory=80 * GIB)
    unmeasured = _device_with_stats(free_memory=None, total_memory=80 * GIB)
    assert FreeMemory.summed([exact, unmeasured]) == FreeMemory(
        lo=10 * GIB, hi=90 * GIB
    )


def test_free_memory__smallest_takes_each_bound_independently() -> None:
    devices = [
        _device_with_stats(free_memory=10 * GIB, total_memory=80 * GIB),
        _device_with_stats(free_memory=None, total_memory=80 * GIB),
    ]
    assert FreeMemory.smallest(devices) == FreeMemory(lo=0, hi=10 * GIB)
    assert FreeMemory.smallest([]) == FreeMemory(lo=0, hi=0)


_ALLREDUCE = "max.nn.legacy.comm.allreduce"


@pytest.mark.parametrize(
    "num_devices,num_tokens,hidden_size,expected",
    [
        # Below the rank floor: no pool regardless of payload.
        (2, 8192, 8192, 0),
        # Exactly num_tokens x hidden_size x bf16.
        (4, 8192, 4096, 8192 * 4096 * 2),
        (8, 8192, 8192, 8192 * 8192 * 2),
        # Unaligned payloads are returned as is; the driver does the rounding.
        (4, 1000, 4096, 1000 * 4096 * 2),
        # Past the ceiling the pool is disabled outright, not truncated.
        (8, 8192, 65536, 0),
        # An architecture reporting no hidden width or no token budget gets
        # no pool.
        (8, 8192, 0, 0),
        (8, 0, 4096, 0),
    ],
)
def test_multicast_pool_capacity_bytes(
    num_devices: int, num_tokens: int, hidden_size: int, expected: int
) -> None:
    """``capacity_bytes`` is the single sizing authority for the NVLS pool."""
    assert (
        MulticastPool.capacity_bytes(
            num_devices=num_devices,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
        )
        == expected
    )


def _gpu_config(
    num_devices: int, hidden_size: int | str | None
) -> DummyPipelineConfig:
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=[DeviceSpec.accelerator(id=i) for i in range(num_devices)],
    )
    if hidden_size is not None:
        cfg.model.huggingface_config.hidden_size = hidden_size
    return cfg


def _driver_pool(supported: bool = True) -> MagicMock:
    driver_pool = MagicMock()
    driver_pool.is_supported.return_value = supported
    return driver_pool


@pytest.fixture
def multicast_pool_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MulticastPool, "ENABLED", True)


def test_multicast_pool_bytes__disabled_by_default() -> None:
    """Until a kernel consumes the pool, serving does not size it: the driver
    is not even asked."""
    assert MulticastPool.ENABLED is False
    cfg = _gpu_config(8, hidden_size=4096)
    driver_pool = _driver_pool()
    with (
        patch(f"{_CONFIG}.load_devices", return_value=[]) as load,
        patch(f"{_CONFIG}.SymmetricPool", driver_pool),
    ):
        assert cfg.multicast_pool_bytes() == 0
    load.assert_not_called()
    driver_pool.is_supported.assert_not_called()


@pytest.mark.usefixtures("multicast_pool_enabled")
def test_multicast_pool_bytes__charges_the_capacity() -> None:
    """The pool is the sized capacity once the driver confirms support."""
    cfg = _gpu_config(8, hidden_size=4096)
    capacity = MulticastPool.capacity_bytes(
        8, cfg.runtime.max_batch_input_tokens, 4096
    )
    assert capacity > 0
    driver_pool = _driver_pool()
    with (
        patch(f"{_CONFIG}.load_devices", return_value=[]),
        patch(f"{_CONFIG}.SymmetricPool", driver_pool),
    ):
        assert cfg.multicast_pool_bytes() == capacity
    driver_pool.is_supported.assert_called_once_with([])


@pytest.mark.usefixtures("multicast_pool_enabled")
def test_multicast_pool_bytes__unsupported_devices_is_zero() -> None:
    """A box the driver cannot back a pool on gets none."""
    cfg = _gpu_config(8, hidden_size=4096)
    driver_pool = _driver_pool(supported=False)
    with (
        patch(f"{_CONFIG}.load_devices", return_value=[]),
        patch(f"{_CONFIG}.SymmetricPool", driver_pool),
    ):
        assert cfg.multicast_pool_bytes() == 0


@pytest.mark.usefixtures("multicast_pool_enabled")
@pytest.mark.parametrize(
    "num_devices,hidden_size",
    [
        # Below the rank floor.
        (2, 4096),
        # No integer hidden width: a MagicMock attribute, or a string.
        (8, None),
        (8, "4096"),
        # Payload past the ceiling.
        (8, 65536),
    ],
)
def test_multicast_pool_bytes__sizing_rejections_skip_the_driver(
    num_devices: int, hidden_size: int | str | None
) -> None:
    """Sizing rejects before the driver is consulted."""
    cfg = _gpu_config(num_devices, hidden_size=hidden_size)
    driver_pool = _driver_pool()
    with (
        patch(f"{_CONFIG}.load_devices", return_value=[]) as load,
        patch(f"{_CONFIG}.SymmetricPool", driver_pool),
    ):
        assert cfg.multicast_pool_bytes() == 0
    load.assert_not_called()
    driver_pool.is_supported.assert_not_called()


def test_multicast_pool_allocate__zero_bytes_skips_the_driver() -> None:
    """A zero charge means no pool: the driver is neither queried nor asked
    to allocate."""
    driver_pool = MagicMock()
    with patch(f"{_ALLREDUCE}.DriverSymmetricPool", driver_pool):
        assert MulticastPool.allocate([MagicMock()], 0) is None
    driver_pool.lookup.assert_not_called()
    driver_pool.allocate.assert_not_called()


def test_multicast_pool_allocate__reuses_the_registered_pool() -> None:
    """A second model over the same devices gets the existing pool instead
    of a failing second allocation."""
    driver_pool = MagicMock()
    existing = object()
    driver_pool.lookup.return_value = existing
    with (
        patch(f"{_ALLREDUCE}.is_virtual_device_mode", return_value=False),
        patch(f"{_ALLREDUCE}.DriverSymmetricPool", driver_pool),
    ):
        assert MulticastPool.allocate([MagicMock()], 4096) is existing
    driver_pool.allocate.assert_not_called()


def test_multicast_pool_allocate__failure_falls_back_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed allocation is a fallback, but a warned one."""
    driver_pool = MagicMock()
    driver_pool.lookup.return_value = None
    driver_pool.allocate.side_effect = RuntimeError("cuMulticastCreate failed")
    with (
        patch(f"{_ALLREDUCE}.is_virtual_device_mode", return_value=False),
        patch(f"{_ALLREDUCE}.DriverSymmetricPool", driver_pool),
        caplog.at_level(logging.WARNING, logger=_ALLREDUCE),
    ):
        assert MulticastPool.allocate([MagicMock()], 4096) is None
    assert any(
        record.levelno == logging.WARNING
        and "cuMulticastCreate failed" in record.getMessage()
        and "4096" in record.getMessage()
        for record in caplog.records
    )


# ---- Measured KV-cache sizing (max.pipelines.lib.memory_ledger) ------------


def _ledger(
    resident: int, eager: int = 0, capture: int = 0, total: int = 100
) -> memory_ledger.DeviceLedger:
    return memory_ledger.DeviceLedger(
        total_bytes=total * GIB,
        resident_bytes=resident * GIB,
        eager_bytes=eager * GIB,
        capture_bytes=capture * GIB,
    )


def test_cache_budget__the_least_free_device_bounds_every_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    ledgers = [
        _ledger(resident=40, eager=10, capture=2),
        _ledger(resident=40, eager=14, capture=2),
    ]
    # 100 * 0.9 - 40 - 14 - 2 - 1 = 33 GiB on the tighter device, twice.
    assert memory_ledger.cache_budget(ledgers, 0.9) == 2 * 33 * GIB


def test_cache_budget__never_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(memory_ledger.MARGIN_ENV, raising=False)
    assert memory_ledger.cache_budget([_ledger(resident=95)], 0.9) == 0


def test_measures__only_accelerators_with_measuring_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(memory_ledger.DISABLE_ENV, raising=False)
    gpu = MagicMock(is_host=False, api="cuda")
    host = MagicMock(is_host=True, api="cpu")
    npu = MagicMock(is_host=False, api="npu")
    assert memory_ledger.measures([gpu, gpu])
    assert not memory_ledger.measures([host])
    assert not memory_ledger.measures([])
    # An accelerator that loads precompiled programs is sized before loading.
    assert not memory_ledger.measures([npu])
    with memory_ledger.measuring_off():
        assert not memory_ledger.measures([gpu])
    assert memory_ledger.measures([gpu])


def test_sizes_after_load__gpus_even_with_measuring_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(memory_ledger.DISABLE_ENV, raising=False)
    gpu = MagicMock(is_host=False, api="cuda")
    host = MagicMock(is_host=True, api="cpu")
    npu = MagicMock(is_host=False, api="npu")
    assert memory_ledger.sizes_after_load([gpu, gpu])
    assert not memory_ledger.sizes_after_load([host])
    assert not memory_ledger.sizes_after_load([])
    assert not memory_ledger.sizes_after_load([npu])
    # An exporter turns measuring off, but its GPUs still size the cache
    # from what the weights left rather than from the checkpoint's size.
    with memory_ledger.measuring_off():
        assert memory_ledger.sizes_after_load([gpu])
        assert not memory_ledger.measures([gpu])


def test_unmeasured_budget__holds_back_a_fixed_reserve_per_device() -> None:
    devices = [
        _device_with_stats(free_memory=60 * GIB, total_memory=100 * GIB),
        _device_with_stats(free_memory=50 * GIB, total_memory=100 * GIB),
    ]
    for d in devices:
        d.is_host = False
        d.api = "cuda"
    # min(90 - 40, 90 - 50) - 6 = 34 GiB per device.
    assert memory_ledger.unmeasured_budget(devices, 0.9) == 2 * 34 * GIB


def test_unmeasured_budget__a_precompiled_accelerator_takes_a_share_of_free() -> (
    None
):
    device = _device_with_stats(free_memory=20 * GIB, total_memory=32 * GIB)
    device.is_host = False
    device.api = "npu"
    # Its programs' memory sits outside the budget: 0.7 of what is free.
    assert memory_ledger.unmeasured_budget([device], 0.7) == int(20 * GIB * 0.7)


def test_unmeasured_budget__an_accelerator_that_reports_no_free_memory() -> (
    None
):
    device = _device_with_stats(free_memory=None, total_memory=32 * GIB)
    device.is_host = False
    device.api = "npu"
    # A plugin HAL can report None: the most that could be free is the total.
    assert memory_ledger.unmeasured_budget([device], 0.7) == int(32 * GIB * 0.7)


def _live_device(live: int) -> MagicMock:
    device = MagicMock()
    device.memory_manager_stats = {"live_bytes": live}
    return device


def test_check_capture__holds_capture_to_its_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    pipeline = MagicMock()
    pipeline._devices = [_live_device(12 * GIB)]
    pipeline.memory_ledger = memory_ledger.Ledger(
        devices=[_ledger(resident=40, eager=8, capture=3)],
        live_before_capture=[8 * GIB],
    )
    # Kept 4 GiB: the 3 GiB share plus the 1 GiB margin.
    memory_ledger.check_capture(pipeline)
    pipeline._devices = [_live_device(13 * GIB)]
    with pytest.raises(RuntimeError, match="Graph capture kept"):
        memory_ledger.check_capture(pipeline)


def test_check_capture__without_a_ledger_is_a_no_op() -> None:
    pipeline = MagicMock(spec=["_devices"])
    memory_ledger.check_capture(pipeline)


def _sizing_pipeline(max_batch_size: int | None) -> MagicMock:
    from max.pipelines.kv_cache import NullKVCache

    device = _device_with_stats(free_memory=50 * GIB, total_memory=100 * GIB)
    device.is_host = False
    device.max_single_alloc_size = 1 << 50
    device.memory_manager_stats = {
        "live_bytes": 50 * GIB,
        "peak_live_bytes": 50 * GIB,
        "max_request_bytes": 0,
        "driver_fallbacks": 0,
    }
    pipeline = MagicMock()
    pipeline._devices = [device]
    pipeline.kv_manager = MagicMock(spec=NullKVCache, slab_bytes=0)
    config = pipeline.pipeline_config
    config.model.kv_cache.device_memory_utilization = 0.9
    config.runtime.max_batch_size = max_batch_size
    config.runtime.max_batch_input_tokens = 8192
    config.runtime.max_batch_total_tokens = None
    config.runtime.device_graph_capture = True
    return pipeline


def test_size_kv_cache__claims_eager_plus_capture_and_installs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    pipeline = _sizing_pipeline(max_batch_size=None)
    plan = MemoryPlan(
        planned_max_batch_size=64,
        footprint=0,
        planned_max_length=32768,
        device_specs=(),
    )
    peaks = {
        "prefill": [6 * GIB],
        "prefill_long": [9 * GIB],
        "decode": [2 * GIB],
    }
    with (
        patch.object(memory_ledger, "run_warmups", return_value=peaks),
        patch("max.pipelines.kv_cache.kv_cache_memory_size", return_value=GIB),
        patch(
            "max.pipelines.kv_cache.max_seq_len_fitting_in_cache",
            return_value=16384,
        ),
        patch("max.pipelines.kv_cache.load_kv_manager") as load,
    ):
        got = memory_ledger.size_kv_cache(pipeline, plan)
    # 90 - 50 resident - 9 eager - 2 capture - 1 margin = 28 GiB.
    assert got.available_cache_memory == 28 * GIB
    assert got.planned_max_length == 16384
    assert got.planned_max_batch_size == 64
    assert load.call_args.kwargs["available_cache_memory"] == 28 * GIB
    installs = pipeline.install_kv_manager.call_args_list
    assert [c.args[0] for c in installs] == [None, load.return_value]
    assert pipeline.memory_ledger.devices[0].capture_bytes == 2 * GIB


def test_size_kv_cache__counts_what_the_warmups_kept_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    pipeline = _sizing_pipeline(max_batch_size=None)
    (device,) = pipeline._devices
    plan = MemoryPlan(
        planned_max_batch_size=64,
        footprint=0,
        planned_max_length=32768,
        device_specs=(),
    )

    def warmups(*_: object) -> dict[str, list[int]]:
        # The warmups keep 2 GiB (an arena built on first use); each shape's
        # peak is above what was live when it ended.
        device.memory_manager_stats["live_bytes"] = 52 * GIB
        return {"prefill": [6 * GIB], "decode": [2 * GIB]}

    with (
        patch.object(memory_ledger, "run_warmups", side_effect=warmups),
        patch("max.pipelines.kv_cache.kv_cache_memory_size", return_value=GIB),
        patch(
            "max.pipelines.kv_cache.max_seq_len_fitting_in_cache",
            return_value=None,
        ),
        patch("max.pipelines.kv_cache.load_kv_manager"),
    ):
        got = memory_ledger.size_kv_cache(pipeline, plan)
    # 90 - (50 + 2 kept) resident - 6 eager - 2 capture - 1 margin = 29 GiB.
    assert got.available_cache_memory == 29 * GIB
    assert pipeline.memory_ledger.devices[0].resident_bytes == 52 * GIB


def test_size_kv_cache__the_capped_cache_must_hold_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    pipeline = _sizing_pipeline(max_batch_size=64)
    # The budget is ample, but the device's largest allocation is not.
    pipeline._devices[0].max_single_alloc_size = GIB // 2
    plan = MemoryPlan(
        planned_max_batch_size=64,
        footprint=0,
        planned_max_length=4096,
        device_specs=(),
    )
    with (
        patch.object(
            memory_ledger, "run_warmups", return_value={"decode": [GIB]}
        ),
        patch("max.pipelines.kv_cache.kv_cache_memory_size", return_value=GIB),
        patch("max.pipelines.kv_cache.load_kv_manager") as load,
        pytest.raises(RuntimeError, match="one prefill chunk needs"),
    ):
        memory_ledger.size_kv_cache(pipeline, plan)
    load.assert_not_called()


def test_size_kv_cache__halves_an_unset_batch_after_an_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from max.pipelines.kv_cache.paged_kv_cache import null_cache_manager

    monkeypatch.setenv(memory_ledger.MARGIN_ENV, str(GIB))
    monkeypatch.setattr(
        null_cache_manager.NullKVCache, "__init__", lambda self, **_: None
    )
    pipeline = _sizing_pipeline(max_batch_size=None)
    plan = MemoryPlan(
        planned_max_batch_size=64,
        footprint=0,
        planned_max_length=4096,
        device_specs=(),
    )
    calls: list[int] = []

    def warmups(
        pipeline: object, tokenizer: object, batch: int, *_: object
    ) -> dict[str, list[int]]:
        calls.append(batch)
        if batch > 16:
            raise RuntimeError("CUDA error: out of memory")
        return {"decode": [GIB]}

    with (
        patch.object(memory_ledger, "run_warmups", side_effect=warmups),
        patch("max.pipelines.kv_cache.kv_cache_memory_size", return_value=GIB),
        patch(
            "max.pipelines.kv_cache.max_seq_len_fitting_in_cache",
            return_value=None,
        ),
        patch("max.pipelines.kv_cache.load_kv_manager"),
    ):
        got = memory_ledger.size_kv_cache(pipeline, plan)
    assert calls == [64, 32, 16]
    assert got.planned_max_batch_size == 16


def test_size_kv_cache__a_user_batch_that_does_not_fit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _sizing_pipeline(max_batch_size=64)
    plan = MemoryPlan(
        planned_max_batch_size=64,
        footprint=0,
        planned_max_length=4096,
        device_specs=(),
    )
    with (
        patch.object(
            memory_ledger,
            "run_warmups",
            side_effect=RuntimeError("CUDA error: out of memory"),
        ),
        patch("max.pipelines.kv_cache.kv_cache_memory_size", return_value=GIB),
        pytest.raises(RuntimeError, match="max_batch_size=64"),
    ):
        memory_ledger.size_kv_cache(pipeline, plan)


class _ImageTokenizer:
    """Builds contexts whose token count grows with the image side."""

    def __init__(
        self, tokens_per_side: dict[int, int], limit: int | None = None
    ):
        self.tokens_per_side = tokens_per_side
        self.max_images_per_request = limit
        self.built: list[tuple[int, int]] = []

    async def new_context(self, request: Any) -> Any:
        from PIL import Image

        n = len(request.images)
        side = Image.open(io.BytesIO(request.images[0])).size[0] if n else 0
        self.built.append((n, side))
        return SimpleNamespace(
            images=[object()] * n,
            tokens=_Tokens([0] * (n * self.tokens_per_side.get(side, 0))),
        )


class _TextTokenizer:
    async def new_context(self, request: Any) -> Any:
        return SimpleNamespace(tokens=[0] * 10)


def test_image_warmup_fills_a_chunk_with_the_largest_images() -> None:
    tok = _ImageTokenizer({2048: 1000})
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=2)
    # The scheduler gives each replica a full chunk: eight 1000-token images.
    assert len(contexts) == 2
    assert tok.built[-2:] == [(8, 2048), (8, 2048)]


def test_image_warmup_chunks_an_image_longer_than_the_chunk() -> None:
    tok = _ImageTokenizer({2048: 5000})
    (ctx,) = memory_ledger._image_contexts(tok, chunk=4096, n=1)
    # The whole image is encoded in its first chunk's step.
    assert tok.built[-1] == (1, 2048)
    assert ctx.tokens.active_length == 4096


def test_image_warmup_without_chunking_keeps_the_whole_prompt() -> None:
    tok = _ImageTokenizer({2048: 5000})
    contexts = memory_ledger._image_contexts(
        tok, chunk=20000, n=1, chunked=False
    )
    # The scheduler runs the whole prompt in one step: four images, uncut.
    assert tok.built[-1] == (4, 2048)
    assert [c.tokens.active_length for c in contexts] == [20000]


def test_image_warmup_respects_the_per_request_image_limit() -> None:
    tok = _ImageTokenizer({2048: 100}, limit=3)
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=1)
    # 81 images fill the chunk: 27 requests of three.
    assert len(contexts) == 27
    assert tok.built[-1] == (3, 2048)


def test_image_warmup_splits_images_for_a_one_image_tokenizer() -> None:
    tok = _ImageTokenizer({2048: 1000}, limit=1)
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=2)
    # Sixteen one-image requests carry the eight per replica.
    assert len(contexts) == 16
    assert tok.built[-1] == (1, 2048)


class _FirstImageTokenizer(_ImageTokenizer):
    """Names no limit, but keeps only the first image's pixels, so a request
    of several images fails its placeholder check."""

    async def new_context(self, request: Any) -> Any:
        if len(request.images) > 1:
            raise ValueError("30 image placeholder runs, but 1 images")
        return await super().new_context(request)


def test_image_warmup_takes_one_image_per_request_when_several_fail() -> None:
    tok = _FirstImageTokenizer({2048: 1000})
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=2)
    # Sixteen one-image requests carry the eight per replica.
    assert len(contexts) == 16
    assert all(len(ctx.images) == 1 for ctx in contexts)


def test_image_warmup_holds_each_replica_to_its_batch_size() -> None:
    tok = _ImageTokenizer({2048: 100}, limit=1)
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=2, max_rows=4)
    # 40 images would fill each replica's chunk, but a step takes four rows.
    assert len(contexts) == 8


def test_image_warmup_takes_the_side_with_the_most_tokens() -> None:
    tok = _ImageTokenizer({2048: 1000, 4096: 4000})
    contexts = memory_ledger._image_contexts(tok, chunk=8192, n=1)
    # A processor that keeps 4096 px images: two of them fill the chunk.
    assert len(contexts) == 1
    assert tok.built[-1] == (2, 4096)


def test_no_image_warmup_for_a_text_only_image_limit() -> None:
    tok = _ImageTokenizer({2048: 1000}, limit=0)
    # A limit of 0 serves text only (the router rejects images).
    assert memory_ledger._image_contexts(tok, chunk=8192, n=1) == []
    assert tok.built == []


def test_no_image_warmup_for_a_text_tokenizer() -> None:
    assert (
        memory_ledger._image_contexts(_TextTokenizer(), chunk=8192, n=1) == []
    )


class _Tokens(list[int]):
    """A prompt whose next step can be chunked, as a TokenBuffer's can."""

    @property
    def active_length(self) -> int:
        return getattr(self, "_chunk", len(self))

    def chunk(self, n: int) -> None:
        self._chunk = n


class _VideoTokenizer:
    """Lays a clip out as the tokens ``tokens_per_clip`` gives its shape."""

    def __init__(self, tokens_per_clip: dict[bytes, int]):
        self.tokens_per_clip = tokens_per_clip
        self.built: list[tuple[int, bytes]] = []

    async def new_context(self, request: Any) -> Any:
        n = len(request.videos)
        clip = request.videos[0] if n else b""
        self.built.append((n, clip))
        return SimpleNamespace(
            tokens=_Tokens([0] * (10 + n * self.tokens_per_clip.get(clip, 0)))
        )


def _fake_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_ledger, "_clip", lambda frames, side: f"{frames}x{side}".encode()
    )


def test_video_warmup_takes_the_clip_with_the_most_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clips(monkeypatch)
    tok = _VideoTokenizer({b"2x3584": 3000, b"16x1280": 3000, b"64x640": 2000})
    contexts = memory_ledger._video_contexts(tok, chunk=8192, n=1)
    # A tie goes to the fewest, largest frames; two 3000-token clips fit.
    assert tok.built[-1] == (2, b"2x3584")
    assert len(contexts) == 1 and not hasattr(contexts[0].tokens, "_chunk")


def test_video_warmup_chunks_a_clip_longer_than_the_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clips(monkeypatch)
    tok = _VideoTokenizer({b"2x3584": 12288})
    (ctx,) = memory_ledger._video_contexts(tok, chunk=8192, n=1)
    assert tok.built[-1] == (1, b"2x3584")
    assert ctx.tokens.active_length == 8192


def test_video_warmup_without_chunking_keeps_the_whole_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clips(monkeypatch)
    tok = _VideoTokenizer({b"2x3584": 12288})
    (ctx,) = memory_ledger._video_contexts(tok, chunk=30000, n=1, chunked=False)
    # Two clips and the prompt's text fit the longest prompt, uncut.
    assert tok.built[-1] == (2, b"2x3584")
    assert ctx.tokens.active_length == 10 + 2 * 12288


@pytest.mark.parametrize("chunked", [True, False])
def test_media_warmups_fill_each_replica_widest_prefill(
    monkeypatch: pytest.MonkeyPatch, chunked: bool
) -> None:
    calls: dict[str, tuple[Any, ...]] = {}

    def media(key: str) -> Any:
        def build(tokenizer: Any, *args: Any) -> list[Any]:
            calls[key] = args
            return []

        return build

    monkeypatch.setattr(memory_ledger, "_contexts", lambda *a: [])
    monkeypatch.setattr(memory_ledger, "_cached", lambda *a: [])
    monkeypatch.setattr(memory_ledger, "_run", lambda *a, **k: None)
    monkeypatch.setattr(memory_ledger, "_serves_images", lambda c: True)
    monkeypatch.setattr(memory_ledger, "_image_contexts", media("image"))
    monkeypatch.setattr(memory_ledger, "_video_contexts", media("video"))
    config = SimpleNamespace(
        model=SimpleNamespace(data_parallel_degree=2),
        runtime=SimpleNamespace(
            max_batch_input_tokens=8192, enable_chunked_prefill=chunked
        ),
    )
    memory_ledger.run_warmups(
        SimpleNamespace(pipeline_config=config),
        tokenizer=None,
        max_batch_size=16,
        max_length=100004,
    )
    # Each of two replicas fills a chunk, or without chunking the longest
    # prompt, with eight rows.
    want = (8192 if chunked else 100000, 2, 8, chunked)
    assert calls == {"image": want, "video": want}


def test_no_video_warmup_for_a_text_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clips(monkeypatch)
    assert (
        memory_ledger._video_contexts(_TextTokenizer(), chunk=8192, n=1) == []
    )


def test_no_video_warmup_for_a_text_only_video_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clips(monkeypatch)
    tok = _VideoTokenizer({b"2x3584": 3000})
    tok.max_videos_per_request = 0  # type: ignore[attr-defined]
    assert memory_ledger._video_contexts(tok, chunk=8192, n=1) == []
    assert tok.built == []


def test_synthetic_clip_decodes_to_its_frames() -> None:
    import av

    clip = memory_ledger._clip(4, 64)
    with av.open(io.BytesIO(clip), mode="r") as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 4
    assert (frames[0].width, frames[0].height) == (64, 64)


class _CountingTextTokenizer:
    """Builds real text contexts and counts how many it built."""

    def __init__(self) -> None:
        self.built = 0

    async def new_context(self, request: Any) -> Any:
        import numpy as np
        from max.pipelines.context import TextContext, TokenBuffer

        self.built += 1
        return TextContext(
            max_length=len(request.prompt) + 8,
            tokens=TokenBuffer(np.array(request.prompt, dtype=np.int64)),
            request_id=request.request_id,
        )


def test_cached_contexts_tokenize_once_and_keep_their_own_state() -> None:
    tok = _CountingTextTokenizer()
    contexts = memory_ledger._cached(tok, prompt_len=4096, cached=4095, n=64)

    assert tok.built == 1
    assert len(contexts) == 64
    assert len({c.request_id for c in contexts}) == 64
    assert all(c.tokens.processed_length == 4095 for c in contexts)
    assert all(c.tokens.active_length == 1 for c in contexts)

    # Advancing one row leaves the others' processing state alone.
    contexts[0].tokens.advance_with_token(7)
    assert contexts[0].tokens.processed_length == 4096
    assert all(c.tokens.processed_length == 4095 for c in contexts[1:])
    assert all(len(c.tokens) == 4096 for c in contexts[1:])


@pytest.mark.parametrize(
    ("served", "want"), [(4096, 4096), (None, 8192), (16384, 8192)]
)
def test_clamp_tokenizer_to_pipeline__lowers_to_the_served_length(
    served: int | None, want: int
) -> None:
    from max.pipelines.lib.registry import clamp_tokenizer_to_pipeline

    tokenizer = SimpleNamespace(max_length=8192)
    pipeline = SimpleNamespace(
        memory_plan=MemoryPlan(
            planned_max_batch_size=1,
            footprint=0,
            planned_max_length=served,
        )
    )
    clamp_tokenizer_to_pipeline(tokenizer, pipeline)
    # Only ever lowered: the tokenizer never accepts more than it was built for.
    assert tokenizer.max_length == want
