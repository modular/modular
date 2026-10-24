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

"""Unit tests for the KV cache host and disk capacity preflights."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from max.driver import CPU
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVCacheMemory,
    KVConnectorType,
)
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.kv_cache.connectors import tier_connector


def _set_available_host_memory(
    monkeypatch: pytest.MonkeyPatch, available: int | None
) -> None:
    monkeypatch.setattr(
        tier_connector, "available_host_memory", lambda: available
    )


def test_host_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 1024)

    with pytest.raises(RuntimeError, match="host_offload_max_gb"):
        tier_connector._check_host_memory_capacity(2048)


def test_host_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 4096)

    tier_connector._check_host_memory_capacity(4096)


def test_host_capacity_skips_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_available_host_memory(monkeypatch, None)

    tier_connector._check_host_memory_capacity(1 << 60)
    assert "skipping KV cache host capacity preflight" in caplog.text


def test_default_host_offload_is_one_and_a_half_times_the_device_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, 1 << 40)

    assert (
        tier_connector._default_host_offload_bytes(8 << 30, gtt_bound=False)
        == 12 << 30
    )


def test_default_host_offload_is_capped_to_what_the_process_may_use(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_available_host_memory(monkeypatch, 20 << 30)

    assert (
        tier_connector._default_host_offload_bytes(64 << 30, gtt_bound=False)
        == 18 << 30
    )
    assert "Reduced the default KV cache host offload budget" in caplog.text


def test_default_host_offload_uncapped_when_availability_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_host_memory(monkeypatch, None)

    assert (
        tier_connector._default_host_offload_bytes(64 << 30, gtt_bound=False)
        == 96 << 30
    )


def _set_gtt_usage(
    monkeypatch: pytest.MonkeyPatch, usage: tier_connector._GttUsage | None
) -> None:
    monkeypatch.setattr(tier_connector, "_read_gtt_usage", lambda: usage)


def test_gtt_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gtt_usage(
        monkeypatch, tier_connector._GttUsage(total_bytes=1024, used_bytes=0)
    )

    with pytest.raises(RuntimeError, match="host_offload_max_gb"):
        tier_connector._check_gtt_capacity(2048)


def test_gtt_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gtt_usage(
        monkeypatch, tier_connector._GttUsage(total_bytes=4096, used_bytes=0)
    )

    tier_connector._check_gtt_capacity(4096)


def test_gtt_capacity_accounts_for_what_the_node_already_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Headroom is total - used, not total: another process on the node
    # holding GTT must shrink what this preflight accepts.
    _set_gtt_usage(
        monkeypatch,
        tier_connector._GttUsage(total_bytes=4096, used_bytes=3072),
    )

    with pytest.raises(RuntimeError, match="host_offload_max_gb"):
        tier_connector._check_gtt_capacity(1025)
    tier_connector._check_gtt_capacity(1024)


def test_gtt_capacity_skips_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_gtt_usage(monkeypatch, None)

    tier_connector._check_gtt_capacity(1 << 60)
    assert "skipping the KV cache driver-staging capacity preflight" in (
        caplog.text
    )


def test_default_host_offload_gtt_bound_is_one_and_a_half_times_the_device_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gtt_usage(
        monkeypatch,
        tier_connector._GttUsage(total_bytes=1 << 40, used_bytes=0),
    )

    assert (
        tier_connector._default_host_offload_bytes(8 << 30, gtt_bound=True)
        == 12 << 30
    )


def test_default_host_offload_gtt_bound_is_capped_to_free_gtt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_gtt_usage(
        monkeypatch,
        tier_connector._GttUsage(total_bytes=20 << 30, used_bytes=0),
    )

    assert (
        tier_connector._default_host_offload_bytes(64 << 30, gtt_bound=True)
        == 18 << 30
    )
    assert "Reduced the default KV cache host offload budget" in caplog.text
    assert "free GTT" in caplog.text


def test_default_host_offload_gtt_bound_accounts_for_what_the_node_already_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Half the node's GTT is already held by another process, so the cap is
    # half what it would be against the raw total.
    _set_gtt_usage(
        monkeypatch,
        tier_connector._GttUsage(total_bytes=40 << 30, used_bytes=20 << 30),
    )

    assert (
        tier_connector._default_host_offload_bytes(64 << 30, gtt_bound=True)
        == 18 << 30
    )


def test_default_host_offload_gtt_bound_uncapped_when_availability_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gtt_usage(monkeypatch, None)

    assert (
        tier_connector._default_host_offload_bytes(64 << 30, gtt_bound=True)
        == 96 << 30
    )


def test_disk_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1024),
    )

    with pytest.raises(RuntimeError, match="disk_offload_max_gb"):
        tier_connector._check_disk_capacity("/tmp", 2048)


def test_disk_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=4096),
    )

    tier_connector._check_disk_capacity("/tmp", 4096)


def _create_with_driver_staging_on_hip(
    monkeypatch: pytest.MonkeyPatch,
    use_disk_direct_io: bool = False,
    disk_offload_max_gb: float = 1.0,
    host_staging_mlock: bool = False,
) -> None:
    """Drives ``TierConnector.create`` up to its HSA_USERPTR_FOR_PAGED_MEM
    check with ``host_staging_alloc='driver'`` on a stubbed HIP device.

    An empty ``replica_kv_memory`` skips the preceding CPU-device check, and
    ``create`` resolves the native class before any other check, so the
    stubbed ``_native_connector_class`` stands in for the `kv_tier_connector`
    extension this CPU-only test target doesn't depend on, and this never
    needs a GPU.
    """
    monkeypatch.setattr(tier_connector, "accelerator_api", lambda: "hip")
    monkeypatch.setattr(
        tier_connector,
        "_native_connector_class",
        lambda: object,
    )
    cfg = SimpleNamespace(
        type=KVConnectorType.rust_tiered,
        host_staging_alloc="driver",
        use_disk_direct_io=use_disk_direct_io,
        host_staging_mlock=host_staging_mlock,
        host_offload_max_gb=1.0,
        disk_offload_max_gb=disk_offload_max_gb,
    )
    tier_connector.TierConnector.create(
        leaves={"full": KVCacheGroupId.full()},
        replica_kv_memory=[],
        params=SimpleNamespace(kv_connector_config=cfg),
        device_memory_bytes=0,
    )


def test_create_refuses_driver_staging_on_hip_without_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one thing standing between a correctly protected deployment and
    one that silently pays the full driver-allocation cost for none of the
    protection (ENABLE-3039) -- so it must fail closed, not warn. Covered
    here with no GPU: the guard is pure Python reading ``os.environ``, and
    CI's own GPU runs always pass the variable, so they never exercise its
    absence."""
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)

    with pytest.raises(RuntimeError, match="HSA_USERPTR_FOR_PAGED_MEM"):
        _create_with_driver_staging_on_hip(monkeypatch)


def test_create_refuses_driver_staging_on_hip_with_the_wrong_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the literal '0' satisfies the check -- HIP falls back to movable
    userptr memory for anything else, including a truthy-looking '1'."""
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "1")

    with pytest.raises(RuntimeError, match="HSA_USERPTR_FOR_PAGED_MEM"):
        _create_with_driver_staging_on_hip(monkeypatch)


def test_create_refuses_direct_io_with_driver_staging_on_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AMD's driver-allocated GTT staging memory rejects O_DIRECT (EFAULT),
    and direct disk I/O has no buffered fallback, so the combination must be
    refused up front with a message naming the conflicting settings rather
    than surfacing as an errno from the native probe."""
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")

    with pytest.raises(ValueError, match="use_disk_direct_io"):
        _create_with_driver_staging_on_hip(monkeypatch, use_disk_direct_io=True)


def test_create_ignores_direct_io_without_a_disk_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``disk_offload_max_gb=0`` the connector is host-only, so no
    O_DIRECT operation can happen and the driver-staging refusal must not
    fire. Creation proceeds past the direct-I/O checks and only then trips on
    the empty ``replica_kv_memory`` this stub passes."""
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    monkeypatch.setattr(tier_connector, "_check_gtt_capacity", lambda _: None)

    with pytest.raises(IndexError):
        _create_with_driver_staging_on_hip(
            monkeypatch, use_disk_direct_io=True, disk_offload_max_gb=0
        )


def test_connector_config_defaults_disk_io_to_buffered() -> None:
    """Direct disk I/O is opt-in: the default must be byte-for-byte the
    buffered behavior that shipped before the flag existed."""
    assert KVConnectorConfig().use_disk_direct_io is False
    assert KVConnectorConfig(use_disk_direct_io=True).use_disk_direct_io is True


def _create_with_unaligned_direct_io(
    monkeypatch: pytest.MonkeyPatch, host_staging_mlock: bool = False
) -> None:
    """Drives ``TierConnector.create`` with registered staging on CUDA up to
    its direct-I/O alignment check, which raises."""
    monkeypatch.setattr(tier_connector, "accelerator_api", lambda: "cuda")
    monkeypatch.setattr(
        tier_connector,
        "_native_connector_class",
        lambda: object,
    )
    # A leaf of 4096+16 bytes: a token-scaled geometry can produce this with
    # a small page size; the check must fire before the staging region.
    unaligned = SimpleNamespace(
        leaf_id="full",
        host_bytes_per_page=4096 + 16,
        buffers=[SimpleNamespace(device=SimpleNamespace(is_host=False))],
    )
    cfg = SimpleNamespace(
        type=KVConnectorType.rust_tiered,
        host_staging_alloc="registered",
        use_disk_direct_io=True,
        host_staging_mlock=host_staging_mlock,
        host_offload_max_gb=None,
        disk_offload_max_gb=1.0,
        disk_offload_dir=None,
        num_disk_workers=2,
    )
    with pytest.raises(ValueError, match="offending leaves"):
        tier_connector.TierConnector.create(
            leaves={"full": KVCacheGroupId.full()},
            replica_kv_memory=cast(
                "Sequence[Mapping[str, KVCacheMemory]]",
                [{"full": unaligned}],
            ),
            params=SimpleNamespace(kv_connector_config=cfg),
            device_memory_bytes=0,
        )


def test_create_refuses_direct_io_with_unaligned_leaf_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every leaf whose block width is not a 4 KiB multiple must be named in
    one error before anything is allocated, since no buffered fallback
    exists and a single unaligned leaf makes every one of its blocks unable
    to take the direct path."""
    _create_with_unaligned_direct_io(monkeypatch)


@pytest.mark.parametrize(
    ("sysctl", "warns"),
    [("allowed", True), ("disallowed", False), ("missing", False)],
)
def test_create_warns_on_mlock_only_when_compaction_migrates_locked_pages(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    sysctl: str,
    warns: bool,
) -> None:
    (tmp_path / "allowed").write_text("1\n")
    (tmp_path / "disallowed").write_text("0\n")
    path = tmp_path / sysctl
    read_sysctl = tier_connector._compact_unevictable_allowed
    monkeypatch.setattr(
        tier_connector,
        "_compact_unevictable_allowed",
        lambda: read_sysctl(str(path)),
    )

    _create_with_unaligned_direct_io(monkeypatch, host_staging_mlock=True)

    assert ("vm.compact_unevictable_allowed=1" in caplog.text) is warns


@pytest.mark.parametrize(
    "cpp_message",
    [
        pytest.param(
            "ChunkedStagingRegion: driver_alloc chunk is not driver-owned "
            "GTT memory (...). HSA_USERPTR_FOR_PAGED_MEM=0 must be set in "
            "the process environment before the process starts.",
            id="hardening_check",
        ),
        pytest.param(
            "ChunkedStagingRegion: driver_alloc is not available in this "
            "build configuration (--config=hal-device-context); use the "
            "registered path instead.",
            id="hal_unsupported",
        ),
        pytest.param(
            "ChunkedStagingRegion: driver host allocation of 4294967296 "
            "bytes failed at chunk 2 of 4 (1073741824 bytes, after "
            "2147483648 bytes allocated): out of memory",
            id="genuine_allocation_failure",
        ),
    ],
)
def test_driver_staging_alloc_failure_propagates_unwrapped(
    monkeypatch: pytest.MonkeyPatch, cpp_message: str
) -> None:
    """``_ChunkedStagingRegion`` raises the same ``RuntimeError`` type for
    three distinct failures, each with its own already-actionable message.
    Re-wrapping them all as "reduce host_offload_max_gb" would be correct
    for only one of the three and actively wrong -- and message-hiding --
    for the other two, so the caller must see the original message intact."""

    def refuse(**kwargs: object) -> None:
        raise RuntimeError(cpp_message)

    monkeypatch.setattr(tier_connector, "_ChunkedStagingRegion", refuse)

    with pytest.raises(RuntimeError, match=re.escape(cpp_message)):
        tier_connector._alloc_chunked_staging_region(
            4, 1024**3, CPU(), "driver"
        )


def test_connector_config_defaults_mlock_off() -> None:
    assert KVConnectorConfig().host_staging_mlock is False
    assert KVConnectorConfig(host_staging_mlock=True).host_staging_mlock is True


def test_create_refuses_mlock_with_driver_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver-allocated memory is already driver-owned, so locking it is a
    configuration mistake and is refused up front."""
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")

    with pytest.raises(ValueError, match="host_staging_mlock"):
        _create_with_driver_staging_on_hip(monkeypatch, host_staging_mlock=True)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [("1\n", True), ("0\n", False)],
)
def test_compact_unevictable_allowed_reads_the_sysctl(
    tmp_path: Path, contents: str, expected: bool
) -> None:
    path = tmp_path / "compact_unevictable_allowed"
    path.write_text(contents)

    assert tier_connector._compact_unevictable_allowed(str(path)) is expected


def test_compact_unevictable_allowed_is_false_when_unreadable(
    tmp_path: Path,
) -> None:
    missing = str(tmp_path / "missing")

    assert tier_connector._compact_unevictable_allowed(missing) is False


@pytest.fixture
def fresh_native_class_cache() -> Iterator[None]:
    """Clears the cached native-class lookup around a test that stubs the import."""
    tier_connector._native_connector_class.cache_clear()
    yield
    tier_connector._native_connector_class.cache_clear()


def _stub_import_module(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def import_module(name: str) -> object:
        raise error

    monkeypatch.setattr(
        tier_connector.importlib, "import_module", import_module
    )


@pytest.mark.usefixtures("fresh_native_class_cache")
def test_create_fails_on_missing_extension_before_acquiring_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build without the extension must fail before the offload directory is
    acquired or any capacity is checked, so nothing is left to release."""
    _stub_import_module(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'kv_tier_connector'", name="kv_tier_connector"
        ),
    )

    def acquire_offload_dir(*args: object, **kwargs: object) -> None:
        raise AssertionError("acquire_offload_dir ran before the class check")

    monkeypatch.setattr(
        tier_connector, "acquire_offload_dir", acquire_offload_dir
    )
    cfg = SimpleNamespace(type=KVConnectorType.rust_tiered)

    with pytest.raises(ValueError, match="does not ship") as exc_info:
        tier_connector.TierConnector.create(
            leaves={"full": KVCacheGroupId.full()},
            replica_kv_memory=[],
            params=SimpleNamespace(kv_connector_config=cfg),
            device_memory_bytes=0,
        )
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.usefixtures("fresh_native_class_cache")
def test_native_class_propagates_a_broken_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the extension itself being absent is the "not shipped" error; a
    missing dependency inside it is a broken install and must surface."""
    _stub_import_module(
        monkeypatch, ModuleNotFoundError("No module named 'dep'", name="dep")
    )

    with pytest.raises(ModuleNotFoundError, match="dep"):
        tier_connector._native_connector_class()
