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
"""CPU lifecycle tests for Engram tables shared across processes in host memory.

Processes serving the same checkpoint must share one copy of the tables, and
must never attach to a copy that is stale, incomplete, from another
checkpoint, or reachable through a registry another local user could have
written. No GPU is needed: the ``register`` hook is a stub here.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import mmap
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41 import (
    engram_shared_host as esh,
)
from max.pipelines.architectures.deepseekV41 import model as model_mod
from max.pipelines.architectures.deepseekV41 import weight_adapters

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="memfd sharing is Linux-only"
)

_SHAPES = {"a.weight": (64, 32), "a.scale": (64, 1), "b.weight": (3, 5)}


def _tables(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(0, 256, size=shape, dtype=np.uint8)
        for name, shape in _SHAPES.items()
    }


class _Fill:
    def __init__(self, tables: Mapping[str, np.ndarray]) -> None:
        self.tables = tables
        self.calls = 0

    def __call__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.calls += 1
        for name, array in arrays.items():
            np.copyto(array, self.tables[name])


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(esh.DIR_ENV, str(tmp_path / "registry"))
    monkeypatch.delenv(esh.DISABLE_ENV, raising=False)
    return tmp_path / "registry"


@pytest.fixture
def opened() -> Iterator[list[esh.SharedHostTables]]:
    handles: list[esh.SharedHostTables] = []
    yield handles
    for handle in handles:
        handle.close()


def _private_mkdir(path: Path) -> Path:
    path.mkdir(exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _records(registry: Path, key: str) -> list[Path]:
    return sorted((registry / key).glob("provider-*.json"))


def _assert_tables(
    shared: esh.SharedHostTables, tables: Mapping[str, np.ndarray]
) -> None:
    assert set(shared.arrays) == set(tables)
    for name, array in tables.items():
        np.testing.assert_array_equal(shared.arrays[name], array)
        assert shared.arrays[name].ctypes.data % (2 << 20) == 0


def test_create_publishes_and_attach_reuses(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    tables = _tables()
    fill = _Fill(tables)
    first = esh.open_shared_tables("k1", _SHAPES, fill)
    opened.append(first)
    assert first.created and fill.calls == 1
    _assert_tables(first, tables)
    assert len(_records(registry, "k1")) == 1
    assert esh.complete_copy_exists("k1", _SHAPES)
    assert _mode(registry) == _mode(registry / "k1") == 0o700
    assert _mode(_records(registry, "k1")[0]) == 0o600

    second = esh.open_shared_tables("k1", _SHAPES, fill)
    opened.append(second)
    assert not second.created and fill.calls == 1
    assert second.address != first.address
    _assert_tables(second, tables)
    # The same pages: a write through one mapping shows in the other.
    first.arrays["b.weight"][0, 0] ^= 0xFF
    assert second.arrays["b.weight"][0, 0] == first.arrays["b.weight"][0, 0]
    assert len(_records(registry, "k1")) == 2

    second.close()
    assert len(_records(registry, "k1")) == 1


def test_register_called_per_process_mapping(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    calls: list[tuple[int, int]] = []
    unpinned: list[int] = []

    def register(address: int, size: int) -> Callable[[], None]:
        calls.append((address, size))
        return lambda: unpinned.append(address)

    first = esh.open_shared_tables(
        "k2", _SHAPES, _Fill(_tables()), register=register
    )
    second = esh.open_shared_tables(
        "k2", _SHAPES, _Fill(_tables()), register=register
    )
    assert calls == [(first.address, first.size), (second.address, second.size)]
    first.close()
    second.close()
    assert unpinned == [first.address, second.address]


_CHILD = textwrap.dedent(
    """
    import sys, numpy as np
    from max.pipelines.architectures.deepseekV41 import engram_shared_host as esh
    shapes = {shapes!r}
    rng = np.random.default_rng(0)
    tables = {{n: rng.integers(0, 256, size=s, dtype=np.uint8) for n, s in shapes.items()}}
    def fill(arrays):
        for n, a in arrays.items():
            np.copyto(a, tables[n])
    shared = esh.open_shared_tables("kx", shapes, fill)
    print("created" if shared.created else "attached", flush=True)
    sys.stdin.readline()
    """
)


def test_other_process_attaches_and_outlives_creator(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(shapes=_SHAPES)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline().strip() == "created"
        fill = _Fill(_tables())
        attached = esh.open_shared_tables("kx", _SHAPES, fill)
        opened.append(attached)
        assert not attached.created and fill.calls == 0
        _assert_tables(attached, _tables())
        child.stdin.write("\n")
        child.stdin.flush()
        assert child.wait(timeout=60) == 0
    finally:
        if child.poll() is None:
            child.kill()
    # The creator is gone; its (atexit-removed or stale) record no longer
    # serves, but this process's mapping keeps the copy alive and reachable.
    _assert_tables(attached, _tables())
    again = esh.open_shared_tables("kx", _SHAPES, _Fill(_tables(1)))
    opened.append(again)
    assert not again.created
    _assert_tables(again, _tables())


def _write_record(
    registry: Path, key: str, name: str, record: dict[str, int]
) -> Path:
    _private_mkdir(registry)
    _private_mkdir(registry / key)
    path = registry / key / f"provider-{name}.json"
    path.write_text(json.dumps(record))
    os.chmod(path, 0o600)
    return path


def test_dead_process_record_is_pruned(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _, size = esh.layout(_SHAPES)
    stale = _write_record(
        registry,
        "k3",
        "dead",
        {"pid": dead.pid, "fd": 3, "dev": 1, "ino": 1, "size": size},
    )
    garbage = registry / "k3" / "provider-garbage.json"
    garbage.write_text("{not json")
    fill = _Fill(_tables())
    shared = esh.open_shared_tables("k3", _SHAPES, fill)
    opened.append(shared)
    assert shared.created and fill.calls == 1
    assert not stale.exists() and not garbage.exists()


def test_reused_fd_number_is_not_attached(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    """Checks a record whose live pid's fd now names another file is skipped."""
    _, size = esh.layout(_SHAPES)
    with open(__file__) as other:
        stale = _write_record(
            registry,
            "k4",
            "reused",
            {
                "pid": os.getpid(),
                "fd": other.fileno(),
                "dev": 0,
                "ino": 0,
                "size": size,
            },
        )
        fill = _Fill(_tables())
        shared = esh.open_shared_tables("k4", _SHAPES, fill)
        opened.append(shared)
    assert shared.created and fill.calls == 1 and not stale.exists()


def _publish_raw_memfd(
    registry: Path, key: str, *, state: int, seal: bool, meta_key: str
) -> tuple[int, Path]:
    """Publishes a provider record for a hand-built memfd in a given state."""
    tables, size = esh.layout(_SHAPES)
    fd = esh._memfd_create("test")
    os.ftruncate(fd, size)
    address = esh._mmap(fd, size, esh._PROT_READ | esh._PROT_WRITE)
    esh._write_header(address, state, esh._metadata(meta_key, tables))
    esh._munmap(address, size)
    if seal:
        fcntl.fcntl(fd, esh._F_ADD_SEALS, esh._REQUIRED_SEALS)
    st = os.fstat(fd)
    path = _write_record(
        registry,
        key,
        f"raw-{fd}",
        {
            "pid": os.getpid(),
            "fd": fd,
            "dev": st.st_dev,
            "ino": st.st_ino,
            "size": size,
        },
    )
    return fd, path


@pytest.mark.parametrize(
    "state,seal,meta_key",
    [
        (esh._STATE_FILLING, True, "k5"),  # partial: never completed
        (esh._STATE_COMPLETE, False, "k5"),  # resizable: could SIGBUS
        (esh._STATE_COMPLETE, True, "other"),  # another checkpoint's copy
    ],
)
def test_partial_unsealed_or_foreign_copy_is_not_attached(
    registry: Path,
    opened: list[esh.SharedHostTables],
    state: int,
    seal: bool,
    meta_key: str,
) -> None:
    fd, record = _publish_raw_memfd(
        registry, "k5", state=state, seal=seal, meta_key=meta_key
    )
    try:
        fill = _Fill(_tables())
        shared = esh.open_shared_tables("k5", _SHAPES, fill)
        opened.append(shared)
        assert shared.created and fill.calls == 1
        assert not record.exists()
        _assert_tables(shared, _tables())
    finally:
        os.close(fd)


def test_layout_mismatch_is_not_attached(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    first = esh.open_shared_tables("k6", _SHAPES, _Fill(_tables()))
    opened.append(first)
    other_shapes = {**_SHAPES, "b.weight": (5, 3)}  # same size, other shape
    other = {**_tables(), "b.weight": np.zeros((5, 3), np.uint8)}
    second = esh.open_shared_tables("k6", other_shapes, _Fill(other))
    opened.append(second)
    assert second.created


def test_failed_fill_leaves_nothing(
    registry: Path, opened: list[esh.SharedHostTables]
) -> None:
    def broken(arrays: Mapping[str, np.ndarray]) -> None:
        raise OSError("checkpoint read failed")

    with pytest.raises(OSError):
        esh.open_shared_tables("k7", _SHAPES, broken)
    assert _records(registry, "k7") == []
    assert not esh.complete_copy_exists("k7", _SHAPES)
    fill = _Fill(_tables())
    shared = esh.open_shared_tables("k7", _SHAPES, fill)
    opened.append(shared)
    assert shared.created and fill.calls == 1


def test_failed_register_leaves_nothing(registry: Path) -> None:
    def register(address: int, size: int) -> Callable[[], None]:
        raise esh.SharedTablesUnavailable("no pinning")

    with pytest.raises(esh.SharedTablesUnavailable):
        esh.open_shared_tables(
            "k8", _SHAPES, _Fill(_tables()), register=register
        )
    assert _records(registry, "k8") == []


def test_concurrent_creator_lock_times_out(registry: Path) -> None:
    _private_mkdir(registry)
    _private_mkdir(registry / "k9")
    holder = os.open(registry / "k9" / "lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(esh.SharedTablesUnavailable, match="timed out"):
            esh.open_shared_tables(
                "k9", _SHAPES, _Fill(_tables()), lock_timeout_s=0.5
            )
    finally:
        os.close(holder)


def test_disabled_by_env(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(esh.DISABLE_ENV, "0")
    assert not esh.enabled()
    with pytest.raises(esh.SharedTablesUnavailable, match="disabled"):
        esh.open_shared_tables("k10", _SHAPES, _Fill(_tables()))


def test_registry_root_is_private_to_the_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(esh.DIR_ENV, raising=False)
    euid = os.geteuid()
    fallback = Path(f"/dev/shm/max-engram-shared-{euid}")
    runtime = _private_mkdir(tmp_path / "run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    assert esh.registry_dir() == runtime / "max-engram-shared"

    os.chmod(runtime, 0o755)
    assert esh.registry_dir() == fallback
    os.chmod(runtime, 0o700)

    link = tmp_path / "run-link"
    link.symlink_to(runtime, target_is_directory=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(link))
    assert esh.registry_dir() == fallback

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(os, "geteuid", lambda: euid + 1)
    assert esh.registry_dir() == Path(f"/dev/shm/max-engram-shared-{euid + 1}")

    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert esh.registry_dir() == Path(f"/dev/shm/max-engram-shared-{euid + 1}")


@pytest.mark.parametrize("which", ["root", "key"])
def test_symlinked_registry_dir_is_refused(
    registry: Path, tmp_path: Path, which: str
) -> None:
    target = _private_mkdir(tmp_path / "elsewhere")
    if which == "root":
        registry.symlink_to(target, target_is_directory=True)
    else:
        _private_mkdir(registry)
        (registry / "ks").symlink_to(target, target_is_directory=True)
    fill = _Fill(_tables())
    with pytest.raises(esh.SharedTablesUnavailable, match="not a directory"):
        esh.open_shared_tables("ks", _SHAPES, fill)
    assert fill.calls == 0
    assert not esh.complete_copy_exists("ks", _SHAPES)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "which,mode", [("root", 0o750), ("root", 0o701), ("key", 0o770)]
)
def test_registry_dir_open_to_others_is_refused(
    registry: Path, which: str, mode: int
) -> None:
    _private_mkdir(registry)
    path = registry if which == "root" else _private_mkdir(registry / "kp")
    os.chmod(path, mode)
    fill = _Fill(_tables())
    with pytest.raises(esh.SharedTablesUnavailable, match="not a directory"):
        esh.open_shared_tables("kp", _SHAPES, fill)
    assert fill.calls == 0
    assert not esh.complete_copy_exists("kp", _SHAPES)
    assert _mode(path) == mode  # refused, not repaired


def test_registry_of_another_user_is_refused(
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened: list[esh.SharedHostTables],
) -> None:
    shared = esh.open_shared_tables("ku", _SHAPES, _Fill(_tables()))
    opened.append(shared)
    (path,) = _records(registry, "ku")
    record = esh._read_record(path)
    assert record is not None
    fd = esh._open_provider(record, shared.size, os.O_RDONLY)
    assert fd is not None
    os.close(fd)

    # Seen as another user, every directory, record and provider process
    # here belongs to someone else.
    other = os.geteuid() + 1
    monkeypatch.setattr(os, "geteuid", lambda: other)
    fill = _Fill(_tables())
    with pytest.raises(esh.SharedTablesUnavailable, match="not a directory"):
        esh.open_shared_tables("ku", _SHAPES, fill)
    assert fill.calls == 0
    assert not esh.complete_copy_exists("ku", _SHAPES)
    assert esh._read_record(path) is None
    assert esh._open_provider(record, shared.size, os.O_RDONLY) is None


@pytest.mark.parametrize("kind", ["group_writable", "symlink"])
def test_untrusted_record_is_not_attached(
    registry: Path,
    tmp_path: Path,
    opened: list[esh.SharedHostTables],
    kind: str,
) -> None:
    first = esh.open_shared_tables("kr", _SHAPES, _Fill(_tables()))
    opened.append(first)
    (good,) = _records(registry, "kr")
    content = good.read_text()
    first._unpublish()
    # The record names a live, complete copy, so only its file's
    # ownership and mode keep it from being attached.
    bad = registry / "kr" / "provider-bad.json"
    if kind == "group_writable":
        bad.write_text(content)
        os.chmod(bad, 0o620)
    else:
        outside = tmp_path / "outside.json"
        outside.write_text(content)
        os.chmod(outside, 0o600)
        bad.symlink_to(outside)
    fill = _Fill(_tables())
    second = esh.open_shared_tables("kr", _SHAPES, fill)
    opened.append(second)
    assert second.created and fill.calls == 1
    assert not bad.is_symlink() and not bad.exists()
    if kind == "symlink":
        assert outside.read_text() == content


def _checkpoint_views(path: Path) -> dict[str, np.ndarray]:
    tables = _tables()
    blob = b"".join(t.tobytes() for t in tables.values())
    path.write_bytes(b"\0" * 100 + blob)
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    views, offset = {}, 100
    for name, array in tables.items():
        views[name] = mapped[offset : offset + array.nbytes].reshape(
            array.shape
        )
        offset += array.nbytes
    return views


def test_source_key_tracks_the_checkpoint_file(tmp_path: Path) -> None:
    path = tmp_path / "shard.safetensors"
    views = _checkpoint_views(path)
    key = esh.source_key(views)
    assert key is not None and key == esh.source_key(views)
    os.utime(path, ns=(1, 1))
    assert esh.source_key(views) not in (None, key)
    assert esh.source_key(_tables()) is None  # anonymous: not shareable


class _Registrar:
    read_only = False

    def __init__(self, device_ordinal: int) -> None:
        pass

    def __call__(self, address: int, size: int) -> Callable[[], None]:
        return lambda: None


class _Device:
    id = 0


def test_model_shares_file_backed_tables(
    registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    views = _checkpoint_views(tmp_path / "shard.safetensors")
    monkeypatch.setattr(esh, "CudaHostRegistrar", _Registrar)
    first = model_mod._open_shared_engram_tables(views, _Device())  # type: ignore[arg-type]
    second = model_mod._open_shared_engram_tables(views, _Device())  # type: ignore[arg-type]
    assert first is not None and second is not None
    try:
        assert first.created and not second.created
        _assert_tables(second, _tables())
    finally:
        first.close()
        second.close()


def test_model_falls_back_to_private_copy(
    registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not file-backed: no key, nothing to share.
    assert model_mod._open_shared_engram_tables(_tables(), _Device()) is None  # type: ignore[arg-type]

    # File-backed, but pinning is unavailable (no CUDA here, or a stub that
    # refuses): the loader pins a private copy instead of failing.
    def refuse(device_ordinal: int) -> NoReturn:
        raise esh.SharedTablesUnavailable("no CUDA")

    monkeypatch.setattr(esh, "CudaHostRegistrar", refuse)
    views = _checkpoint_views(tmp_path / "shard.safetensors")
    assert model_mod._open_shared_engram_tables(views, _Device()) is None  # type: ignore[arg-type]
    assert not list(registry.glob("*/provider-*.json"))

    monkeypatch.setenv(esh.DISABLE_ENV, "0")
    assert model_mod._open_shared_engram_tables(views, _Device()) is None  # type: ignore[arg-type]


def test_model_falls_back_on_untrusted_registry(
    registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(esh, "CudaHostRegistrar", _Registrar)
    _private_mkdir(registry)
    os.chmod(registry, 0o755)
    views = _checkpoint_views(tmp_path / "shard.safetensors")
    assert model_mod._open_shared_engram_tables(views, _Device()) is None  # type: ignore[arg-type]
    assert not list(registry.iterdir())


def _resident(path: Path) -> list[bool]:
    """Returns, per page of ``path``, whether it is in the page cache."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    size = path.stat().st_size
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    vec = ctypes.create_string_buffer(-(-size // mmap.PAGESIZE))
    assert libc.mincore(mapped.ctypes.data, size, vec) == 0
    return [bool(b & 1) for b in vec.raw]


def _cache(path: Path) -> None:
    with open(path, "rb") as f:
        f.read()


def test_checkpoint_page_cache_is_dropped(tmp_path: Path) -> None:
    pages = 64
    path = tmp_path / "shard.bin"
    path.write_bytes(os.urandom(pages * mmap.PAGESIZE))
    os.sync()
    _cache(path)
    if not all(_resident(path)):
        pytest.skip("this filesystem does not cache the file")
    weight_adapters.drop_page_cache([path])
    if any(_resident(path)):
        pytest.skip("this filesystem ignores POSIX_FADV_DONTNEED (tmpfs)")

    # Exactly the given range's pages leave the page cache, even when this
    # process never faulted them in through its mapping (MADV_PAGEOUT alone
    # would skip them).
    _cache(path)
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    view = mapped[8 * mmap.PAGESIZE : 24 * mmap.PAGESIZE]
    start = view.ctypes.data
    assert weight_adapters._in_readonly_file_mapping(start, start + view.nbytes)
    weight_adapters._drop_file_pages(start, start + view.nbytes)
    resident = _resident(path)
    assert not any(resident[8:24])
    assert all(resident[:8]) and all(resident[24:])
