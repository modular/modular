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

"""Disk-tier ``kv-disk-cache.meta.json`` algorithm lock-in tests.

The ``DiskTier`` persists block-keyed files under a per-deployment cache
directory. Mixing ``ahash64`` (16-hex u64) and ``sha256`` (64-hex digest)
filenames in the same directory would silently produce cache misses or
collisions, so ``DiskTier._verify_or_record_algo`` writes (or validates)
a ``kv-disk-cache.meta.json`` sidecar on startup and refuses to start
against a directory locked to a different algorithm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from max.pipelines.kv_cache.connectors.disk_tier import DiskTier
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes

_META = "kv-disk-cache.meta.json"
_BLOCK_NBYTES = 4096
_MAX_DISK_BYTES = 1 << 30  # 1 GiB; plenty for tests that never write blocks.


def _make_disk_tier(cache_dir: Path, kv_hash_algo: str) -> DiskTier:
    """Construct a DiskTier rooted at ``cache_dir`` with minimal workers.

    Uses a single worker thread so test teardown doesn't leave a wide
    PriorityExecutor pool spinning across the file's tests.
    """
    return DiskTier(
        cache_dir=str(cache_dir),
        block_nbytes=_BLOCK_NBYTES,
        max_disk_size_bytes=_MAX_DISK_BYTES,
        kv_hash_algo=kv_hash_algo,  # type: ignore[arg-type]
        num_workers=1,
    )


def _shutdown(disk_tier: DiskTier) -> None:
    """Best-effort teardown of the executor created in DiskTier.__init__."""
    disk_tier._executor.shutdown(wait=True)


def test_fresh_dir_writes_meta_with_configured_algo(tmp_path: Path) -> None:
    """First start in an empty dir writes the algo lock-in sidecar."""
    dt = _make_disk_tier(tmp_path, "ahash64")
    try:
        meta_path = tmp_path / _META
        assert meta_path.exists()
        assert json.loads(meta_path.read_text()) == {"hash_algo": "ahash64"}
    finally:
        _shutdown(dt)


def test_reopen_with_matching_algo_succeeds(tmp_path: Path) -> None:
    """Reopening a dir with the same algo as recorded in meta succeeds."""
    first = _make_disk_tier(tmp_path, "sha256")
    _shutdown(first)
    second = _make_disk_tier(tmp_path, "sha256")
    try:
        assert second._hash_algo == "sha256"
    finally:
        _shutdown(second)


def test_reopen_with_mismatched_algo_raises(tmp_path: Path) -> None:
    """Reopening with a different algo than recorded raises with remediation."""
    first = _make_disk_tier(tmp_path, "ahash64")
    _shutdown(first)
    with pytest.raises(RuntimeError) as excinfo:
        _make_disk_tier(tmp_path, "sha256")
    msg = str(excinfo.value)
    assert "ahash64" in msg
    assert "sha256" in msg
    assert str(tmp_path) in msg
    assert "Delete" in msg


def test_preexisting_16hex_files_with_sha256_config_raises(
    tmp_path: Path,
) -> None:
    """Legacy ahash64 cache (16-hex stems) must not be reopened as sha256."""
    (tmp_path / f"{0xDEADBEEFCAFEBABE:016x}.bin").write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError) as excinfo:
        _make_disk_tier(tmp_path, "sha256")
    msg = str(excinfo.value)
    assert "16-char" in msg or "int-hash" in msg
    assert str(tmp_path) in msg


def test_preexisting_64hex_files_with_ahash64_config_raises(
    tmp_path: Path,
) -> None:
    """sha256 cache (64-hex stems) must not be reopened as ahash64."""
    digest_hex = "ab" * 32  # 64 hex chars
    (tmp_path / f"{digest_hex}.bin").write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError) as excinfo:
        _make_disk_tier(tmp_path, "ahash64")
    msg = str(excinfo.value)
    assert "SHA-256" in msg or "64-char" in msg
    assert str(tmp_path) in msg


def test_preexisting_16hex_files_with_ahash64_config_records_meta(
    tmp_path: Path,
) -> None:
    """A legacy cache (no meta, 16-hex files) is adopted when the configured
    algo is compatible. The meta file is written atomically on startup so
    future starts skip the inference branch.
    """
    (tmp_path / f"{0xDEADBEEFCAFEBABE:016x}.bin").write_bytes(b"\x00" * 8)
    dt = _make_disk_tier(tmp_path, "ahash64")
    try:
        meta_path = tmp_path / _META
        assert meta_path.exists()
        assert json.loads(meta_path.read_text()) == {"hash_algo": "ahash64"}
    finally:
        _shutdown(dt)


def test_preexisting_64hex_files_with_sha256_config_records_meta(
    tmp_path: Path,
) -> None:
    """A pre-meta sha256 cache (64-hex files) is adopted on a sha256 start."""
    digest_hex = "cd" * 32
    (tmp_path / f"{digest_hex}.bin").write_bytes(b"\x00" * 8)
    dt = _make_disk_tier(tmp_path, "sha256")
    try:
        meta_path = tmp_path / _META
        assert meta_path.exists()
        assert json.loads(meta_path.read_text()) == {"hash_algo": "sha256"}
    finally:
        _shutdown(dt)


def test_corrupt_meta_raises_with_remediation(tmp_path: Path) -> None:
    """A malformed meta.json must abort startup with a clear remediation."""
    (tmp_path / _META).write_text("{ this is not json")
    with pytest.raises(RuntimeError) as excinfo:
        _make_disk_tier(tmp_path, "ahash64")
    msg = str(excinfo.value)
    assert str(tmp_path) in msg
    assert "Delete" in msg


def test_sha256_64_negative_int_hash_filename_is_canonical(
    tmp_path: Path,
) -> None:
    """Regression test for the connector-boundary signed-BE encoding.

    sha256_64 truncates a 32-byte SHA-256 digest to 8 bytes; for roughly
    half of all inputs the resulting 8-byte hash, viewed as a signed
    int64, is negative. The boundary coercion in
    ``kv_connector.to_block_hash_bytes`` uses ``signed=True`` so the
    encoding is total over the full int64 range. Without ``signed=True``,
    a negative int hash would either raise ``OverflowError``
    (``signed=False``) or alias onto a different bit-pattern, silently
    producing cache misses against on-disk files written under the
    canonical filename.

    This test pins down (a) the helper's encoding for the int64 boundary
    sentinels (-1, INT64_MIN, INT64_MAX, 0) and the 8-/32-byte
    pass-through cases, and (b) that the DiskTier maps the encoded bytes
    to the matching hex filename — i.e. the on-disk schema that
    ``_load_existing`` has to round-trip across restarts.
    """
    # int64-boundary sentinels: the encoding must be total and lossless.
    assert to_block_hash_bytes(-1) == b"\xff" * 8
    assert to_block_hash_bytes(-(1 << 63)) == b"\x80" + b"\x00" * 7
    assert to_block_hash_bytes((1 << 63) - 1) == b"\x7f" + b"\xff" * 7
    assert to_block_hash_bytes(0) == b"\x00" * 8

    # Bytes inputs of canonical lengths pass through unchanged.
    assert to_block_hash_bytes(b"\xff" * 8) == b"\xff" * 8
    assert to_block_hash_bytes(b"\xab" * 32) == b"\xab" * 32

    # The DiskTier maps the encoded bytes to the matching hex filename;
    # this is what ``_load_existing`` has to round-trip across restarts.
    dt = _make_disk_tier(tmp_path, "sha256_64")
    try:
        negative_hash = to_block_hash_bytes(-1)
        assert dt._hash_to_path(negative_hash).name == "ffffffffffffffff.bin"
    finally:
        _shutdown(dt)
