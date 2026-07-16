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

"""End-to-end tests for sha256 hashing through BlockManager.

Exercises the kv_hash_algo / kv_hash_seed / cache_salt plumbing added to
BlockManager.compute_hashes_for_request:

- sha256 produces 32-byte bytes hashes per block.
- Identical tokens + identical seed/salt => identical hash chain (cache hit).
- Different cache_salt => different hash chain (multi-tenant isolation).
- Different kv_hash_seed => different hash chain (cluster isolation).
- kv_hash_algo="ahash64" (default) still yields int hashes (no regression).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.connectors.dkv.connector import DKVConnector
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVHashAlgo
from max.pipelines.modeling.types import RequestID


def _make_ctx(
    tokens: np.ndarray,
    request_id: RequestID = RequestID("req-1"),
    *,
    cache_salt: str | None = None,
) -> TextContext:
    """Build a minimal TextGenerationContext-like stub.
    BlockManager.compute_hashes_for_request accesses ``ctx.request_id``,
    ``len(ctx.tokens)``, ``ctx.tokens[i:j]``, ``ctx.images`` (via an
    ``isinstance`` check that fails for SimpleNamespace), and
    ``ctx.cache_salt`` (direct attribute access — the real ``TextContext``
    always defines this attribute, so the stub must too, even when no
    caller-supplied salt is set).
    """
    ctx = SimpleNamespace(
        request_id=request_id,
        tokens=tokens,
        cache_salt=cache_salt,
    )
    return cast(TextContext, ctx)


def _make_block_manager(
    *,
    block_size: int = 8,
    total_blocks: int = 32,
    kv_hash_algo: KVHashAlgo = "ahash64",
    kv_hash_seed: bytes | None = None,
) -> BlockManager:
    return BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=total_blocks,
        block_size=block_size,
        connector=cast(object, NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=True,
        kv_hash_algo=kv_hash_algo,
        kv_hash_seed=kv_hash_seed,
    )


def test_sha256_produces_32_byte_hashes() -> None:
    bm = _make_block_manager(kv_hash_algo="sha256")
    # 33 tokens => 32 hashable (last reserved) => 4 full blocks of 8.
    tokens = np.arange(33, dtype=np.int32)
    ctx = _make_ctx(tokens)

    bm.compute_hashes_for_request(ctx)

    hashes = bm.req_to_hashes[ctx.request_id]
    assert len(hashes) == 4
    for h in hashes:
        assert isinstance(h, bytes)
        assert len(h) == 32


def test_sha256_same_tokens_same_hashes() -> None:
    """Two identical contexts produce identical hash chains (cache-hit potential)."""
    tokens = np.arange(33, dtype=np.int32)

    bm1 = _make_block_manager(kv_hash_algo="sha256")
    bm1.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-A")))

    bm2 = _make_block_manager(kv_hash_algo="sha256")
    bm2.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-B")))

    assert (
        bm1.req_to_hashes[RequestID("req-A")]
        == bm2.req_to_hashes[RequestID("req-B")]
    )


def test_sha256_salt_isolation() -> None:
    """Same tokens + different cache_salt => disjoint hashes (multi-tenant safety)."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager(kv_hash_algo="sha256")

    bm.compute_hashes_for_request(
        _make_ctx(tokens, RequestID("req-tenant-A"), cache_salt="tenant-A")
    )
    bm.compute_hashes_for_request(
        _make_ctx(tokens, RequestID("req-tenant-B"), cache_salt="tenant-B")
    )

    a = bm.req_to_hashes[RequestID("req-tenant-A")]
    b = bm.req_to_hashes[RequestID("req-tenant-B")]
    assert a != b
    assert set(a).isdisjoint(set(b))


def test_sha256_seed_isolation() -> None:
    """Same tokens + different kv_hash_seed => disjoint hashes (cluster isolation)."""
    tokens = np.arange(33, dtype=np.int32)

    bm1 = _make_block_manager(kv_hash_algo="sha256", kv_hash_seed=b"\x00" * 32)
    bm1.compute_hashes_for_request(_make_ctx(tokens))

    bm2 = _make_block_manager(kv_hash_algo="sha256", kv_hash_seed=b"\x01" * 32)
    bm2.compute_hashes_for_request(_make_ctx(tokens))

    a = bm1.req_to_hashes[RequestID("req-1")]
    b = bm2.req_to_hashes[RequestID("req-1")]
    assert a != b
    assert set(a).isdisjoint(set(b))


def test_ahash64_default_unchanged() -> None:
    """Default kv_hash_algo yields list[int]; legacy path unchanged."""
    bm = _make_block_manager()  # default = ahash64

    tokens = np.arange(33, dtype=np.int32)
    bm.compute_hashes_for_request(_make_ctx(tokens))

    hashes = bm.req_to_hashes[RequestID("req-1")]
    assert len(hashes) == 4
    for h in hashes:
        assert isinstance(h, int)


def test_ahash64_with_cache_salt_drops_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Under ahash64, a request-supplied cache_salt is dropped (not hashed in)
    and the BlockManager emits exactly one warning per process for the entire
    deployment, no matter how many salted requests arrive.

    This guards the operator-visible policy decision: ahash64 deployments do
    NOT silently provide multi-tenant isolation; they advertise via a one-time
    warning that cache_salt is inert, and produce identical hashes for
    identical tokens regardless of salt (so prefix-cache hit rates remain
    intact for the legacy path)."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager()  # default ahash64

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        bm.compute_hashes_for_request(
            _make_ctx(tokens, RequestID("req-A"), cache_salt="tenant-A")
        )
        bm.compute_hashes_for_request(
            _make_ctx(tokens, RequestID("req-B"), cache_salt="tenant-B")
        )
        bm.compute_hashes_for_request(
            _make_ctx(tokens, RequestID("req-C"), cache_salt="tenant-C")
        )

    a = bm.req_to_hashes[RequestID("req-A")]
    b = bm.req_to_hashes[RequestID("req-B")]
    c = bm.req_to_hashes[RequestID("req-C")]
    assert a == b == c, (
        "ahash64 must ignore cache_salt; identical tokens => identical hashes"
    )

    matching = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "cache_salt was supplied" in r.message
    ]
    assert len(matching) == 1, (
        f"expected exactly one cache_salt-dropped warning across three "
        f"salted requests, got {len(matching)}"
    )


def test_ahash64_without_cache_salt_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If no request supplies a cache_salt, the warning must not fire."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager()  # default ahash64

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        bm.compute_hashes_for_request(_make_ctx(tokens))

    matching = [
        r for r in caplog.records if "cache_salt was supplied" in r.message
    ]
    assert matching == [], (
        f"unexpected salt-dropped warning when no salt was supplied: "
        f"{[r.message for r in matching]}"
    )


# ---------------------------------------------------------------------------
# Connector capability matrix
# ---------------------------------------------------------------------------


class _StubConnector:
    """Minimal :class:`KVConnector`-shaped stub for capability-gate tests.

    BlockManager construction only reads ``supported_hash_algos`` and
    ``name`` during the capability check, so we don't need to implement the
    full protocol surface here. The ``load`` / ``offload`` methods raise on
    use to fail loudly if any code path unexpectedly invokes them.
    """

    def __init__(
        self,
        *,
        supported_hash_algos: frozenset[KVHashAlgo],
        num_host_blocks: int = 0,
    ) -> None:
        self._supported = supported_hash_algos
        self._num_host_blocks = num_host_blocks

    @property
    def name(self) -> str:
        return "StubConnector"

    @property
    def num_host_blocks(self) -> int:
        return self._num_host_blocks

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        return self._supported

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
    ) -> int:
        raise NotImplementedError("StubConnector.load must not be called")

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
    ) -> None:
        raise NotImplementedError("StubConnector.offload must not be called")


_LEGACY: frozenset[KVHashAlgo] = frozenset({"ahash64"})
_FULL: frozenset[KVHashAlgo] = frozenset({"ahash64", "sha256", "sha256_64"})


@pytest.mark.parametrize(
    ("algo", "supported", "should_pass"),
    [
        ("ahash64", _LEGACY, True),
        ("ahash64", _FULL, True),
        ("sha256", _LEGACY, False),
        ("sha256", _FULL, True),
        ("sha256_64", _LEGACY, False),
        ("sha256_64", _FULL, True),
    ],
    ids=[
        "ahash64-on-legacy",
        "ahash64-on-full",
        "sha256-on-legacy-rejected",
        "sha256-on-full",
        "sha256_64-on-legacy-rejected",
        "sha256_64-on-full",
    ],
)
def test_block_manager_capability_guard(
    algo: KVHashAlgo,
    supported: frozenset[KVHashAlgo],
    should_pass: bool,
) -> None:
    """BlockManager refuses to start when ``kv_hash_algo`` is unsupported.

    Exercises the capability check that replaced the legacy ahash64-only
    guard: BlockManager must accept every algo declared in
    ``connector.supported_hash_algos`` and reject every one outside it,
    regardless of the connector's ``num_host_blocks`` (the legacy guard
    skipped this for offload-less connectors).
    """
    connector = _StubConnector(
        supported_hash_algos=supported, num_host_blocks=4
    )

    def _construct() -> BlockManager:
        return BlockManager(
            device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
            total_num_blocks=32,
            block_size=8,
            connector=cast(object, connector),  # type: ignore[arg-type]
            enable_prefix_caching=True,
            kv_hash_algo=algo,
        )

    if should_pass:
        bm = _construct()
        assert bm.kv_hash_algo == algo
    else:
        with pytest.raises(ValueError, match="not supported by"):
            _construct()


def test_block_manager_capability_check_runs_even_without_host_blocks() -> None:
    """Legacy guard only fired when ``num_host_blocks > 0``; the capability
    check must run unconditionally so a no-host-block connector still
    refuses an algo it does not claim to support.
    """
    connector = _StubConnector(supported_hash_algos=_LEGACY, num_host_blocks=0)
    with pytest.raises(ValueError, match="not supported by"):
        BlockManager(
            device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
            total_num_blocks=32,
            block_size=8,
            connector=cast(object, connector),  # type: ignore[arg-type]
            enable_prefix_caching=True,
            kv_hash_algo="sha256",
        )


# ---------------------------------------------------------------------------
# Real-connector declared capabilities
# ---------------------------------------------------------------------------


def test_null_connector_supports_all_algos() -> None:
    """NullConnector is the no-op host tier and must accept every algo."""
    assert NullConnector().supported_hash_algos == _FULL


def test_local_and_tiered_connectors_declare_full_sha256_support() -> None:
    """Lock the host-tier connectors' declared capabilities at the class
    level. Both rely on numpy-keyed dicts so they natively handle the
    ``bytes`` SHA-256 hashes alongside ``int`` ahash64 hashes.
    """
    # Both classes expose ``supported_hash_algos`` as a property; read it
    # off the descriptor to avoid constructing real KV memory buffers.
    for cls in (LocalConnector, TieredConnector):
        prop = inspect.getattr_static(cls, "supported_hash_algos")
        assert isinstance(prop, property), (
            f"{cls.__name__}.supported_hash_algos must be a property"
        )
        # The property body is a single ``return frozenset({...})`` literal,
        # so calling ``fget`` against ``None`` is unsafe. Instead, assert
        # the literal source matches the expected set via a smoke roundtrip
        # through a fresh subclass instance with __init__ patched out.
        instance = cls.__new__(cls)
        assert prop.fget is not None
        assert prop.fget(instance) == _FULL


# ---------------------------------------------------------------------------
# DKV connector capability wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["ahash64", "sha256", "sha256_64"])
def test_block_manager_accepts_dkv_advertised_algos(
    algo: KVHashAlgo,
) -> None:
    """BlockManager accepts every algo the dkv connector advertises.

    Pins the wiring between :attr:`DKVConnector.supported_hash_algos` (now
    extended to accept full SHA-256 via boundary truncation, see
    ``max/python/max/pipelines/kv_cache/connectors/dkv/connector.py``) and
    the BlockManager capability gate. Skips ``__init__`` so no real dkv
    client is constructed.
    """
    dkv_advertised = DKVConnector.__new__(DKVConnector).supported_hash_algos
    assert algo in dkv_advertised, (
        f"plan invariant: dkv must advertise {algo}; got {dkv_advertised}"
    )

    connector = _StubConnector(
        supported_hash_algos=dkv_advertised, num_host_blocks=4
    )
    bm = BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=32,
        block_size=8,
        connector=cast(object, connector),  # type: ignore[arg-type]
        enable_prefix_caching=True,
        kv_hash_algo=algo,
    )
    assert bm.kv_hash_algo == algo
