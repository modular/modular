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

"""Resolve the kv_cache_hash_seed from operator config.

The CLI / config layer carries the seed as an optional 64-character hex
string. This module decodes it to 32 raw bytes, validates length, and
when one is required but not provided, generates a random 32-byte seed
that is cached for the lifetime of the process. The active seed hex is
logged exactly once on first resolution.
"""

from __future__ import annotations

import logging
import secrets
import threading

from .block_utils import KVHashAlgo

logger = logging.getLogger("max.pipelines.kv_cache")

_lock = threading.Lock()
_cached_random_seed: bytes | None = None
_seed_logged: bool = False


def resolve_kv_hash_seed(
    algo: KVHashAlgo,
    seed_hex: str | None,
) -> bytes | None:
    """Resolve the operator-supplied kv_cache_hash_seed.
    Args:
        algo: The selected hash algorithm.
        seed_hex: Optional 64-character hex string (32 bytes after decode).
    Returns:
        - ``None`` for ``ahash64`` (the legacy hasher does not use a seed).
        - 32 raw bytes for ``sha256`` / ``sha256_64``. If ``seed_hex`` is
          ``None``, a random seed is generated once per process and reused
          on subsequent calls.
    Raises:
        ValueError: If ``seed_hex`` is provided but is not a valid
            64-character hex string decoding to exactly 32 bytes.
    """
    if algo == "ahash64":
        if seed_hex is not None:
            logger.warning(
                "kv_cache_hash_seed=%r ignored because "
                "kv_cache_hash_algo=ahash64.",
                seed_hex,
            )
        return None
    if seed_hex is None:
        return _get_or_create_random_seed()

    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise ValueError(
            f"kv_cache_hash_seed must be a hex string; got {seed_hex!r}"
        ) from exc
    if len(seed) != 32:
        raise ValueError(
            f"kv_cache_hash_seed must decode to exactly 32 bytes; "
            f"got {len(seed)} bytes from {seed_hex!r}"
        )
    _log_active_seed_once(seed, generated=False)
    return seed


def _get_or_create_random_seed() -> bytes:
    global _cached_random_seed
    with _lock:
        if _cached_random_seed is None:
            _cached_random_seed = secrets.token_bytes(32)
    _log_active_seed_once(_cached_random_seed, generated=True)
    return _cached_random_seed


def _log_active_seed_once(seed: bytes, *, generated: bool) -> None:
    global _seed_logged
    with _lock:
        if _seed_logged:
            return
        _seed_logged = True
    logger.info(
        "Active KV-cache hash seed: %s (%s)",
        seed.hex(),
        "auto-generated" if generated else "from config",
    )
