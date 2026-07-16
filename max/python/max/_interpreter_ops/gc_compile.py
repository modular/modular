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

"""Shared compile-mode state for the eager interpreter's GC model caches.

The per-op-type GC caches (``matmul_gc``, ``unary_elementwise_gc``, and any
future one) share process-wide singletons defined here:

- :func:`should_precompile` — whether to compile the full GC matrix at import
  or compile each target lazily on first dispatch (the default).
- :data:`COMPILE_LOCK` — serializes the lazy check-compile-cache so concurrent
  first-dispatches don't race.
- :func:`session_for` — a per-device :class:`~max.engine.InferenceSession`
  cache, shared across op types so lazy compiles reuse one session per device.
- :func:`write_warm_stamp` / :func:`warm_stamp_matches` — marker letting a
  separate lazy process adopt a batched warm instead of compiling target-by-target.

This module must not import from ``handlers.py``.
"""

import json
import os
import platform
import threading
from pathlib import Path
from typing import Any

from max import engine
from max.driver import (
    Device,
    accelerator_architecture_name,
    accelerator_count,
)

# Lazy-per-dispatch by default; ``=1`` precompiles the whole matrix at import
# (MXF-508).
EAGER_OP_PRECOMPILE_ENV_VAR = "MAX_EAGER_OP_PRECOMPILE"

# Stored in the MEF cache dir (see _cache_dir) so it can't outlive the artifacts
# it vouches for.
_WARM_STAMP_NAME = "eager_gc_warm_stamp.json"


def should_precompile() -> bool:
    """Returns whether to precompile the full GC matrix at import.

    Read at call time, not import time: the sweep runs from ``__init__``, which
    may be imported before a launcher or test harness sets the env var.
    """
    return os.environ.get(EAGER_OP_PRECOMPILE_ENV_VAR, "0") == "1"


def _derived_root() -> Path | None:
    """The ``MODULAR_DERIVED_PATH`` root, or None if unset."""
    derived = os.environ.get("MODULAR_DERIVED_PATH")
    return Path(derived) if derived else None


def _cache_dir() -> Path | None:
    """MEF cache dir the warm stamp lives in, or None if unset.

    Keyed off ``MODULAR_DERIVED_PATH`` — the redirect knob warmer and consumer
    both set to agree on location (matching ``tools/interpreter_warm_cache``);
    unset → no stamp → lazy. A config-file ``cache_dir`` would still win in the
    engine (GEX-3884).
    """
    root = _derived_root()
    return root / "cache" / ".max_cache" if root else None


def _context_signature() -> str:
    """Signature a warm must match before a lazy process adopts it.

    Pins host arch + accelerator count + SKU. ``accelerator_architecture_name``
    raises on a CPU device, so it's only queried when an accelerator is present.
    The device *count* is the leading field and is matched as a ceiling, not for
    equality, by :func:`warm_stamp_matches`.
    """
    n = accelerator_count()
    accel = accelerator_architecture_name() if n else ""
    return f"accelerators={n};cpu={platform.machine()};accel={accel}"


def _split_device_count(signature: str) -> tuple[int, str]:
    """Split a context signature into ``(device count, host+accelerator SKU)``.

    The SKU (everything after the leading ``accelerators=N`` field) must match
    exactly between warm and consumer; the count is compared as a ceiling.

    Example:
        ``"accelerators=8;cpu=x86_64;accel=sm_100a"`` ->
        ``(8, "cpu=x86_64;accel=sm_100a")``.
    """
    head, _, sku = signature.partition(";")
    return int(head.removeprefix("accelerators=")), sku


def write_warm_stamp() -> bool:
    """Records a batched warm for this context. Returns False if the cache dir
    can't be located (the warm is then unadoptable; caller should surface it)."""
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / _WARM_STAMP_NAME).write_text(
        json.dumps({"context": _context_signature()})
    )
    return True


def warm_stamp_matches() -> bool:
    """Returns whether a warm stamp this process can adopt is present.

    Requires the same host + accelerator SKU, and that this process needs no
    *more* devices than were warmed. Per-slot single-graph MEFs are device-
    count-independent, so a warm made for N devices serves any consumer with
    ``<= N`` devices (it reuses slots ``0..k-1``); a consumer needing more
    devices than were warmed must not adopt (the extra slots were never
    compiled) and falls back to per-target lazy compilation.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    try:
        stamp = json.loads((cache_dir / _WARM_STAMP_NAME).read_text())
        warmed_count, warmed_sku = _split_device_count(stamp["context"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    current_count, current_sku = _split_device_count(_context_signature())
    return current_sku == warmed_sku and current_count <= warmed_count


_MANIFEST_NAME = "manifest.json"
_ADOPT_ASSERTED_ENV = "MODULAR_EAGER_WARM_ADOPT_ASSERTED"


def _manifest_path() -> Path | None:
    """Path to the force-load manifest in MODULAR_DERIVED_PATH, or None."""
    root = _derived_root()
    return root / _MANIFEST_NAME if root else None


def write_manifest(
    envelope: dict[str, object], entries: list[dict[str, str]]
) -> bool:
    """Write the force-load manifest. Returns False if MODULAR_DERIVED_PATH is
    unset (nowhere to write it)."""
    path = _manifest_path()
    if path is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"envelope": envelope, "entries": entries}, indent=2)
    )
    return True


def read_manifest() -> dict[str, Any] | None:
    """Read the force-load manifest, or None if absent/unreadable."""
    path = _manifest_path()
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def manifest_adoptable(manifest: dict[str, Any]) -> bool:
    """Whether this process may force-load from the manifest.

    Two manifest kinds, told apart by the envelope's ``gpu`` key:

    - **GPU manifest** (``gpu`` present): adoptable on a box whose accelerator
      arch matches ``gpu.arch`` and whose device count is *at most* the warmed
      count. The manifest indexes one MEF per device slot (a CPU MEF, a gpu:0
      MEF, a gpu:1 MEF; not one combined MEF), so a consumer needing ``k``
      accelerators force-loads slots ``0..k-1``: a warm made for ``N >= k``
      slots adopts (extras unused), but one for fewer than ``k`` cannot (those
      slots were never compiled), so device count is a ceiling, not equality.
    - **CPU-only manifest** (``gpu`` absent/None): adoptable only on a box with
      no accelerator (``accelerator_count() == 0``). There is no GPU slot to
      match, and a box that *does* have an accelerator would need GPU MEFs this
      warm never built.

    Both kinds also require the asserted-toolchain opt-in (closed-loop CI only),
    an asserted-mode envelope, and an *exact* host-arch match: a force-loaded
    MEF's embedded host-ELF kernels are ABI-specific to the host arch (the
    compiled CPU *target* is a floor, so a coarse host-arch match suffices).
    """
    if os.environ.get(_ADOPT_ASSERTED_ENV) != "1":
        return False
    envelope = manifest.get("envelope", {})
    if envelope.get("toolchain", {}).get("mode") != "asserted":
        return False
    # Coarse: platform.machine() misses cpu_target's microarch level; a sub-v3
    # x86 host could adopt v3 kernels and SIGILL. A self-enforcing vN check is
    # deferred, level detection is per-arch (x86 flags; aarch64 core map).
    if envelope.get("host_arch") != platform.machine():
        return False
    gpu = envelope.get("gpu")
    n = accelerator_count()
    if gpu is None:
        # CPU-only manifest: no GPU slot to match, so adoptable only where there
        # is no accelerator that would need GPU MEFs this warm never built.
        return n == 0
    # Device-count ceiling for the manifest path (the stamp path enforces the
    # same via warm_stamp_matches). Default 0 rejects a manifest missing
    # device_count (0 < n) instead of a None comparison.
    if n == 0 or envelope.get("device_count", 0) < n:
        return False
    return gpu.get("arch") == accelerator_architecture_name()


def manifest_entry_path(
    manifest: dict[str, Any], family: str, device_class: str
) -> Path | None:
    """Absolute path to the MEF for (family, device_class), or None if the
    manifest has no such entry."""
    root = _derived_root()
    if root is None:
        return None
    for entry in manifest.get("entries", []):
        if (
            entry.get("family") == family
            and entry.get("device_class") == device_class
        ):
            mef = entry.get("mef")
            return root / mef if mef else None
    return None


# Op families whose models were force-loaded from the warm manifest this process
# (vs compiled). A consumer can query this to assert the warm actually took
# effect, not merely that it was wired, without parsing the stderr marker.
_MANIFEST_ADOPTED: set[str] = set()


def note_manifest_adoption(family: str) -> None:
    """Record that ``family``'s models were force-loaded from the warm manifest."""
    _MANIFEST_ADOPTED.add(family)


def adopted_from_manifest(family: str) -> bool:
    """Whether ``family`` ("matmul"/"unary") sourced its models from the warm
    manifest this process, rather than a cold or lazy compile."""
    return family in _MANIFEST_ADOPTED


# Serializes lazy first-dispatches so concurrent threads don't race on cache
# mutation or a shared session's ``load_all``.
COMPILE_LOCK = threading.Lock()

# Per-device InferenceSession cache. Keyed by (label, id) since a CPU and an
# accelerator can share id 0.
_SESSION_CACHE: dict[tuple[str, int], engine.InferenceSession] = {}


def session_for(device: Device) -> engine.InferenceSession:
    """Returns a cached single-device :class:`~max.engine.InferenceSession`.

    Caching keeps lazy single-target compiles from recreating a session on
    every cache miss; the session is reused for the process lifetime.
    """
    cache_key = (device.label, device.id)
    session = _SESSION_CACHE.get(cache_key)
    if session is None:
        session = engine.InferenceSession(devices=[device])
        _SESSION_CACHE[cache_key] = session
    return session
