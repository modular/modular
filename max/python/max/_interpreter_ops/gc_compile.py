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


def _cache_dir() -> Path | None:
    """MEF cache dir the warm stamp lives in, or None if unset.

    Keyed off ``MODULAR_DERIVED_PATH`` — the redirect knob warmer and consumer
    both set to agree on location (matching ``tools/interpreter_warm_cache``);
    unset → no stamp → lazy. A config-file ``cache_dir`` would still win in the
    engine (GEX-3884).
    """
    derived = os.environ.get("MODULAR_DERIVED_PATH")
    return Path(derived) / "cache" / ".max_cache" if derived else None


def _context_signature() -> str:
    """Signature a warm must match before a lazy process adopts it.

    Pins host arch + accelerator count/SKU so a different machine can't falsely
    match and trigger a cold recompile. ``accelerator_architecture_name`` raises
    on a CPU device, so it's only queried when an accelerator is present.
    """
    n = accelerator_count()
    accel = accelerator_architecture_name() if n else ""
    return f"accelerators={n};cpu={platform.machine()};accel={accel}"


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
    """Returns whether a warm stamp matching this context is present."""
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    try:
        stamp = json.loads((cache_dir / _WARM_STAMP_NAME).read_text())
    except (OSError, ValueError):
        return False
    return stamp.get("context") == _context_signature()


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
