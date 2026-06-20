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
future one) share three process-wide singletons defined here:

- :func:`should_precompile` — whether to compile the full GC matrix at import
  (the default) or compile each target lazily on first dispatch.
- :data:`COMPILE_LOCK` — serializes the lazy check-compile-cache so concurrent
  first-dispatches don't race.
- :func:`session_for` — a per-device :class:`~max.engine.InferenceSession`
  cache, shared across op types so lazy compiles reuse one session per device.

This module must not import from ``handlers.py``.
"""

import os
import threading

from max import engine
from max.driver import Device

# Precompile the full GC matrix at import (the default; a dispatch-time cache
# miss is then a hard error), or with ``MAX_EAGER_OP_PRECOMPILE=0`` compile each
# target lazily on first dispatch, bounding compile cost to what a program uses
# (MXF-508). The same flag gates every op-type cache, so they toggle together.
EAGER_OP_PRECOMPILE_ENV_VAR = "MAX_EAGER_OP_PRECOMPILE"


def should_precompile() -> bool:
    """Returns whether to eagerly precompile the full GC matrix at import.

    Reads ``MAX_EAGER_OP_PRECOMPILE`` (default on) *when called* rather than
    caching it at import. The eager sweep runs from ``__init__`` as a side
    effect of importing this package, and that import can happen — directly or
    transitively — before a launcher, test harness, or startup hook has set the
    env var. Latching the value at import time made ``=0`` silently ineffective
    in those cases (the sweep had already run), which is hard to diagnose since
    ``os.environ`` still reports ``0`` afterward. Reading it at the point the
    sweep is gated keeps the flag honest regardless of import order.
    """
    return os.environ.get(EAGER_OP_PRECOMPILE_ENV_VAR, "1") != "0"


# Serializes the lazy check-compile-cache in the per-op-type model lookups:
# eager dispatch gives no single-threaded guarantee, so the lock keeps
# concurrent first-dispatches from racing on cache mutation and a shared
# session's ``load_all``.
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
