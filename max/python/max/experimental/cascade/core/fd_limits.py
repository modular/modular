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
"""Open-file limit setup for processes that hold many in-flight calls."""

from __future__ import annotations

import ctypes
import logging
import resource
import sys

logger = logging.getLogger(__name__)


def raise_nofile_soft_limit() -> None:
    """Raises the soft ``RLIMIT_NOFILE`` toward the hard limit.

    Darwin rejects a soft limit above ``kern.maxfilesperproc`` (older
    releases silently clamp it instead), and launchd commonly leaves the hard
    limit unlimited, so the requested value is capped there.

    Failure to raise is logged and otherwise ignored: a genuinely exhausted
    budget surfaces as ``OSError`` from the runtime call that hits it.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard
    if sys.platform == "darwin":
        cap = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(cap))
        if (
            ctypes.CDLL(None).sysctlbyname(
                b"kern.maxfilesperproc",
                ctypes.byref(cap),
                ctypes.byref(size),
                None,
                0,
            )
            == 0
        ):
            target = min(target, cap.value)
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as error:
        logger.debug("Could not raise RLIMIT_NOFILE to %d: %s", target, error)
