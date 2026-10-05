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

"""Host-memory introspection that respects this process's cgroup limit."""

from __future__ import annotations

import os
from typing import NamedTuple

import psutil

# cgroup v1 reports "unlimited" as a near-2**63 sentinel rather than "max".
_NO_CGROUP_LIMIT = 1 << 62


class _CgroupMemory(NamedTuple):
    """Memory controller file paths and ``memory.stat`` keys for one cgroup."""

    limit: str
    usage: str
    stat: str
    cache_key: str
    shmem_key: str


def _v2(directory: str) -> _CgroupMemory:
    return _CgroupMemory(
        limit=f"{directory}/memory.max",
        usage=f"{directory}/memory.current",
        stat=f"{directory}/memory.stat",
        cache_key="file",
        shmem_key="shmem",
    )


def _v1(directory: str) -> _CgroupMemory:
    return _CgroupMemory(
        limit=f"{directory}/memory.limit_in_bytes",
        usage=f"{directory}/memory.usage_in_bytes",
        stat=f"{directory}/memory.stat",
        cache_key="total_cache",
        shmem_key="total_shmem",
    )


def _cgroups() -> list[_CgroupMemory]:
    """Returns the memory cgroups bounding this process.

    Outside a container the cgroup mount isn't namespaced, so the root paths
    miss a unit-level limit such as systemd's ``MemoryMax=``; this process's
    own cgroup comes from ``/proc/self/cgroup``.
    """
    cgroups = [_v2("/sys/fs/cgroup"), _v1("/sys/fs/cgroup/memory")]
    try:
        with open("/proc/self/cgroup") as cgroup_file:
            entries = cgroup_file.readlines()
    except OSError:
        return cgroups

    for entry in entries:
        fields = entry.strip().split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, cgroup_path = fields
        relative = cgroup_path.lstrip("/")
        if not relative:
            continue
        if hierarchy == "0":
            cgroups.append(_v2(f"/sys/fs/cgroup/{relative}"))
        elif "memory" in controllers.split(","):
            cgroups.append(_v1(f"/sys/fs/cgroup/memory/{relative}"))
    return cgroups


def _read_int(path: str) -> int | None:
    try:
        with open(path) as value_file:
            raw = value_file.read().strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_stat(path: str, key: str) -> int | None:
    try:
        with open(path) as stat_file:
            for line in stat_file:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        return None
    return None


def _limit(cgroup: _CgroupMemory) -> int | None:
    limit = _read_int(cgroup.limit)
    if limit is None or not (0 < limit < _NO_CGROUP_LIMIT):
        return None
    return limit


def _headroom(cgroup: _CgroupMemory) -> int | None:
    """Returns what ``cgroup`` may still be charged, or ``None`` if unlimited.

    Reclaimable page cache (``file`` minus ``shmem``) is not counted as used,
    so a server that just read a large checkpoint isn't treated as full.
    """
    limit = _limit(cgroup)
    if limit is None:
        return None
    usage = _read_int(cgroup.usage)
    if usage is None:
        return None

    cache = _read_stat(cgroup.stat, cgroup.cache_key)
    shmem = _read_stat(cgroup.stat, cgroup.shmem_key)
    if cache is not None and shmem is not None:
        usage -= max(0, cache - shmem)

    # Usage can briefly exceed the limit during reclaim.
    return max(0, limit - usage)


def host_memory_limit() -> int | None:
    """Returns the host memory this process may use, or ``None`` if unknown.

    The smallest of every enclosing cgroup limit and physical RAM.
    """
    limits: list[int] = []
    for cgroup in _cgroups():
        limit = _limit(cgroup)
        if limit is not None:
            limits.append(limit)

    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass

    return min(limits) if limits else None


def available_host_memory() -> int | None:
    """Returns the host memory this process may still allocate, or ``None``.

    The smaller of free machine memory and cgroup headroom.
    """
    bounds: list[int] = []
    for cgroup in _cgroups():
        headroom = _headroom(cgroup)
        if headroom is not None:
            bounds.append(headroom)

    try:
        bounds.append(psutil.virtual_memory().available)
    except (OSError, RuntimeError):
        pass

    return min(bounds) if bounds else None
