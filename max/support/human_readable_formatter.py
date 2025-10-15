# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Private helper function for formatting various quantities into human readable strings."""


def to_human_readable_bytes(bytes: int) -> str:
    """Convert bytes to human readable memory size."""
    KiB = 1024
    MiB = KiB * 1024
    GiB = MiB * 1024
    TiB = GiB * 1024
    bytes = int(bytes)
    if bytes > TiB:
        return f"{bytes / TiB:.2f} TiB"
    if bytes > GiB:
        return f"{bytes / GiB:.2f} GiB"
    if bytes > MiB:
        return f"{bytes / MiB:.2f} MiB"
    return f"{bytes / KiB:.2f} KiB"


def to_human_readable_latency(s: float) -> str:
    """Converts seconds to human readable latency."""
    if s >= 1:
        return f"{s:.2f}s"
    ms = s * 1e3
    if ms >= 1:
        return f"{ms:.2f}ms"
    us = ms * 1e3
    if us >= 1:
        return f"{us:.2f}us"
    ns = us * 1e3
    return f"{ns:.2f}ns"
