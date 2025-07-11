# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import deque

import max.nn as nn


def get_all_subclasses() -> set[type[nn.Module]]:
    """Recursively collects all subclasses of nn.Module."""
    all_subclasses = set()

    q = deque([nn.Module])

    while q:
        cls = q.popleft()
        if cls not in all_subclasses:
            all_subclasses.add(cls)
            q.extend(cls.__subclasses__())

    # Exclude nn.Module from the set.
    all_subclasses.remove(nn.Module)

    return all_subclasses
