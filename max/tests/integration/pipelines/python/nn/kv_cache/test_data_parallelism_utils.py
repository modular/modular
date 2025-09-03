# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.nn.kv_cache.data_parallelism_utils import split_into_groups


def test_split_into_groups():
    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    data_parallel_degree = 1
    expected_devices = [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert split_into_groups(devices, data_parallel_degree) == expected_devices

    data_parallel_degree = 2
    expected_devices = [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert split_into_groups(devices, data_parallel_degree) == expected_devices

    data_parallel_degree = 3
    expected_devices = [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert split_into_groups(devices, data_parallel_degree) == expected_devices

    data_parallel_degree = 10
    expected_devices = [[0], [1], [2], [3], [4], [5], [6], [7], [], []]
    assert split_into_groups(devices, data_parallel_degree) == expected_devices
