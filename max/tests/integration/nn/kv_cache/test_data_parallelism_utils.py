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

from __future__ import annotations

from max.nn.kv_cache.data_parallelism_utils import split_into_groups


def test_split_into_groups() -> None:
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
