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

from unittest.mock import patch

from max.driver import calculate_virtual_device_count_from_cli


@patch("max.driver.driver.accelerator_count", return_value=4)
def test_cli_gpu_all_uses_visible_accelerator_count(_mock: object) -> None:
    assert calculate_virtual_device_count_from_cli("gpu:all") == 4


@patch("max.driver.driver.accelerator_count", return_value=0)
def test_cli_gpu_all_with_zero_accelerators_yields_minimum_virtual_count(
    _mock: object,
) -> None:
    """`gpu:all` with N=0 accelerators leaves max_gpu_id at -1.

    The helper always returns at least 1 virtual slot (`max(1, max_gpu_id + 1)`),
    same as an empty / no-GPU CLI device list. That is *not* claiming one
    physical GPU exists; later parsing (`device_specs`) rejects `gpu:all` when
    no GPUs are visible.
    """
    assert calculate_virtual_device_count_from_cli("gpu:all") == 1


def test_cli_gpu_colon_list_parses_max_id() -> None:
    assert calculate_virtual_device_count_from_cli("gpu:0,3") == 4
