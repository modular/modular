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
"""Which ``wo_a`` configs take the block-diagonal MXFP8 matmul."""

import pytest
from max.driver import CPU
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV4_modulev3.layers.kernels import (
    device_architectures,
    wo_a_mxfp8_supported,
)

# DeepSeek-V4-Flash: 64 heads of 512 in 8 groups of rank 1024.
O_GROUPS = 8
O_LORA_RANK = 1024
GROUP_WIDTH = 64 * 512 // O_GROUPS


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_flash_takes_the_fused_path(tp: int) -> None:
    assert wo_a_mxfp8_supported(
        ["sm_100a"] * tp, O_GROUPS // tp, O_LORA_RANK, GROUP_WIDTH
    )


@pytest.mark.parametrize(
    "architectures, groups, o_lora_rank, group_width",
    [
        # TP8: one group a device.
        (["sm_100a"] * 8, 1, O_LORA_RANK, GROUP_WIDTH),
        # A group of 64 rows is half a scale tile.
        (["sm_100a"], 2, 64, GROUP_WIDTH),
        (["sm_100a"], 8, O_LORA_RANK, 4160),
        (["sm_90a"], O_GROUPS, O_LORA_RANK, GROUP_WIDTH),
        (["sm_100a", "sm_90a"], O_GROUPS // 2, O_LORA_RANK, GROUP_WIDTH),
        ([""], O_GROUPS, O_LORA_RANK, GROUP_WIDTH),
        ([], O_GROUPS, O_LORA_RANK, GROUP_WIDTH),
    ],
)
def test_falls_back(
    architectures: list[str], groups: int, o_lora_rank: int, group_width: int
) -> None:
    assert not wo_a_mxfp8_supported(
        architectures, groups, o_lora_rank, group_width
    )


def test_a_cpu_has_no_architecture() -> None:
    assert device_architectures([CPU(), DeviceRef.CPU()]) == ["", ""]
