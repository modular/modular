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
"""Runs repeat_per_image's device copies on a GPU buffer."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.pipelines.architectures.flux2.flux2_executor import repeat_per_image


def test_repeats_a_device_row_per_image() -> None:
    row = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
    repeated = repeat_per_image(Buffer.from_dlpack(row).to(Accelerator()), 3)
    assert not repeated.is_host
    np.testing.assert_array_equal(
        np.from_dlpack(repeated.to(CPU())), np.repeat(row, 3, axis=0)
    )


def test_repeats_a_device_timestep_slice() -> None:
    # The denoise loops repeat one step's slice of the timestep schedule.
    timesteps = Buffer.from_dlpack(
        np.array([1.0, 0.75, 0.5], dtype=np.float32)
    ).to(Accelerator())
    repeated = repeat_per_image(timesteps[1:2], 2)
    np.testing.assert_array_equal(
        np.from_dlpack(repeated.to(CPU())), [0.75, 0.75]
    )


def test_repeats_an_empty_device_buffer() -> None:
    empty = Buffer.from_dlpack(np.zeros((1, 0, 4), dtype=np.int64)).to(
        Accelerator()
    )
    repeated = repeat_per_image(empty, 2)
    assert repeated.shape == (2, 0, 4)
    assert not repeated.is_host
