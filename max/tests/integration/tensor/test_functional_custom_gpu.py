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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import os
from pathlib import Path

import pytest
from conftest import assert_all_close
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_custom_external_cubin(kernel_verification_ops_path: Path) -> None:
    # Kernel expects float32
    x = Tensor.zeros([64], dtype=DType.float32)
    y = Tensor.ones_like(x)

    assert x.device.api == "cuda"

    result = F.custom(
        "op_with_external_cubin",
        device=x.device,
        values=[x, y],
        out_types=[x.type],
        custom_extensions=[kernel_verification_ops_path],
    )[0]

    assert result.real
    assert result.type == x.type
    assert_all_close(result, y)
