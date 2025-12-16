# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Sequence

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from rms_helpers import SHAPES, run_test_norm

CPU_DTYPES = (DType.float32, DType.float64)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_norm(
    session: InferenceSession, shape: Sequence[int], dtype: DType
) -> None:
    run_test_norm(
        session,
        TensorType(dtype, shape, device=DeviceRef.CPU()),
        rtol=1e-4,
        atol=1e-8,
    )
