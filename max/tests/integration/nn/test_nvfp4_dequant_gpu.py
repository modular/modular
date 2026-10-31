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
"""GPU tests for ``nvfp4_dequant``.

Runs the same checks as ``test_nvfp4_dequant.py`` on the optimized GPU kernel,
plus shapes large enough to exercise multi-block grids.
"""

from __future__ import annotations

import pytest
import shared_nvfp4_dequant_impl as impl
from max.driver import Accelerator, accelerator_count
from max.dtype import DType

pytestmark = pytest.mark.skipif(accelerator_count() == 0, reason="needs a GPU")

OUT_DTYPES = [DType.float32, DType.bfloat16]
SHAPES = [(1, 16), (4, 512), (37, 80), (130, 192), (256, 2880)]


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_dequant_rank2_scales(shape: tuple[int, int], out_dtype: DType) -> None:
    impl.check_dequant(Accelerator(), shape, out_dtype, tensor_sf=0.0123)


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape", [*SHAPES, (200, 320)])
def test_dequant_interleaved_scales(
    shape: tuple[int, int], out_dtype: DType
) -> None:
    impl.check_dequant(
        Accelerator(), shape, out_dtype, tensor_sf=0.0123, interleaved=True
    )


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_dequant_large(out_dtype: DType, interleaved: bool) -> None:
    impl.check_dequant(
        Accelerator(),
        (4096, 4096),
        out_dtype,
        tensor_sf=0.0123,
        interleaved=interleaved,
    )


@pytest.mark.parametrize("shape", [(3, 8, 64), (2, 130, 48)])
def test_dequant_rank3_weights(shape: tuple[int, int, int]) -> None:
    impl.check_dequant(Accelerator(), shape, DType.float32, tensor_sf=0.5)
