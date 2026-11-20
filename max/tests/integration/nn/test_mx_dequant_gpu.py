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
"""GPU tests for the MXFP4 / MXFP8 dequant ops.

Runs the dequant checks on the optimized GPU kernel, including shapes large
enough to exercise multi-block grids. Argument validation is in
``test_mx_dequant.py``.

MXFP4 with bfloat16 output and checkpoint-layout scales is not tested here: on
a GPU that combination still dispatches to the original H100/AMD-oriented
``mxfp4_dequant`` kernel, which has its own tests. Float32 output, and every
MXFP8 and interleaved-scale case, run on the shared kernel.
"""

from __future__ import annotations

import pytest
import shared_mx_dequant_impl as impl
from max.driver import Accelerator, accelerator_count
from max.dtype import DType

pytestmark = pytest.mark.skipif(accelerator_count() == 0, reason="needs a GPU")

FORMATS: list[impl.Format] = ["mxfp4", "mxfp8"]
OUT_DTYPES = [DType.float32, DType.bfloat16]
SHAPES = [(1, 32), (4, 512), (37, 96), (130, 160), (256, 2880)]


def _skip_legacy_path(
    fmt: impl.Format, out_dtype: DType, interleaved: bool
) -> None:
    if fmt == "mxfp4" and out_dtype == DType.bfloat16 and not interleaved:
        pytest.skip("covered by the original mxfp4_dequant kernel's tests")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_dequant_rank2_scales(
    fmt: impl.Format, shape: tuple[int, int], out_dtype: DType
) -> None:
    _skip_legacy_path(fmt, out_dtype, interleaved=False)
    impl.check_dequant(Accelerator(), fmt, shape, out_dtype)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape", [*SHAPES, (200, 320)])
def test_dequant_interleaved_scales(
    fmt: impl.Format, shape: tuple[int, int], out_dtype: DType
) -> None:
    impl.check_dequant(Accelerator(), fmt, shape, out_dtype, interleaved=True)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_dequant_large(
    fmt: impl.Format, out_dtype: DType, interleaved: bool
) -> None:
    _skip_legacy_path(fmt, out_dtype, interleaved)
    impl.check_dequant(
        Accelerator(),
        fmt,
        (4096, 4096),
        out_dtype,
        interleaved=interleaved,
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", [(3, 8, 64), (2, 130, 96)])
def test_dequant_rank3_weights(
    fmt: impl.Format, shape: tuple[int, int, int]
) -> None:
    impl.check_dequant(Accelerator(), fmt, shape, DType.float32)
