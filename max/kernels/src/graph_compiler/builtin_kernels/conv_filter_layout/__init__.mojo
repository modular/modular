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

"""Filter packing shared by the convolution registrations and `mo.layout.transform`."""

from layout import TileTensor
from nn.conv.conv import pack_filter as _pack_conv_filter
from nn.conv.conv import pack_filter_from_fcrs as _pack_conv_filter_from_fcrs
from nn.conv.conv_transpose import pack_filter as _pack_conv_transpose_filter


def layout_transform_conv_transpose_filter_common[
    dtype: DType,
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    """Packs a transposed-convolution filter into the layout expected by the conv_transpose kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv_transpose kernels.
        filter: Input transposed-convolution filter tensor to pack.
    """
    comptime assert filter.rank + 1 == packed_filter.rank
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(filter, packed_filter, 1)


def layout_transform_conv_filter_common[
    dtype: DType, //, num_groups: Int
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    """Packs a convolution filter into the layout expected by the conv kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.
        num_groups: Number of convolution groups for groupwise
            convolution.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv kernels.
        filter: Input convolution filter tensor to pack.
    """
    comptime assert packed_filter.rank == filter.rank + 1

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(filter, packed_filter, num_groups)


def _layout_transform_conv_filter_from_fcrs[
    dtype: DType, //, num_groups: Int
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    comptime assert packed_filter.rank == filter.rank + 1

    # With the compiler-level FCRS→RSCF transpose in PatternFusion,
    # this kernel should no longer be called. But keep it as a fallback
    # using int64 convention (same as the RSCF path).
    _pack_conv_filter_from_fcrs(filter, packed_filter, num_groups)
