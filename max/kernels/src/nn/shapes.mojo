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

from math import ceildiv


@always_inline("nodebug")
fn get_sliding_window_out_dim[
    ceil_mode: Bool = False,
](in_dim: Int, ft_dim: Int, dilation: Int, stride: Int, pad: Int) -> Int:
    """
    Return output dimension for a sliding window operation along some dimension.

    Parameters:
        ceil_mode: Define rounding mode for shape calculation.

    Args:
        in_dim: The size of the input dimension.
        ft_dim: The size of the corresponding filter dimension.
        dilation: The dilation for the sliding window operation.
        stride: The stride for the sliding window operation.
        pad: The total padding for the sliding window operation.

    Returns:
        The size of the output dimension.

    """

    @parameter
    if ceil_mode:
        return 1 + ceildiv(in_dim + pad - (1 + dilation * (ft_dim - 1)), stride)
    else:
        return 1 + (in_dim + pad - (1 + dilation * (ft_dim - 1))) // stride
