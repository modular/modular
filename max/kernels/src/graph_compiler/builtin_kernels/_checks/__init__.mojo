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

"""Argument checks the collective and tuple-taking registrations share."""

from std.math import ceildiv
from std.sys.info import size_of

from comm import Signal
from extensibility import Tensor


def _check_tensor_tuple[size: Int, dtype: DType, rank: Int, *Ts: Tensor]():
    """Checks that a collective's tensor tuple has one element per device, each
    of the given dtype and rank."""
    comptime assert len(Ts) == size, "expected one tensor per device"
    comptime for i in range(len(Ts)):
        comptime assert Ts[i].dtype == dtype, "unexpected tensor dtype"
        comptime assert Ts[i].rank == rank, "unexpected tensor rank"


@inline(.always)
def _check_signal_buffer_size(
    signal_buffer_size: Int, input_size_bytes: Int
) raises:
    # The signal buffer has to be large enough to hold the entire input buffer.
    var min_signal_buffer_size = size_of[Signal]() + input_size_bytes
    if signal_buffer_size < min_signal_buffer_size:
        raise Error(
            "Expected signal buffer to be at least ",
            min_signal_buffer_size,
            " bytes, but got ",
            signal_buffer_size,
            (
                ". This error can appear when running large requests through"
                " MAX serve without chunked prefill. If so, try enabling"
                " chunked prefill with --enable-chunked-prefill. Otherwise,"
                " consider increasing the signal buffer size."
            ),
        )


def _partitioned_scratch_requirement[
    num_devices: Int, dtype: DType
](input_elems: Int) -> Int:
    """Calculate a trivial scratch memory requirement for comm kernels.

    This applies for comm kernels which simply partition the input tensor between devices.
    """
    comptime pessemistic_simd_width = 32
    var num_vecs = ceildiv(input_elems, pessemistic_simd_width)
    var vecs_per_device = ceildiv(num_vecs, num_devices)

    return vecs_per_device * pessemistic_simd_width * size_of[dtype]()
