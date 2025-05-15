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
"""This module defines GPU device attributes that can be queried from CUDA-compatible devices.

The module provides the `DeviceAttribute` struct which encapsulates the various device
properties and capabilities that can be queried through the CUDA driver API. Each attribute
is represented as a constant with a corresponding integer value that maps to the CUDA
driver's attribute enumeration.

These attributes allow applications to query specific hardware capabilities and limitations
of GPU devices, such as maximum thread counts, memory sizes, compute capabilities, and
supported features.
"""


@fieldwise_init("implicit")
@register_passable("trivial")
struct DeviceAttribute:
    """
    Represents CUDA device attributes that can be queried from a GPU device.

    This struct encapsulates the various device properties and capabilities that can be
    queried through the CUDA driver API. Each attribute is represented as a constant
    with a corresponding integer value that maps to the CUDA driver's attribute enum.
    """

    var _value: Int32
    """The integer value representing the specific device attribute."""

    alias MAX_THREADS_PER_BLOCK = Self(1)
    """Maximum number of threads per block
    """

    alias MAX_BLOCK_DIM_X = Self(2)
    """Maximum block dimension X
    """

    alias MAX_BLOCK_DIM_Y = Self(3)
    """Maximum block dimension Y
    """

    alias MAX_BLOCK_DIM_Z = Self(4)
    """Maximum block dimension Z
    """

    alias MAX_GRID_DIM_X = Self(5)
    """Maximum grid dimension X
    """

    alias MAX_GRID_DIM_Y = Self(6)
    """Maximum grid dimension Y
    """

    alias MAX_GRID_DIM_Z = Self(7)
    """Maximum grid dimension Z
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK = Self(8)
    """Maximum shared memory available per block in bytes
    """

    alias WARP_SIZE = Self(10)
    """Warp size in threads
    """

    alias MAX_REGISTERS_PER_BLOCK = Self(12)
    """Maximum number of 32-bit registers available per block
    """

    alias CLOCK_RATE = Self(13)
    """Typical clock frequency in kilohertz
    """

    alias MULTIPROCESSOR_COUNT = Self(16)
    """Number of multiprocessors on device
    """

    alias MAX_THREADS_PER_MULTIPROCESSOR = Self(39)
    """Maximum resident threads per multiprocessor
    """

    alias COMPUTE_CAPABILITY_MAJOR = Self(75)
    """Major compute capability version number
    """

    alias COMPUTE_CAPABILITY_MINOR = Self(76)
    """Minor compute capability version number
    """
    alias MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = Self(81)
    """Maximum shared memory available per multiprocessor in bytes
    """
    alias MAX_REGISTERS_PER_MULTIPROCESSOR = Self(82)
    """Maximum number of 32-bit registers available per multiprocessor
    """
    alias MAX_BLOCKS_PER_MULTIPROCESSOR = Self(106)
    """Maximum resident blocks per multiprocessor
    """
    alias MAX_ACCESS_POLICY_WINDOW_SIZE = Self(109)
    """ CUDA-only: Maximum value of CUaccessPolicyWindow::num_bytes.
    """
