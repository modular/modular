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
"""Test for MixedLayout GPU memory codegen with runtime indices."""

from gpu.host.compile import _compile_code, get_gpu_target
from layout._mixed_layout import MixedLayout
from layout._mixed_tuple import Idx, MixedIntTuple
from layout.int_tuple import IntTuple
from memory.unsafe_pointer import UnsafePointer
from testing import assert_true
import sys


fn kernel_mixed_dimensions(x: Int, ptr: UnsafePointer[Int32]):
    """
    Args:
        x: Runtime value for dynamic indexing.
        ptr: Output pointer to store results.
    """
    # Create layout with mixed compile-time and runtime dimensions
    var layout = MixedLayout(
        shape=[Idx[8](), Idx(x)], stride=[Idx(x), Idx[1]()]
    )

    var coords = MixedIntTuple(Idx[0](), Idx(x - 1))

    # Store results
    ptr[0] = Int32(layout(coords))


fn test_mixed_layout_mixed_dimensions_codegen() raises:
    """Test codegen for MixedLayout with mixed compile-time and runtime dimensions.
    """

    # Test AMD GPU codegen
    var amd_asm = _compile_code[
        kernel_mixed_dimensions, target = get_gpu_target["amdgpu:gfx942"]()
    ]().asm

    # Should not load from buffer for compile-time known values
    assert_true("ds_read" not in amd_asm and "ds_write" not in amd_asm)

    # Test NVIDIA GPU codegen
    var nvidia_asm = _compile_code[
        kernel_mixed_dimensions, target = get_gpu_target["sm_80"]()
    ]().asm

    # Should not use local memory for compile-time known values
    assert_true("ld.local" not in nvidia_asm and "st.local" not in nvidia_asm)


fn main() raises:
    test_mixed_layout_mixed_dimensions_codegen()
