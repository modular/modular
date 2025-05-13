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

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.tcgen05 import (
    TensorMemory,
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from memory import UnsafePointer, stack_allocation
from testing import assert_true


fn alloc_test_fn[cta_group: Int32]():
    var tmem = TensorMemory(32)
    tcgen05_alloc[cta_group](tmem)


fn test_tcgen05_alloc() raises:
    var asm1 = _compile_code_asm[
        alloc_test_fn[1],
        target = _get_gpu_target["sm_100a"](),
    ]()
    var asm2 = _compile_code_asm[
        alloc_test_fn[2],
        target = _get_gpu_target["sm_100a"](),
    ]()
    assert_true(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" in asm1
    )
    assert_true(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32" in asm2
    )


fn alloc_dealloc_test_fn():
    var tmem = TensorMemory(32)
    tcgen05_alloc[1](tmem)
    tcgen05_release_allocation_lock()
    tcgen05_dealloc[1](tmem)


fn test_tcgen05_dealloc() raises:
    var asm = _compile_code_asm[
        alloc_dealloc_test_fn,
        target = _get_gpu_target["sm_100a"](),
    ]()
    assert_true(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" in asm
    )
    assert_true("tcgen05.dealloc.cta_group::1.sync.aligned.b32" in asm)


fn ld_test_fn():
    var tmem = TensorMemory(32)
    tcgen05_alloc[1](tmem)
    _ = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=64,
        type = DType.float32,
        pack=False,
        width=64,
    ](tmem)
    tcgen05_load_wait()
    tcgen05_dealloc[1](tmem)


fn test_tcgen05_ld() raises:
    var asm = _compile_code_asm[
        ld_test_fn,
        target = _get_gpu_target["sm_100a"](),
    ]()
    assert_true("tcgen05.ld.sync.aligned.32x32b.x64.b32" in asm)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm)


fn main() raises:
    test_tcgen05_alloc()
    test_tcgen05_dealloc()
    test_tcgen05_ld()
