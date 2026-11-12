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


from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.memory import CacheEviction, async_copy
from max.gpu.sync import async_copy_arrive, mbarrier_init, mbarrier_test_wait
from std.memory import unsafe_stack_allocation
from std.testing import assert_true


def test_mbarrier[
    origin: MutOrigin
](
    addr0: MutPointer[Int8, origin],
    addr1: MutPointer[UInt8, origin],
    addr2: MutPointer[Float32, origin, address_space=.SHARED],
    addr3: MutPointer[Float64, origin, address_space=.SHARED],
):
    async_copy_arrive(addr0)
    async_copy_arrive(addr1)
    async_copy_arrive(addr2)
    async_copy_arrive(addr3)


def _verify_mbarrier(asm: StringSpan) raises -> None:
    # Mbarrier objects must live in shared memory. NVPTX rewrites generic
    # mbarrier operands to the `.shared` form (llvm 64eb69b652ea), so valid
    # addresses lower to the shared arrive instruction.
    assert_true("cp.async.mbarrier.arrive.shared.b64" in asm)


def test_mbarrier_global_rejected[
    origin: MutOrigin
](addr: MutPointer[Float32, origin, address_space=.GLOBAL]):
    async_copy_arrive(addr)


def _verify_mbarrier_global_rejected(asm: StringSpan) raises -> None:
    # A global address can never hold an mbarrier, so the new NVPTX lowers
    # the provably-invalid arrive to `trap` instead of emitting the
    # address-space-generic form (llvm 64eb69b652ea).
    assert_true("trap" in asm)
    assert_true("cp.async.mbarrier.arrive" not in asm)


def test_mbarrier_sm80() raises:
    print("test_mbarrier_sm80")
    var asm = _compile_code[
        test_mbarrier[MutUntrackedOrigin], target=get_gpu_target()
    ]().asm
    _verify_mbarrier(asm)


def test_mbarrier_sm90() raises:
    print("test_mbarrier_sm90")
    var asm = _compile_code[
        test_mbarrier[MutUntrackedOrigin], target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_mbarrier(asm)


def test_mbarrier_global_rejected_sm80() raises:
    print("test_mbarrier_global_rejected_sm80")
    var asm = _compile_code[
        test_mbarrier_global_rejected[MutUntrackedOrigin],
        target=get_gpu_target(),
    ]().asm
    _verify_mbarrier_global_rejected(asm)


def test_mbarrier_global_rejected_sm90() raises:
    print("test_mbarrier_global_rejected_sm90")
    var asm = _compile_code[
        test_mbarrier_global_rejected[MutUntrackedOrigin],
        target=get_gpu_target["sm_90"](),
    ]().asm
    _verify_mbarrier_global_rejected(asm)


def test_mbarrier_init[
    origin: MutOrigin
](shared_mem: MutPointer[Int32, origin, address_space=.SHARED],):
    mbarrier_init(shared_mem, 4)


def _verify_mbarrier_init(asm: StringSpan) raises -> None:
    assert_true("ld.param::entry.b32" in asm)
    assert_true("mov.b32" in asm)
    assert_true("mbarrier.init.shared.b64" in asm)


def test_mbarrier_init_sm80() raises:
    print("test_mbarrier_init_sm80")
    var asm = _compile_code[
        test_mbarrier_init[MutUntrackedOrigin], target=get_gpu_target()
    ]().asm

    _verify_mbarrier_init(asm)


def test_mbarrier_init_sm90() raises:
    print("test_mbarrier_init_sm90")
    var asm = _compile_code[
        test_mbarrier_init[MutUntrackedOrigin], target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_mbarrier_init(asm)


def test_mbarrier_test_wait[
    origin: MutOrigin
](shared_mem: MutPointer[Int32, origin, address_space=.SHARED], state: Int,):
    var done = False
    while not done:
        done = mbarrier_test_wait(shared_mem, state)


def _verify_mbarrier_test_wait(asm: StringSpan) raises -> None:
    assert_true("mbarrier.test_wait.shared.b64" in asm)


def test_mbarrier_test_wait_sm80() raises:
    print("test_mbarrier_test_wait_sm80")
    var asm = _compile_code[
        test_mbarrier_test_wait[MutUntrackedOrigin], target=get_gpu_target()
    ]().asm
    _verify_mbarrier_test_wait(asm)


def test_mbarrier_test_wait_sm90() raises:
    print("test_mbarrier_test_wait_sm90")
    var asm = _compile_code[
        test_mbarrier_test_wait[MutUntrackedOrigin],
        target=get_gpu_target["sm_90"](),
    ]().asm
    assert_true("mbarrier.test_wait.shared.b64" in asm)


def test_async_copy[
    origin: ImmOrigin
](src: ImmPointer[Float32, origin, address_space=.GLOBAL]):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4](src, shared_mem)
    async_copy[16](src, shared_mem)


def _verify_async_copy(asm: StringSpan) raises -> None:
    assert_true("cp.async.ca.shared.global" in asm)
    assert_true("cp.async.cg.shared.global" in asm)


def test_async_copy_sm80() raises:
    print("test_async_copy_sm80")
    var asm = _compile_code[
        test_async_copy[ImmUntrackedOrigin], target=get_gpu_target()
    ]().asm
    _verify_async_copy(asm)


def test_async_copy_sm90() raises:
    print("test_async_copy_sm90")
    var asm = _compile_code[
        test_async_copy[ImmUntrackedOrigin], target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_async_copy(asm)


def test_async_copy_l2_prefetch[
    origin: ImmOrigin
](src: ImmPointer[Float32, origin, address_space=.GLOBAL]):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, bypass_L1_16B=False, l2_prefetch=128](src, shared_mem)
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64](src, shared_mem)


def _verify_async_copy_l2_prefetch(asm: StringSpan) raises -> None:
    assert_true("cp.async.ca.shared.global.L2::128B" in asm)
    assert_true("cp.async.ca.shared.global.L2::64B" in asm)


def test_async_copy_l2_prefetch_sm80() raises:
    print("test_async_l2_prefetch_sm80")
    var asm = _compile_code[
        test_async_copy_l2_prefetch[ImmUntrackedOrigin], target=get_gpu_target()
    ]().asm
    _verify_async_copy_l2_prefetch(asm)


def test_async_copy_l2_prefetch_sm90() raises:
    print("test_async_l2_prefetch_sm90")
    var asm = _compile_code[
        test_async_copy_l2_prefetch[ImmUntrackedOrigin],
        target=get_gpu_target["sm_90"](),
    ]().asm
    _verify_async_copy_l2_prefetch(asm)


def test_async_copy_with_zero_fill_kernel[
    origin: ImmOrigin
](src: ImmPointer[Float32, origin, address_space=.GLOBAL]):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, bypass_L1_16B=False, l2_prefetch=128, fill=Float32(0)](
        src, shared_mem
    )
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64, fill=Float32(0)](
        src, shared_mem
    )


def _find_required(
    text: StringSpan, needle: StaticString, start: Int = 0
) raises -> Int:
    """Returns where `needle` first occurs in `text`, at or after `start`."""
    match text.find(needle, start):
        case .Some(index):
            return index
        case .None:
            raise Error(t"expected to find {needle}")


def _rfind_required(text: StringSpan, needle: StaticString) raises -> Int:
    """Returns where `needle` last occurs in `text`."""
    match text.rfind(needle):
        case .Some(index):
            return index
        case .None:
            raise Error(t"expected to find {needle}")


def _verify_test_async_copy_with_zero_fill(asm: StringSpan) raises -> None:
    # Should contain something like:
    #     cp.async.ca.shared.global.L2::128B [%r1], [%rd2], 4, %r2;
    # Regex would be nice here, but alas...
    var cp128_begin = _find_required(asm, "cp.async.ca.shared.global.L2::128B")
    var cp128_end = _find_required(asm, ";", cp128_begin)
    var cp128_str = asm[byte=cp128_begin:cp128_end]
    # Find various parts of the string we expect to be there
    var cp128_first_reg32_pos = _find_required(cp128_str, "[%r")
    var cp128_reg64_pos = _find_required(cp128_str, "[%rd")
    var cp128_bytes_pos = _find_required(cp128_str, ", 4,")
    var cp128_last_reg32_pos = _rfind_required(cp128_str, ", %r")
    # Assert they're in the right order
    assert_true(cp128_first_reg32_pos < cp128_reg64_pos)
    assert_true(cp128_reg64_pos < cp128_bytes_pos)
    assert_true(cp128_bytes_pos < cp128_last_reg32_pos)

    # Should contain something like:
    #     cp.async.ca.shared.global.L2::64B [%r1], [%rd1], 16, %r4;
    # Regex would be nice here, but alas...
    var cp64_begin = _find_required(asm, "cp.async.ca.shared.global.L2::64B")
    var cp64_end = _find_required(asm, ";", cp64_begin)
    var cp64_str = asm[byte=cp64_begin:cp64_end]
    # Find various parts of the string we expect to be there
    var cp64_first_reg32_pos = _find_required(cp64_str, "[%r")
    var cp64_reg64_pos = _find_required(cp64_str, "[%rd")
    var cp64_bytes_pos = _find_required(cp64_str, ", 16,")
    var cp64_last_reg32_pos = _rfind_required(cp64_str, ", %r")
    # Assert they're in the right order
    assert_true(cp64_first_reg32_pos < cp64_reg64_pos)
    assert_true(cp64_reg64_pos < cp64_bytes_pos)
    assert_true(cp64_bytes_pos < cp64_last_reg32_pos)


def test_async_copy_with_zero_fill() raises:
    print("test_async_copy_zero_fill")
    var asm = _compile_code[
        test_async_copy_with_zero_fill_kernel[ImmUntrackedOrigin],
        target=get_gpu_target(),
    ]().asm
    _verify_test_async_copy_with_zero_fill(asm)


def test_async_copy_with_eviction[
    origin: ImmOrigin
](src: ImmPointer[Float32, origin, address_space=.GLOBAL]):
    print("test_async_copy_with_eviction")
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, eviction_policy=CacheEviction.EVICT_FIRST](src, shared_mem)
    async_copy[16, eviction_policy=CacheEviction.EVICT_FIRST](src, shared_mem)
    async_copy[16, eviction_policy=CacheEviction.EVICT_LAST](src, shared_mem)


def async_copy_with_non_zero_fill_kernel[
    origin: ImmOrigin
](src: ImmPointer[Int32, origin, address_space=.GLOBAL]):
    var shared_mem = unsafe_stack_allocation[
        4, DType.int32, address_space=.SHARED
    ]()
    async_copy[16, bypass_L1_16B=False, l2_prefetch=128, fill=Int32(32)](
        src, shared_mem, predicate=True
    )
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64, fill=Int32(32)](
        src, shared_mem, predicate=False
    )


def _verify_async_copy_with_non_zero_fill(asm: StringSpan) raises -> None:
    assert_true("mov.b32 	%r3, 32;" in asm)
    assert_true("@p cp.async.ca.shared.global.L2::128B" in asm and "16" in asm)
    assert_true("@p cp.async.ca.shared.global.L2::64B" in asm and "16" in asm)
    assert_true("@!p st.shared.v4.b32" in asm)


def test_async_copy_with_non_zero_fill() raises:
    print("test_async_copy_with_non_zero_fill")
    var asm = _compile_code[
        async_copy_with_non_zero_fill_kernel[ImmUntrackedOrigin],
        target=get_gpu_target(),
    ]().asm
    _verify_async_copy_with_non_zero_fill(asm)


def _verify_async_copy_with_eviction(asm: StringSpan) raises -> None:
    assert_true("createpolicy.fractional.L2::evict_first.b64" in asm)
    assert_true("createpolicy.fractional.L2::evict_last.b64" in asm)
    assert_true("cp.async.ca.shared.global" in asm)


def test_async_copy_with_eviction_sm80() raises:
    print("test_async_copy_with_eviction_sm80")
    var asm = _compile_code[
        test_async_copy_with_eviction[ImmUntrackedOrigin],
        target=get_gpu_target["sm_80"](),
    ]().asm
    _verify_async_copy_with_eviction(asm)


def test_async_copy_with_eviction_sm90() raises:
    print("test_async_copy_with_eviction_sm90")
    var asm = _compile_code[
        test_async_copy_with_eviction[ImmUntrackedOrigin],
        target=get_gpu_target["sm_90"](),
    ]().asm
    _verify_async_copy_with_eviction(asm)


def main() raises:
    test_mbarrier_sm80()
    test_mbarrier_sm90()
    test_mbarrier_global_rejected_sm80()
    test_mbarrier_global_rejected_sm90()
    test_mbarrier_init_sm80()
    test_mbarrier_init_sm90()
    test_mbarrier_test_wait_sm80()
    test_mbarrier_test_wait_sm90()
    test_async_copy_sm80()
    test_async_copy_sm90()
    test_async_copy_l2_prefetch_sm80()
    test_async_copy_l2_prefetch_sm90()
    test_async_copy_with_zero_fill()
    test_async_copy_with_non_zero_fill()
    test_async_copy_with_eviction_sm80()
    test_async_copy_with_eviction_sm90()
