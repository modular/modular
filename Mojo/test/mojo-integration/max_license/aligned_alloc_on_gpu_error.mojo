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

# RUN: kgen -emit=llvm %s --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_89 --target-features="" --verify-diagnostics


@always_inline
def _malloc[
    type: AnyType, /, *, alignment: Int = 1
](size: Int, /) -> Pointer[type, MutUntrackedOrigin, address_space=.GENERIC]:
    return {
        _mlir_value = __mlir_op.`pop.aligned_alloc`[
            _type=Pointer[
                type, MutUntrackedOrigin, address_space=.GENERIC
            ]._mlir_type
            # expected-error @+2 {{heap allocation is not supported on GPU}}
            # expected-note-re @+1 {{see current operation: {{.*}}}}
        ](alignment.__mlir_index__(), size.__mlir_index__())
    }


@always_inline
def _free(ptr: Pointer[_, MutUntrackedOrigin, address_space=.GENERIC]):
    # expected-error @+2 {{heap deallocation is not supported on GPU}}
    # expected-note-re @+1 {{see current operation: {{.*}}}}
    __mlir_op.`pop.aligned_free`(ptr._get_kgen_pointer())


@export
def heap_allocation() abi("Mojo"):
    var ptr = _malloc[Int32](42)
    _free(ptr)
