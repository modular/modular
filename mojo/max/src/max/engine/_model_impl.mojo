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

from sys.ffi import DLHandle

from max._utils import call_dylib_func
from memory import UnsafePointer

from ._compilation import CCompiledModel
from ._status import Status
from .session import InferenceSession


@value
@register_passable("trivial")
struct CModel:
    """Mojo representation of Engine's AsyncModel pointer.
    Useful for C inter-op.
    """

    var ptr: UnsafePointer[NoneType]

    alias FreeModelFnName = "M_freeModel"
    alias WaitForModelFnName = "M_waitForModel"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn await_model(self, lib: DLHandle) raises:
        var status = Status(lib)
        call_dylib_func(
            lib, Self.WaitForModelFnName, self.ptr, status.borrow_ptr()
        )
        if status:
            raise status.__str__()

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeModelFnName, self)
