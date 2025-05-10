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

from ._tensor_impl import CTensor

alias CMojoVal = UnsafePointer[UInt8]


@value
@register_passable("trivial")
struct CValue:
    """Represents an AsyncValue pointer from Engine."""

    var ptr: UnsafePointer[NoneType]

    alias _GetTensorFnName = "M_getTensorFromValue"
    alias _GetBoolFnName = "M_getBoolFromValue"
    alias _GetListFnName = "M_getListFromValue"
    alias _TakeMojoValueFnName = "M_takeMojoValueFromValue"
    alias _FreeValueFnName = "M_freeValue"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn get_c_tensor(self, lib: DLHandle) -> CTensor:
        """Get CTensor within value."""
        return call_dylib_func[CTensor](lib, Self._GetTensorFnName, self)

    fn get_bool(self, lib: DLHandle) -> Bool:
        """Get bool within value."""
        return call_dylib_func[Bool](lib, Self._GetBoolFnName, self)

    fn get_list(self, lib: DLHandle) -> CList:
        """Get list within value."""
        return call_dylib_func[CList](lib, Self._GetListFnName, self)

    fn take_mojo_value(self, lib: DLHandle) -> CMojoVal:
        """Take ownership of mojo_val within value."""
        return call_dylib_func[CMojoVal](lib, Self._TakeMojoValueFnName, self)

    fn free(self, lib: DLHandle):
        """Free value."""
        call_dylib_func(lib, Self._FreeValueFnName, self)


@value
@register_passable("trivial")
struct CList:
    """Represents an AsyncList pointer from Engine."""

    var ptr: UnsafePointer[NoneType]

    alias _GetSizeFnName = "M_getListSize"
    alias _GetValueFnName = "M_getListValue"
    alias _AppendFnName = "M_appendToList"
    alias _FreeFnName = "M_freeList"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn get_size(self, lib: DLHandle) -> Int:
        """Get number of elements in list."""
        return call_dylib_func[Int](lib, Self._GetSizeFnName, self)

    fn get_value(self, lib: DLHandle, index: Int) -> CValue:
        """Get value by index in list."""
        return call_dylib_func[CValue](lib, Self._GetValueFnName, self, index)

    fn append(self, lib: DLHandle, value: CValue):
        """Get value by index in list."""
        return call_dylib_func(lib, Self._AppendFnName, self, value)

    fn free(self, lib: DLHandle):
        """Free list."""
        call_dylib_func(lib, Self._FreeFnName, self)
