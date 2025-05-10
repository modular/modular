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


from max._utils import CString, call_dylib_func
from memory import UnsafePointer

from ._driver_library import DriverLibrary


@value
@register_passable("trivial")
struct _CStatus:
    var _ptr: UnsafePointer[NoneType]

    fn is_error(self, lib: DriverLibrary) -> Bool:
        alias is_error_func = "M_isError"
        return call_dylib_func[Int](lib.get_handle(), is_error_func, self)

    fn get_error(self, lib: DriverLibrary) -> String:
        alias get_error_func = "M_getError"
        var err = call_dylib_func[CString](
            lib.get_handle(), get_error_func, self
        )
        return String(err)

    fn free(self, lib: DriverLibrary):
        alias free_func = "M_deleteStatus"
        call_dylib_func(lib.get_handle(), free_func, self)


struct Status:
    var impl: _CStatus
    var lib: DriverLibrary

    @implicit
    fn __init__(out self, lib: DriverLibrary):
        self.impl = call_dylib_func[_CStatus](lib.get_handle(), "M_newStatus")
        self.lib = lib

    fn __bool__(self) -> Bool:
        return self.impl.is_error(self.lib)

    fn __str__(self) -> String:
        return self.impl.get_error(self.lib)

    fn __del__(owned self):
        self.impl.free(self.lib)
