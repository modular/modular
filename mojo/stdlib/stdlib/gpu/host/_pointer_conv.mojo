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
from sys.intrinsics import _type_is_eq
from builtin.rebind import downcast
from memory.legacy_unsafe_pointer import _IsUnsafePointer


fn _is_pointer_convertible[T: AnyType, U: AnyType]() -> Bool:
    """Returns true if T and U are both UnsafePointer types and can be converted
    to each other.
    """

    @parameter
    if conforms_to(T, _IsUnsafePointer) and conforms_to(U, _IsUnsafePointer):
        alias downcast_t = downcast[_IsUnsafePointer, T]
        alias downcast_u = downcast[_IsUnsafePointer, U]
        return _type_is_eq[
            downcast_t._UnsafePointerType, downcast_u._UnsafePointerType
        ]()
    return False
