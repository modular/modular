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

from max._utils import CString, call_dylib_func
from memory import UnsafePointer


@value
@register_passable("trivial")
struct CTensorSpec:
    """Mojo representation of Engine's TensorSpec pointer.
    This doesn't free the memory on destruction.
    """

    alias ptr_type = UnsafePointer[NoneType]
    var ptr: Self.ptr_type

    alias FreeTensorSpecFnName = "M_freeTensorSpec"
    alias GetDimAtFnName = "M_getDimAt"
    alias GetRankFnName = "M_getRank"
    alias GetNameFnName = "M_getName"
    alias GetDTypeFnName = "M_getDtype"
    alias IsDynamicallyRankedFnName = "M_isDynamicRanked"
    alias GetDynamicRankValueFnName = "M_getDynamicRankValue"
    alias GetDynamicDimensionValueFnName = "M_getDynamicDimensionValue"

    fn get_dim_at(self, idx: Int, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDimAtFnName, self, idx)

    fn get_rank(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetRankFnName, self)

    fn get_name(self, lib: DLHandle) -> String:
        var name = call_dylib_func[CString](lib, Self.GetNameFnName, self)
        return name.__str__()

    fn get_dtype(self, lib: DLHandle) -> DType:
        return call_dylib_func[DType](lib, Self.GetDTypeFnName, self)

    fn is_dynamically_ranked(self, lib: DLHandle) -> Bool:
        var is_dynamic = call_dylib_func[Int](
            lib, Self.IsDynamicallyRankedFnName, self
        )
        return is_dynamic == 1

    @staticmethod
    fn get_dynamic_rank_value(lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDynamicRankValueFnName)

    @staticmethod
    fn get_dynamic_dimension_value(lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDynamicDimensionValueFnName)

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorSpecFnName, self)
