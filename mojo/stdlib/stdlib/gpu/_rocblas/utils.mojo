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

from collections.string import StaticString
from pathlib import Path
from sys.ffi import _find_dylib
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias ROCM_ROCBLAS_LIBRARY_PATHS = List[Path](
    "librocblas.so.4",
    "/opt/rocm/lib/librocblas.so.4",
)

alias ROCM_ROCBLAS_LIBRARY = _Global[
    "ROCM_ROCBLAS_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib["ROCm rocBLAS library"](ROCM_ROCBLAS_LIBRARY_PATHS)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        ROCM_ROCBLAS_LIBRARY,
        func_name,
        result_type,
    ]()


@always_inline
fn check_error(stat: Status) raises:
    if stat != Status.SUCCESS:
        raise String("ROCBLAS ERROR:", stat)
