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

from collections import List
from collections.string import StaticString
from os import abort
from pathlib import Path
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUFFT_LIBRARY_PATHS = List[String](
    "/usr/local/cuda/lib64/libcufft.so",
    "/usr/local/cuda/lib64/libcufft.so.11",
)

alias CUDA_CUFFT_LIBRARY = _Global[
    "CUDA_CUFFT_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _get_library_path() -> String:
    for path in CUDA_CUFFT_LIBRARY_PATHS:
        if Path(path[]).exists():
            return path[]
    return abort[String](
        String(
            (
                "the ROCM rocBLAS library was not found in any of the"
                " following paths: "
            ),
            String(", ").join(CUDA_CUFFT_LIBRARY_PATHS),
        )
    )


fn _init_dylib() -> _OwnedDLHandle:
    return _OwnedDLHandle(_get_library_path())


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUFFT_LIBRARY,
        func_name,
        result_type,
    ]()


@always_inline
fn check_error(stat: Status) raises:
    if stat != Status.CUFFT_SUCCESS:
        raise String("CUBLAS ERROR:", stat)
