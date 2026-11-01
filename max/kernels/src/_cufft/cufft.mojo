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


import std.ffi as ffi

from std.complex import ComplexFloat32
from max.gpu.host import CUstream

from .types import Status, Type
from .utils import _get_dylib_function

comptime cufftHandle = ffi.c_uint


def cufftCreate(handle: Pointer[cufftHandle, _]) raises -> Status:
    return _get_dylib_function[
        "cufftCreate", def(type_of(handle)) thin -> Status
    ]()(handle)


def cufftSetWorkArea(
    plan: cufftHandle, work_area: OpaquePointer[_]
) raises -> Status:
    return _get_dylib_function[
        "cufftSetWorkArea", def(cufftHandle, type_of(work_area)) thin -> Status
    ]()(plan, work_area)


def cufftMakePlan1d(
    plan: cufftHandle,
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: Pointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan1d",
        def(
            cufftHandle, ffi.c_int, Type, ffi.c_int, type_of(work_size)
        ) thin -> Status,
    ]()(plan, nx, type, batch, work_size)


def cufftExecC2R(
    plan: cufftHandle,
    idata: Pointer[ComplexFloat32, _],
    odata: Pointer[ffi.c_float, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecC2R",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) thin -> Status,
    ]()(plan, idata, odata)


def cufftSetAutoAllocation(
    plan: cufftHandle, auto_allocate: ffi.c_int
) raises -> Status:
    return _get_dylib_function[
        "cufftSetAutoAllocation", def(cufftHandle, ffi.c_int) thin -> Status
    ]()(plan, auto_allocate)


def cufftEstimate1d(
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: Pointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate1d",
        def(ffi.c_int, Type, ffi.c_int, type_of(work_size)) thin -> Status,
    ]()(nx, type, batch, work_size)


def cufftGetSize(
    handle: cufftHandle, work_size: Pointer[Int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize", def(cufftHandle, type_of(work_size)) thin -> Status
    ]()(handle, work_size)


def cufftSetStream(plan: cufftHandle, stream: CUstream) raises -> Status:
    return _get_dylib_function[
        "cufftSetStream", def(cufftHandle, CUstream) thin -> Status
    ]()(plan, stream)
