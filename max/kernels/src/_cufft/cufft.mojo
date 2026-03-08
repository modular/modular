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


import std.ffi

from std.complex import ComplexFloat32, ComplexFloat64
from std.gpu.host._nvidia_cuda import CUstream

from .types import LibraryProperty, Property, Status, Type
from .utils import _get_dylib_function

comptime cufftHandle = ffi.c_uint


fn cufftCreate(handle: UnsafePointer[cufftHandle, _]) raises -> Status:
    return _get_dylib_function["cufftCreate", fn(type_of(handle)) -> Status]()(
        handle
    )


fn cufftGetVersion(version: UnsafePointer[ffi.c_int, _]) raises -> Status:
    return _get_dylib_function[
        "cufftGetVersion", fn(type_of(version)) -> Status
    ]()(version)


fn cufftExecZ2Z(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64, _],
    odata: UnsafePointer[ComplexFloat64, _],
    direction: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftExecZ2Z",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
            ffi.c_int,
        ) -> Status,
    ]()(plan, idata, odata, direction)


fn cufftExecC2C(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32, _],
    odata: UnsafePointer[ComplexFloat32, _],
    direction: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftExecC2C",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
            ffi.c_int,
        ) -> Status,
    ]()(plan, idata, odata, direction)


fn cufftExecR2C(
    plan: cufftHandle,
    idata: UnsafePointer[ffi.c_float, _],
    odata: UnsafePointer[ComplexFloat32, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecR2C",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftSetWorkArea(
    plan: cufftHandle, work_area: OpaquePointer[_]
) raises -> Status:
    return _get_dylib_function[
        "cufftSetWorkArea", fn(cufftHandle, type_of(work_area)) -> Status
    ]()(plan, work_area)


fn cufftPlan1d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan1d",
        fn(type_of(plan), ffi.c_int, Type, ffi.c_int) -> Status,
    ]()(plan, nx, type, batch)


fn cufftMakePlan2d(
    plan: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan2d",
        fn(
            cufftHandle, ffi.c_int, ffi.c_int, Type, type_of(work_size)
        ) -> Status,
    ]()(plan, nx, ny, type, work_size)


fn cufftSetPlanPropertyInt64(
    plan: cufftHandle, property: Property, input_value_int: ffi.c_long_long
) raises -> Status:
    return _get_dylib_function[
        "cufftSetPlanPropertyInt64",
        fn(cufftHandle, Property, ffi.c_long_long) -> Status,
    ]()(plan, property, input_value_int)


fn cufftPlan2d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan2d",
        fn(type_of(plan), ffi.c_int, ffi.c_int, Type) -> Status,
    ]()(plan, nx, ny, type)


fn cufftMakePlan1d(
    plan: cufftHandle,
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan1d",
        fn(
            cufftHandle, ffi.c_int, Type, ffi.c_int, type_of(work_size)
        ) -> Status,
    ]()(plan, nx, type, batch, work_size)


fn cufftExecC2R(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32, _],
    odata: UnsafePointer[ffi.c_float, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecC2R",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftMakePlanMany(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany",
        fn(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_size),
        ) -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


fn cufftSetAutoAllocation(
    plan: cufftHandle, auto_allocate: ffi.c_int
) raises -> Status:
    return _get_dylib_function[
        "cufftSetAutoAllocation", fn(cufftHandle, ffi.c_int) -> Status
    ]()(plan, auto_allocate)


fn cufftEstimate1d(
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate1d",
        fn(ffi.c_int, Type, ffi.c_int, type_of(work_size)) -> Status,
    ]()(nx, type, batch, work_size)


fn cufftGetSize(
    handle: cufftHandle, work_size: UnsafePointer[Int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize", fn(cufftHandle, type_of(work_size)) -> Status
    ]()(handle, work_size)


fn cufftExecZ2D(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64, _],
    odata: UnsafePointer[ffi.c_double, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecZ2D",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftEstimate2d(
    nx: ffi.c_int, ny: ffi.c_int, type: Type, work_size: UnsafePointer[Int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate2d",
        fn(ffi.c_int, ffi.c_int, Type, type_of(work_size)) -> Status,
    ]()(nx, ny, type, work_size)


fn cufftSetStream(plan: cufftHandle, stream: CUstream) raises -> Status:
    return _get_dylib_function[
        "cufftSetStream", fn(cufftHandle, CUstream) -> Status
    ]()(plan, stream)


fn cufftMakePlanMany64(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_long_long, _],
    inembed: UnsafePointer[ffi.c_long_long, _],
    istride: ffi.c_long_long,
    idist: ffi.c_long_long,
    onembed: UnsafePointer[ffi.c_long_long, _],
    ostride: ffi.c_long_long,
    odist: ffi.c_long_long,
    type: Type,
    batch: ffi.c_long_long,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany64",
        fn(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_long_long,
            ffi.c_long_long,
            type_of(onembed),
            ffi.c_long_long,
            ffi.c_long_long,
            Type,
            ffi.c_long_long,
            type_of(work_size),
        ) -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


fn cufftGetSize1d(
    handle: cufftHandle,
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize1d",
        fn(
            cufftHandle, ffi.c_int, Type, ffi.c_int, type_of(work_size)
        ) -> Status,
    ]()(handle, nx, type, batch, work_size)


fn cufftMakePlan3d(
    plan: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan3d",
        fn(
            cufftHandle,
            ffi.c_int,
            ffi.c_int,
            ffi.c_int,
            Type,
            type_of(work_size),
        ) -> Status,
    ]()(plan, nx, ny, nz, type, work_size)


fn cufftGetSizeMany(
    handle: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_area: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany",
        fn(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_area),
        ) -> Status,
    ]()(
        handle,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_area,
    )


fn cufftPlan3d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan3d",
        fn(type_of(plan), ffi.c_int, ffi.c_int, ffi.c_int, Type) -> Status,
    ]()(plan, nx, ny, nz, type)


fn cufftPlanMany(
    plan: UnsafePointer[cufftHandle, _],
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlanMany",
        fn(
            type_of(plan),
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
        ) -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
    )


fn cufftResetPlanProperty(
    plan: cufftHandle, property: Property
) raises -> Status:
    return _get_dylib_function[
        "cufftResetPlanProperty", fn(cufftHandle, Property) -> Status
    ]()(plan, property)


fn cufftGetSize3d(
    handle: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize3d",
        fn(
            cufftHandle,
            ffi.c_int,
            ffi.c_int,
            ffi.c_int,
            Type,
            type_of(work_size),
        ) -> Status,
    ]()(handle, nx, ny, nz, type, work_size)


fn cufftGetProperty(
    type: LibraryProperty, value: UnsafePointer[ffi.c_int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftGetProperty",
        fn(LibraryProperty, type_of(value)) -> Status,
    ]()(type, value)


fn cufftGetSizeMany64(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_long_long, _],
    inembed: UnsafePointer[ffi.c_long_long, _],
    istride: ffi.c_long_long,
    idist: ffi.c_long_long,
    onembed: UnsafePointer[ffi.c_long_long, _],
    ostride: ffi.c_long_long,
    odist: ffi.c_long_long,
    type: Type,
    batch: ffi.c_long_long,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany64",
        fn(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_long_long,
            ffi.c_long_long,
            type_of(onembed),
            ffi.c_long_long,
            ffi.c_long_long,
            Type,
            ffi.c_long_long,
            type_of(work_size),
        ) -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


fn cufftDestroy(plan: cufftHandle) raises -> Status:
    return _get_dylib_function["cufftDestroy", fn(cufftHandle) -> Status]()(
        plan
    )


fn cufftEstimateMany(
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimateMany",
        fn(
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_size),
        ) -> Status,
    ]()(
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


fn cufftExecD2Z(
    plan: cufftHandle,
    idata: UnsafePointer[ffi.c_double, _],
    odata: UnsafePointer[ComplexFloat64, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecD2Z",
        fn(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftEstimate3d(
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate3d",
        fn(ffi.c_int, ffi.c_int, ffi.c_int, Type, type_of(work_size)) -> Status,
    ]()(nx, ny, nz, type, work_size)


fn cufftGetSize2d(
    handle: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize2d",
        fn(
            cufftHandle, ffi.c_int, ffi.c_int, Type, type_of(work_size)
        ) -> Status,
    ]()(handle, nx, ny, type, work_size)


fn cufftGetPlanPropertyInt64(
    plan: cufftHandle,
    property: Property,
    return_ptr_value: UnsafePointer[ffi.c_long_long, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetPlanPropertyInt64",
        fn(cufftHandle, Property, type_of(return_ptr_value)) -> Status,
    ]()(plan, property, return_ptr_value)
