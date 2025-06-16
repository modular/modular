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

from os import abort

from complex import ComplexFloat32, ComplexFloat64
from gpu.host._nvidia_cuda import CUstream

from .types import Compatibility, LibraryProperty, Property, Status, Type
from .utils import _get_dylib_function

alias cufftHandle = Int16


fn cufftCreate(handle: UnsafePointer[cufftHandle]) -> Status:
    return _get_dylib_function[
        "cufftCreate", fn (UnsafePointer[cufftHandle]) -> Status
    ]()(handle)


fn cufftGetVersion(version: UnsafePointer[Int16]) -> Status:
    return _get_dylib_function[
        "cufftGetVersion", fn (UnsafePointer[Int16]) -> Status
    ]()(version)


fn cufftExecZ2Z(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64],
    odata: UnsafePointer[ComplexFloat64],
    direction: Int16,
) -> Status:
    return _get_dylib_function[
        "cufftExecZ2Z",
        fn (
            cufftHandle,
            UnsafePointer[ComplexFloat64],
            UnsafePointer[ComplexFloat64],
            Int16,
        ) -> Status,
    ]()(plan, idata, odata, direction)


fn cufftExecC2C(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32],
    odata: UnsafePointer[ComplexFloat32],
    direction: Int16,
) -> Status:
    return _get_dylib_function[
        "cufftExecC2C",
        fn (
            cufftHandle,
            UnsafePointer[ComplexFloat32],
            UnsafePointer[ComplexFloat32],
            Int16,
        ) -> Status,
    ]()(plan, idata, odata, direction)


fn cufftExecR2C(
    plan: cufftHandle,
    idata: UnsafePointer[Float32],
    odata: UnsafePointer[ComplexFloat32],
) -> Status:
    return _get_dylib_function[
        "cufftExecR2C",
        fn (
            cufftHandle, UnsafePointer[Float32], UnsafePointer[ComplexFloat32]
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftSetWorkArea(
    plan: cufftHandle, work_area: UnsafePointer[NoneType]
) -> Status:
    return _get_dylib_function[
        "cufftSetWorkArea", fn (cufftHandle, UnsafePointer[NoneType]) -> Status
    ]()(plan, work_area)


fn cufftPlan1d(
    plan: UnsafePointer[cufftHandle], nx: Int16, type: Type, batch: Int16
) -> Status:
    return _get_dylib_function[
        "cufftPlan1d",
        fn (UnsafePointer[cufftHandle], Int16, Type, Int16) -> Status,
    ]()(plan, nx, type, batch)


fn cufftMakePlan2d(
    plan: cufftHandle,
    nx: Int16,
    ny: Int16,
    type: Type,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftMakePlan2d",
        fn (cufftHandle, Int16, Int16, Type, UnsafePointer[Int]) -> Status,
    ]()(plan, nx, ny, type, work_size)


fn cufftSetPlanPropertyInt64(
    plan: cufftHandle, property: Property, input_value_int: Int64
) -> Status:
    return _get_dylib_function[
        "cufftSetPlanPropertyInt64", fn (cufftHandle, Property, Int64) -> Status
    ]()(plan, property, input_value_int)


fn cufftPlan2d(
    plan: UnsafePointer[cufftHandle], nx: Int16, ny: Int16, type: Type
) -> Status:
    return _get_dylib_function[
        "cufftPlan2d",
        fn (UnsafePointer[cufftHandle], Int16, Int16, Type) -> Status,
    ]()(plan, nx, ny, type)


fn cufftMakePlan1d(
    plan: cufftHandle,
    nx: Int16,
    type: Type,
    batch: Int16,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftMakePlan1d",
        fn (cufftHandle, Int16, Type, Int16, UnsafePointer[Int]) -> Status,
    ]()(plan, nx, type, batch, work_size)


fn cufftExecC2R(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32],
    odata: UnsafePointer[Float32],
) -> Status:
    return _get_dylib_function[
        "cufftExecC2R",
        fn (
            cufftHandle, UnsafePointer[ComplexFloat32], UnsafePointer[Float32]
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftMakePlanMany(
    plan: cufftHandle,
    rank: Int16,
    n: UnsafePointer[Int16],
    inembed: UnsafePointer[Int16],
    istride: Int16,
    idist: Int16,
    onembed: UnsafePointer[Int16],
    ostride: Int16,
    odist: Int16,
    type: Type,
    batch: Int16,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany",
        fn (
            cufftHandle,
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            Int16,
            Int16,
            UnsafePointer[Int16],
            Int16,
            Int16,
            Type,
            Int16,
            UnsafePointer[Int],
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


fn cufftSetAutoAllocation(plan: cufftHandle, auto_allocate: Int16) -> Status:
    return _get_dylib_function[
        "cufftSetAutoAllocation", fn (cufftHandle, Int16) -> Status
    ]()(plan, auto_allocate)


fn cufftEstimate1d(
    nx: Int16, type: Type, batch: Int16, work_size: UnsafePointer[Int]
) -> Status:
    return _get_dylib_function[
        "cufftEstimate1d",
        fn (Int16, Type, Int16, UnsafePointer[Int]) -> Status,
    ]()(nx, type, batch, work_size)


fn cufftGetSize(handle: cufftHandle, work_size: UnsafePointer[Int]) -> Status:
    return _get_dylib_function[
        "cufftGetSize", fn (cufftHandle, UnsafePointer[Int]) -> Status
    ]()(handle, work_size)


fn cufftExecZ2D(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64],
    odata: UnsafePointer[Float64],
) -> Status:
    return _get_dylib_function[
        "cufftExecZ2D",
        fn (
            cufftHandle,
            UnsafePointer[ComplexFloat64],
            UnsafePointer[Float64],
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftEstimate2d(
    nx: Int16, ny: Int16, type: Type, work_size: UnsafePointer[Int]
) -> Status:
    return _get_dylib_function[
        "cufftEstimate2d",
        fn (Int16, Int16, Type, UnsafePointer[Int]) -> Status,
    ]()(nx, ny, type, work_size)


fn cufftSetStream(plan: cufftHandle, stream: CUstream) -> Status:
    return _get_dylib_function[
        "cufftSetStream", fn (cufftHandle, CUstream) -> Status
    ]()(plan, stream)


fn cufftMakePlanMany64(
    plan: cufftHandle,
    rank: Int16,
    n: UnsafePointer[Int64],
    inembed: UnsafePointer[Int64],
    istride: Int64,
    idist: Int64,
    onembed: UnsafePointer[Int64],
    ostride: Int64,
    odist: Int64,
    type: Type,
    batch: Int64,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany64",
        fn (
            cufftHandle,
            Int16,
            UnsafePointer[Int64],
            UnsafePointer[Int64],
            Int64,
            Int64,
            UnsafePointer[Int64],
            Int64,
            Int64,
            Type,
            Int64,
            UnsafePointer[Int],
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
    nx: Int16,
    type: Type,
    batch: Int16,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftGetSize1d",
        fn (cufftHandle, Int16, Type, Int16, UnsafePointer[Int]) -> Status,
    ]()(handle, nx, type, batch, work_size)


fn cufftMakePlan3d(
    plan: cufftHandle,
    nx: Int16,
    ny: Int16,
    nz: Int16,
    type: Type,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftMakePlan3d",
        fn (
            cufftHandle, Int16, Int16, Int16, Type, UnsafePointer[Int]
        ) -> Status,
    ]()(plan, nx, ny, nz, type, work_size)


fn cufftGetSizeMany(
    handle: cufftHandle,
    rank: Int16,
    n: UnsafePointer[Int16],
    inembed: UnsafePointer[Int16],
    istride: Int16,
    idist: Int16,
    onembed: UnsafePointer[Int16],
    ostride: Int16,
    odist: Int16,
    type: Type,
    batch: Int16,
    work_area: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany",
        fn (
            cufftHandle,
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            Int16,
            Int16,
            UnsafePointer[Int16],
            Int16,
            Int16,
            Type,
            Int16,
            UnsafePointer[Int],
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
    plan: UnsafePointer[cufftHandle],
    nx: Int16,
    ny: Int16,
    nz: Int16,
    type: Type,
) -> Status:
    return _get_dylib_function[
        "cufftPlan3d",
        fn (UnsafePointer[cufftHandle], Int16, Int16, Int16, Type) -> Status,
    ]()(plan, nx, ny, nz, type)


fn cufftPlanMany(
    plan: UnsafePointer[cufftHandle],
    rank: Int16,
    n: UnsafePointer[Int16],
    inembed: UnsafePointer[Int16],
    istride: Int16,
    idist: Int16,
    onembed: UnsafePointer[Int16],
    ostride: Int16,
    odist: Int16,
    type: Type,
    batch: Int16,
) -> Status:
    return _get_dylib_function[
        "cufftPlanMany",
        fn (
            UnsafePointer[cufftHandle],
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            Int16,
            Int16,
            UnsafePointer[Int16],
            Int16,
            Int16,
            Type,
            Int16,
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


fn cufftResetPlanProperty(plan: cufftHandle, property: Property) -> Status:
    return _get_dylib_function[
        "cufftResetPlanProperty", fn (cufftHandle, Property) -> Status
    ]()(plan, property)


fn cufftGetSize3d(
    handle: cufftHandle,
    nx: Int16,
    ny: Int16,
    nz: Int16,
    type: Type,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftGetSize3d",
        fn (
            cufftHandle, Int16, Int16, Int16, Type, UnsafePointer[Int]
        ) -> Status,
    ]()(handle, nx, ny, nz, type, work_size)


fn cufftGetProperty(
    type: LibraryProperty, value: UnsafePointer[Int16]
) -> Status:
    return _get_dylib_function[
        "cufftGetProperty", fn (LibraryProperty, UnsafePointer[Int16]) -> Status
    ]()(type, value)


fn cufftGetSizeMany64(
    plan: cufftHandle,
    rank: Int16,
    n: UnsafePointer[Int64],
    inembed: UnsafePointer[Int64],
    istride: Int64,
    idist: Int64,
    onembed: UnsafePointer[Int64],
    ostride: Int64,
    odist: Int64,
    type: Type,
    batch: Int64,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany64",
        fn (
            cufftHandle,
            Int16,
            UnsafePointer[Int64],
            UnsafePointer[Int64],
            Int64,
            Int64,
            UnsafePointer[Int64],
            Int64,
            Int64,
            Type,
            Int64,
            UnsafePointer[Int],
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


fn cufftDestroy(plan: cufftHandle) -> Status:
    return _get_dylib_function["cufftDestroy", fn (cufftHandle) -> Status]()(
        plan
    )


fn cufftEstimateMany(
    rank: Int16,
    n: UnsafePointer[Int16],
    inembed: UnsafePointer[Int16],
    istride: Int16,
    idist: Int16,
    onembed: UnsafePointer[Int16],
    ostride: Int16,
    odist: Int16,
    type: Type,
    batch: Int16,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftEstimateMany",
        fn (
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            Int16,
            Int16,
            UnsafePointer[Int16],
            Int16,
            Int16,
            Type,
            Int16,
            UnsafePointer[Int],
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
    idata: UnsafePointer[Float64],
    odata: UnsafePointer[ComplexFloat64],
) -> Status:
    return _get_dylib_function[
        "cufftExecD2Z",
        fn (
            cufftHandle,
            UnsafePointer[Float64],
            UnsafePointer[ComplexFloat64],
        ) -> Status,
    ]()(plan, idata, odata)


fn cufftEstimate3d(
    nx: Int16,
    ny: Int16,
    nz: Int16,
    type: Type,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftEstimate3d",
        fn (Int16, Int16, Int16, Type, UnsafePointer[Int]) -> Status,
    ]()(nx, ny, nz, type, work_size)


fn cufftGetSize2d(
    handle: cufftHandle,
    nx: Int16,
    ny: Int16,
    type: Type,
    work_size: UnsafePointer[Int],
) -> Status:
    return _get_dylib_function[
        "cufftGetSize2d",
        fn (cufftHandle, Int16, Int16, Type, UnsafePointer[Int]) -> Status,
    ]()(handle, nx, ny, type, work_size)


fn cufftGetPlanPropertyInt64(
    plan: cufftHandle,
    property: Property,
    return_ptr_value: UnsafePointer[Int64],
) -> Status:
    return _get_dylib_function[
        "cufftGetPlanPropertyInt64",
        fn (cufftHandle, Property, UnsafePointer[Int64]) -> Status,
    ]()(plan, property, return_ptr_value)
