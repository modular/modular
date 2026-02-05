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
"""DLPack types.

See
- https://dmlc.github.io/dlpack/latest/c_api.html.
- https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

All implementations are based on that reference material.
"""


@fieldwise_init
struct DLDataType(ImplicitlyCopyable, Movable):
    # https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDataTypeCode
    comptime INT: UInt8 = 0
    comptime UINT: UInt8 = 1
    comptime FLOAT: UInt8 = 2
    comptime OPAQUE_HANDLE: UInt8 = 3
    comptime BFLOAT16: UInt8 = 4
    comptime COMPLEX: UInt8 = 5
    comptime BOOL: UInt8 = 6
    comptime FLOAT8_E3M4: UInt8 = 7
    comptime FLOAT8_E4M3: UInt8 = 8
    comptime FLOAT8_E4M3B11FNUZ: UInt8 = 9
    comptime FLOAT8_E4M3FN: UInt8 = 10
    comptime FLOAT8_E4M3FNUZ: UInt8 = 11
    comptime FLOAT8_E5M2: UInt8 = 12
    comptime FLOAT8_E5M2FNUZ: UInt8 = 13
    comptime FLOAT8_E8M0FNU: UInt8 = 14
    comptime FLOAT6_E2M3FN: UInt8 = 15
    comptime FLOAT6_E3M2FN: UInt8 = 16
    comptime FLOAT4_E2M1FN: UInt8 = 17

    comptime CODE_MAP: Dict[DType, UInt8] = {
        DType.bool: Self.BOOL,
        DType.uint: Self.UINT,
        DType.uint8: Self.UINT,
        DType.uint16: Self.UINT,
        DType.uint32: Self.UINT,
        DType.uint64: Self.UINT,
        DType.uint128: Self.UINT,
        DType.uint256: Self.UINT,
        DType.int: Self.INT,
        DType.int8: Self.INT,
        DType.int16: Self.INT,
        DType.int32: Self.INT,
        DType.int64: Self.INT,
        DType.int128: Self.INT,
        DType.int256: Self.INT,
        DType.float4_e2m1fn: Self.FLOAT4_E2M1FN,
        DType.float8_e3m4: Self.FLOAT8_E3M4,
        DType.float8_e4m3fn: Self.FLOAT8_E4M3FN,
        DType.float8_e4m3fnuz: Self.FLOAT8_E4M3FNUZ,
        DType.float8_e8m0fnu: Self.FLOAT8_E8M0FNU,
        DType.float8_e5m2: Self.FLOAT8_E5M2,
        DType.float8_e5m2fnuz: Self.FLOAT8_E5M2FNUZ,
        DType.bfloat16: Self.BFLOAT16,
        DType.float16: Self.FLOAT,
        DType.float32: Self.FLOAT,
        DType.float64: Self.FLOAT,
    }

    var code: UInt8
    var bits: UInt8
    var lanes: UInt16

    @staticmethod
    fn from_dtype[dtype: DType](lanes: UInt16 = 1) -> Self:
        constrained[dtype in Self.CODE_MAP]()
        comptime code: UInt8 = Self.CODE_MAP.get(dtype, 0)
        return Self(code, sys.bit_width_of[dtype](), lanes)


struct DLDevice(ImplicitlyCopyable, Movable):
    # https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDeviceType
    comptime CPU: Int32 = 1
    comptime CUDA: Int32 = 2
    comptime CUDA_HOST: Int32 = 3
    comptime OPENCL: Int32 = 4
    comptime VULKAN: Int32 = 7
    comptime METAL: Int32 = 8
    comptime VPI: Int32 = 9
    comptime ROCM: Int32 = 10
    comptime ROCM_HOST: Int32 = 11
    comptime EXT_DEV: Int32 = 12
    comptime CUDA_MANAGED: Int32 = 13
    comptime ONEAPI: Int32 = 14
    comptime WEBGPU: Int32 = 15
    comptime HEXAGON: Int32 = 16
    comptime MAIA: Int32 = 17
    comptime TRAINIUM: Int32 = 18

    var device_type: Int32
    var device_id: Int32

    fn __init__(out self, device_type: Int32 = Self.CUDA, device_id: Int32 = 0):
        self.device_type = device_type
        self.device_id = device_id


struct DLTensor[rank: Int, dtype: DType](Copyable):
    # https://dmlc.github.io/dlpack/latest/c_api.html#c.DLTensor
    var data: Pointer[Scalar[Self.dtype], MutAnyOrigin]
    var device: DLDevice
    var _rank: Int32
    var data_type: DLDataType
    var _shape_ptr: UnsafePointer[Int64, MutAnyOrigin]
    var _strides_ptr: UnsafePointer[Int64, MutAnyOrigin]
    var byte_offset: UInt64

    # Implementation detail: Store shape and strides locally
    # after the struct.
    # This makes naive copying and moving DLTensor unsafe.
    var shape: IndexList[Self.rank]
    var strides: IndexList[Self.rank]

    fn __init__(
        out self,
        tensor: ManagedTensorSlice[dtype = Self.dtype, rank = Self.rank],
    ):
        self.data = Pointer(to=tensor.unsafe_ptr()[])
        self.device = DLDevice()  # XXX
        self._rank = Int32(Self.rank)
        self.data_type = DLDataType.from_dtype[Self.dtype]()
        self.shape = tensor.shape()
        self.strides = tensor.strides()
        self.byte_offset = 0

        self._shape_ptr = UnsafePointer(to=self.shape).bitcast[Int64]()
        self._strides_ptr = UnsafePointer(to=self.strides).bitcast[Int64]()

    fn __init__(out self, other: Self):
        self.data = other.data
        self.device = other.device
        self._rank = Int32(Self.rank)
        self.data_type = DLDataType.from_dtype[Self.dtype]()
        # Copy constructor needs to update shape and strides ptrs
        self.shape = other.shape
        self.strides = other.strides
        self.byte_offset = 0

        self._shape_ptr = UnsafePointer(to=self.shape).bitcast[Int64]()
        self._strides_ptr = UnsafePointer(to=self.strides).bitcast[Int64]()
