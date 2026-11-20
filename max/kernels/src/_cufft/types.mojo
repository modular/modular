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

from std.os import abort


@fieldwise_init
struct Status(Equatable, Identifiable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUFFT_INVALID_PLAN = Self(1)
    comptime CUFFT_SUCCESS = Self(0)
    comptime CUFFT_ALLOC_FAILED = Self(2)
    comptime CUFFT_INVALID_TYPE = Self(3)
    comptime CUFFT_INVALID_VALUE = Self(4)
    comptime CUFFT_INTERNAL_ERROR = Self(5)
    comptime CUFFT_EXEC_FAILED = Self(6)
    comptime CUFFT_SETUP_FAILED = Self(7)
    comptime CUFFT_INVALID_SIZE = Self(8)
    comptime CUFFT_UNALIGNED_DATA = Self(9)
    comptime CUFFT_INCOMPLETE_PARAMETER_LIST = Self(10)
    comptime CUFFT_INVALID_DEVICE = Self(11)
    comptime CUFFT_PARSE_ERROR = Self(12)
    comptime CUFFT_NO_WORKSPACE = Self(13)
    comptime CUFFT_NOT_IMPLEMENTED = Self(14)
    comptime CUFFT_LICENSE_ERROR = Self(15)
    comptime CUFFT_NOT_SUPPORTED = Self(16)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case Self.CUFFT_SUCCESS:
                writer.write_string("CUFFT_SUCCESS")
            case Self.CUFFT_INVALID_PLAN:
                writer.write_string("CUFFT_INVALID_PLAN")
            case Self.CUFFT_ALLOC_FAILED:
                writer.write_string("CUFFT_ALLOC_FAILED")
            case Self.CUFFT_INVALID_TYPE:
                writer.write_string("CUFFT_INVALID_TYPE")
            case Self.CUFFT_INVALID_VALUE:
                writer.write_string("CUFFT_INVALID_VALUE")
            case Self.CUFFT_INTERNAL_ERROR:
                writer.write_string("CUFFT_INTERNAL_ERROR")
            case Self.CUFFT_EXEC_FAILED:
                writer.write_string("CUFFT_EXEC_FAILED")
            case Self.CUFFT_SETUP_FAILED:
                writer.write_string("CUFFT_SETUP_FAILED")
            case Self.CUFFT_INVALID_SIZE:
                writer.write_string("CUFFT_INVALID_SIZE")
            case Self.CUFFT_UNALIGNED_DATA:
                writer.write_string("CUFFT_UNALIGNED_DATA")
            case Self.CUFFT_INCOMPLETE_PARAMETER_LIST:
                writer.write_string("CUFFT_INCOMPLETE_PARAMETER_LIST")
            case Self.CUFFT_INVALID_DEVICE:
                writer.write_string("CUFFT_INVALID_DEVICE")
            case Self.CUFFT_PARSE_ERROR:
                writer.write_string("CUFFT_PARSE_ERROR")
            case Self.CUFFT_NO_WORKSPACE:
                writer.write_string("CUFFT_NO_WORKSPACE")
            case Self.CUFFT_NOT_IMPLEMENTED:
                writer.write_string("CUFFT_NOT_IMPLEMENTED")
            case Self.CUFFT_LICENSE_ERROR:
                writer.write_string("CUFFT_LICENSE_ERROR")
            case Self.CUFFT_NOT_SUPPORTED:
                writer.write_string("CUFFT_NOT_SUPPORTED")
            case _:
                abort("invalid cufftResult_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftResult_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


#  CUFFT defines and supports the following data types


@fieldwise_init
struct Type(Equatable, Identifiable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUFFT_R2C = Self(0x2A)
    comptime CUFFT_C2R = Self(0x2C)
    comptime CUFFT_C2C = Self(0x29)
    comptime CUFFT_D2Z = Self(0x6A)
    comptime CUFFT_Z2D = Self(0x6C)
    comptime CUFFT_Z2Z = Self(0x69)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case Self.CUFFT_R2C:
                writer.write_string("CUFFT_R2C")
            case Self.CUFFT_C2R:
                writer.write_string("CUFFT_C2R")
            case Self.CUFFT_C2C:
                writer.write_string("CUFFT_C2C")
            case Self.CUFFT_D2Z:
                writer.write_string("CUFFT_D2Z")
            case Self.CUFFT_Z2D:
                writer.write_string("CUFFT_Z2D")
            case Self.CUFFT_Z2Z:
                writer.write_string("CUFFT_Z2Z")
            case _:
                abort("invalid cufftType_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)
