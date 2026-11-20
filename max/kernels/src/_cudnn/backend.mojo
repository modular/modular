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
from std.pathlib import Path
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle

from .infer import cudnnStatus_t

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_LIBRARY_PATHS: List[Path] = [
    "libcudnn.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the CUDNN libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUDNN_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUDNN_LIBRARY = _Global[
    "CUDA_CUDNN_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUDNN_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct cudnnBackendDescriptorType_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_BACKEND_POINTWISE_DESCRIPTOR = Self(0)
    comptime CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = Self(1)
    comptime CUDNN_BACKEND_ENGINE_DESCRIPTOR = Self(2)
    comptime CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = Self(3)
    comptime CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = Self(4)
    comptime CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = Self(5)
    comptime CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = Self(6)
    comptime CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR = Self(7)
    comptime CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR = Self(8)
    comptime CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR = Self(9)
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = Self(10)
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = Self(
        11
    )
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = Self(
        12
    )
    comptime CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = Self(13)
    comptime CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = Self(14)
    comptime CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = Self(15)
    comptime CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = Self(16)
    comptime CUDNN_BACKEND_TENSOR_DESCRIPTOR = Self(17)
    comptime CUDNN_BACKEND_MATMUL_DESCRIPTOR = Self(18)
    comptime CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = Self(19)
    comptime CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = Self(
        20
    )
    comptime CUDNN_BACKEND_REDUCTION_DESCRIPTOR = Self(21)
    comptime CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = Self(22)
    comptime CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = Self(23)
    comptime CUDNN_BACKEND_RESAMPLE_DESCRIPTOR = Self(24)
    comptime CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = Self(25)
    comptime CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = Self(26)
    comptime CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = Self(27)
    comptime CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = Self(28)
    comptime CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = Self(29)
    comptime CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = Self(30)
    comptime CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR = Self(31)
    comptime CUDNN_BACKEND_RNG_DESCRIPTOR = Self(32)
    comptime CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = Self(33)
    comptime CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR = Self(34)
    comptime CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = Self(35)
    comptime CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR = Self(36)
    comptime CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR = Self(
        37
    )
    comptime CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR = Self(38)
    comptime CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR = Self(39)
    comptime CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR = Self(40)
    comptime CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = Self(41)
    comptime CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR = Self(42)
    comptime CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR = Self(43)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_POINTWISE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINECFG_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        ):
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        ):
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_TENSOR_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_TENSOR_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_MATMUL_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_MATMUL_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
        ):
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_REDUCTION_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_RESAMPLE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_RNG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_RNG_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
        ):
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR"
            )
        abort("invalid cudnnBackendDescriptorType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendDescriptorType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendSetAttribute(
    descriptor: OpaquePointer,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: Int64,
    array_of_elements: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendSetAttribute",
        def(
            type_of(descriptor),
            type_of(attribute_name),
            type_of(attribute_type),
            type_of(element_count),
            type_of(array_of_elements),
        ) thin -> cudnnStatus_t,
    ]()(
        descriptor,
        attribute_name,
        attribute_type,
        element_count,
        array_of_elements,
    )


def cudnnBackendCreateDescriptor(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: Pointer[OpaquePointer[AnyOrigin[mut=True]], _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendCreateDescriptor",
        def(
            type_of(descriptor_type), type_of(descriptor)
        ) thin -> cudnnStatus_t,
    ]()(descriptor_type, descriptor)


@fieldwise_init
struct cudnnBackendAttributeType_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_TYPE_HANDLE = Self(0)
    comptime CUDNN_TYPE_DATA_TYPE = Self(1)
    comptime CUDNN_TYPE_BOOLEAN = Self(2)
    comptime CUDNN_TYPE_INT64 = Self(3)
    comptime CUDNN_TYPE_FLOAT = Self(4)
    comptime CUDNN_TYPE_DOUBLE = Self(5)
    comptime CUDNN_TYPE_VOID_PTR = Self(6)
    comptime CUDNN_TYPE_CONVOLUTION_MODE = Self(7)
    comptime CUDNN_TYPE_HEUR_MODE = Self(8)
    comptime CUDNN_TYPE_KNOB_TYPE = Self(9)
    comptime CUDNN_TYPE_NAN_PROPOGATION = Self(10)
    comptime CUDNN_TYPE_NUMERICAL_NOTE = Self(11)
    comptime CUDNN_TYPE_LAYOUT_TYPE = Self(12)
    comptime CUDNN_TYPE_ATTRIB_NAME = Self(13)
    comptime CUDNN_TYPE_POINTWISE_MODE = Self(14)
    comptime CUDNN_TYPE_BACKEND_DESCRIPTOR = Self(15)
    comptime CUDNN_TYPE_GENSTATS_MODE = Self(16)
    comptime CUDNN_TYPE_BN_FINALIZE_STATS_MODE = Self(17)
    comptime CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = Self(18)
    comptime CUDNN_TYPE_BEHAVIOR_NOTE = Self(19)
    comptime CUDNN_TYPE_TENSOR_REORDERING_MODE = Self(20)
    comptime CUDNN_TYPE_RESAMPLE_MODE = Self(21)
    comptime CUDNN_TYPE_PADDING_MODE = Self(22)
    comptime CUDNN_TYPE_INT32 = Self(23)
    comptime CUDNN_TYPE_CHAR = Self(24)
    comptime CUDNN_TYPE_SIGNAL_MODE = Self(25)
    comptime CUDNN_TYPE_FRACTION = Self(26)
    comptime CUDNN_TYPE_NORM_MODE = Self(27)
    comptime CUDNN_TYPE_NORM_FWD_PHASE = Self(28)
    comptime CUDNN_TYPE_RNG_DISTRIBUTION = Self(29)
    comptime CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE = Self(30)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_TYPE_HANDLE:
            return writer.write_string("CUDNN_TYPE_HANDLE")
        if self is Self.CUDNN_TYPE_DATA_TYPE:
            return writer.write_string("CUDNN_TYPE_DATA_TYPE")
        if self is Self.CUDNN_TYPE_BOOLEAN:
            return writer.write_string("CUDNN_TYPE_BOOLEAN")
        if self is Self.CUDNN_TYPE_INT64:
            return writer.write_string("CUDNN_TYPE_INT64")
        if self is Self.CUDNN_TYPE_FLOAT:
            return writer.write_string("CUDNN_TYPE_FLOAT")
        if self is Self.CUDNN_TYPE_DOUBLE:
            return writer.write_string("CUDNN_TYPE_DOUBLE")
        if self is Self.CUDNN_TYPE_VOID_PTR:
            return writer.write_string("CUDNN_TYPE_VOID_PTR")
        if self is Self.CUDNN_TYPE_CONVOLUTION_MODE:
            return writer.write_string("CUDNN_TYPE_CONVOLUTION_MODE")
        if self is Self.CUDNN_TYPE_HEUR_MODE:
            return writer.write_string("CUDNN_TYPE_HEUR_MODE")
        if self is Self.CUDNN_TYPE_KNOB_TYPE:
            return writer.write_string("CUDNN_TYPE_KNOB_TYPE")
        if self is Self.CUDNN_TYPE_NAN_PROPOGATION:
            return writer.write_string("CUDNN_TYPE_NAN_PROPOGATION")
        if self is Self.CUDNN_TYPE_NUMERICAL_NOTE:
            return writer.write_string("CUDNN_TYPE_NUMERICAL_NOTE")
        if self is Self.CUDNN_TYPE_LAYOUT_TYPE:
            return writer.write_string("CUDNN_TYPE_LAYOUT_TYPE")
        if self is Self.CUDNN_TYPE_ATTRIB_NAME:
            return writer.write_string("CUDNN_TYPE_ATTRIB_NAME")
        if self is Self.CUDNN_TYPE_POINTWISE_MODE:
            return writer.write_string("CUDNN_TYPE_POINTWISE_MODE")
        if self is Self.CUDNN_TYPE_BACKEND_DESCRIPTOR:
            return writer.write_string("CUDNN_TYPE_BACKEND_DESCRIPTOR")
        if self is Self.CUDNN_TYPE_GENSTATS_MODE:
            return writer.write_string("CUDNN_TYPE_GENSTATS_MODE")
        if self is Self.CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
            return writer.write_string("CUDNN_TYPE_BN_FINALIZE_STATS_MODE")
        if self is Self.CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
            return writer.write_string("CUDNN_TYPE_REDUCTION_OPERATOR_TYPE")
        if self is Self.CUDNN_TYPE_BEHAVIOR_NOTE:
            return writer.write_string("CUDNN_TYPE_BEHAVIOR_NOTE")
        if self is Self.CUDNN_TYPE_TENSOR_REORDERING_MODE:
            return writer.write_string("CUDNN_TYPE_TENSOR_REORDERING_MODE")
        if self is Self.CUDNN_TYPE_RESAMPLE_MODE:
            return writer.write_string("CUDNN_TYPE_RESAMPLE_MODE")
        if self is Self.CUDNN_TYPE_PADDING_MODE:
            return writer.write_string("CUDNN_TYPE_PADDING_MODE")
        if self is Self.CUDNN_TYPE_INT32:
            return writer.write_string("CUDNN_TYPE_INT32")
        if self is Self.CUDNN_TYPE_CHAR:
            return writer.write_string("CUDNN_TYPE_CHAR")
        if self is Self.CUDNN_TYPE_SIGNAL_MODE:
            return writer.write_string("CUDNN_TYPE_SIGNAL_MODE")
        if self is Self.CUDNN_TYPE_FRACTION:
            return writer.write_string("CUDNN_TYPE_FRACTION")
        if self is Self.CUDNN_TYPE_NORM_MODE:
            return writer.write_string("CUDNN_TYPE_NORM_MODE")
        if self is Self.CUDNN_TYPE_NORM_FWD_PHASE:
            return writer.write_string("CUDNN_TYPE_NORM_FWD_PHASE")
        if self is Self.CUDNN_TYPE_RNG_DISTRIBUTION:
            return writer.write_string("CUDNN_TYPE_RNG_DISTRIBUTION")
        if self is Self.CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE:
            return writer.write_string("CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE")
        abort("invalid cudnnBackendAttributeType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendAttributeType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendFinalize(descriptor: OpaquePointer) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendFinalize", def(type_of(descriptor)) thin -> cudnnStatus_t
    ]()(descriptor)


@fieldwise_init
struct cudnnBackendAttributeName_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_ATTR_POINTWISE_MODE = Self(0)
    comptime CUDNN_ATTR_POINTWISE_MATH_PREC = Self(1)
    comptime CUDNN_ATTR_POINTWISE_NAN_PROPAGATION = Self(2)
    comptime CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = Self(3)
    comptime CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = Self(4)
    comptime CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = Self(5)
    comptime CUDNN_ATTR_POINTWISE_ELU_ALPHA = Self(6)
    comptime CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA = Self(7)
    comptime CUDNN_ATTR_POINTWISE_SWISH_BETA = Self(8)
    comptime CUDNN_ATTR_POINTWISE_AXIS = Self(9)
    comptime CUDNN_ATTR_CONVOLUTION_COMP_TYPE = Self(100)
    comptime CUDNN_ATTR_CONVOLUTION_CONV_MODE = Self(101)
    comptime CUDNN_ATTR_CONVOLUTION_DILATIONS = Self(102)
    comptime CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = Self(103)
    comptime CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = Self(104)
    comptime CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = Self(105)
    comptime CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = Self(106)
    comptime CUDNN_ATTR_ENGINEHEUR_MODE = Self(200)
    comptime CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = Self(201)
    comptime CUDNN_ATTR_ENGINEHEUR_RESULTS = Self(202)
    comptime CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = Self(203)
    comptime CUDNN_ATTR_ENGINEHEUR_DEVICEPROP = Self(204)
    comptime CUDNN_ATTR_ENGINECFG_ENGINE = Self(300)
    comptime CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = Self(301)
    comptime CUDNN_ATTR_ENGINECFG_KNOB_CHOICES = Self(302)
    comptime CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE = Self(303)
    comptime CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED = Self(304)
    comptime CUDNN_ATTR_EXECUTION_PLAN_HANDLE = Self(400)
    comptime CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = Self(401)
    comptime CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = Self(402)
    comptime CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = Self(403)
    comptime CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = Self(404)
    comptime CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = Self(405)
    comptime CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE = Self(406)
    comptime CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP = Self(407)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID = Self(500)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_SIZE = Self(501)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = Self(502)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = Self(503)
    comptime CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE = Self(600)
    comptime CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = Self(601)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = Self(700)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = Self(701)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = Self(702)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = Self(703)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = Self(704)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = Self(705)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = Self(706)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = Self(707)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = Self(708)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = Self(709)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = Self(710)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = Self(711)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = Self(712)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = Self(713)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = Self(714)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = Self(715)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = Self(716)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = Self(717)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = Self(750)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_XDESC = Self(751)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_BDESC = Self(752)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_YDESC = Self(753)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 = Self(754)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 = Self(755)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_DXDESC = Self(756)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_DYDESC = Self(757)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_TDESC = Self(758)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_MODE = Self(770)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = Self(771)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_XDESC = Self(772)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC = Self(773)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = Self(774)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE = Self(780)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC = Self(781)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = Self(782)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = Self(783)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = Self(784)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = Self(785)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = Self(786)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = Self(787)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = Self(
        788
    )
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = Self(
        789
    )
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = Self(790)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = Self(791)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = Self(792)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = Self(793)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = Self(794)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = Self(795)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = Self(
        796
    )
    comptime CUDNN_ATTR_OPERATIONGRAPH_HANDLE = Self(800)
    comptime CUDNN_ATTR_OPERATIONGRAPH_OPS = Self(801)
    comptime CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = Self(802)
    comptime CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = Self(803)
    comptime CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY = Self(804)
    comptime CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = Self(900)
    comptime CUDNN_ATTR_TENSOR_DATA_TYPE = Self(901)
    comptime CUDNN_ATTR_TENSOR_DIMENSIONS = Self(902)
    comptime CUDNN_ATTR_TENSOR_STRIDES = Self(903)
    comptime CUDNN_ATTR_TENSOR_VECTOR_COUNT = Self(904)
    comptime CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = Self(905)
    comptime CUDNN_ATTR_TENSOR_UNIQUE_ID = Self(906)
    comptime CUDNN_ATTR_TENSOR_IS_VIRTUAL = Self(907)
    comptime CUDNN_ATTR_TENSOR_IS_BY_VALUE = Self(908)
    comptime CUDNN_ATTR_TENSOR_REORDERING_MODE = Self(909)
    comptime CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = Self(913)
    comptime CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = Self(1000)
    comptime CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = Self(1001)
    comptime CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = Self(1002)
    comptime CUDNN_ATTR_VARIANT_PACK_WORKSPACE = Self(1003)
    comptime CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS = Self(1010)
    comptime CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES = Self(1011)
    comptime CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES = Self(1012)
    comptime CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = Self(1100)
    comptime CUDNN_ATTR_LAYOUT_INFO_TYPES = Self(1101)
    comptime CUDNN_ATTR_KNOB_INFO_TYPE = Self(1200)
    comptime CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = Self(1201)
    comptime CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = Self(1202)
    comptime CUDNN_ATTR_KNOB_INFO_STRIDE = Self(1203)
    comptime CUDNN_ATTR_ENGINE_OPERATION_GRAPH = Self(1300)
    comptime CUDNN_ATTR_ENGINE_GLOBAL_INDEX = Self(1301)
    comptime CUDNN_ATTR_ENGINE_KNOB_INFO = Self(1302)
    comptime CUDNN_ATTR_ENGINE_NUMERICAL_NOTE = Self(1303)
    comptime CUDNN_ATTR_ENGINE_LAYOUT_INFO = Self(1304)
    comptime CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE = Self(1305)
    comptime CUDNN_ATTR_ENGINE_SM_COUNT_TARGET = Self(1306)
    comptime CUDNN_ATTR_ENGINE_DEVICEPROP = Self(1307)
    comptime CUDNN_ATTR_ENGINE_DISABLE_CLUSTER_COOPERATIVE = Self(1308)
    comptime CUDNN_ATTR_MATMUL_COMP_TYPE = Self(1500)
    comptime CUDNN_ATTR_MATMUL_PADDING_VALUE = Self(1503)
    comptime CUDNN_ATTR_OPERATION_MATMUL_ADESC = Self(1520)
    comptime CUDNN_ATTR_OPERATION_MATMUL_BDESC = Self(1521)
    comptime CUDNN_ATTR_OPERATION_MATMUL_CDESC = Self(1522)
    comptime CUDNN_ATTR_OPERATION_MATMUL_DESC = Self(1523)
    comptime CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = Self(
        1524
    )
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = Self(1525)
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = Self(1526)
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = Self(1527)
    comptime CUDNN_ATTR_REDUCTION_OPERATOR = Self(1600)
    comptime CUDNN_ATTR_REDUCTION_COMP_TYPE = Self(1601)
    comptime CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC = Self(1602)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_XDESC = Self(1610)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_YDESC = Self(1611)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_DESC = Self(1612)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = Self(1620)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = Self(1621)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = Self(1622)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = Self(1623)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = Self(1624)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = Self(1625)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = Self(1626)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = Self(1627)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = Self(1628)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = Self(1629)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = Self(1630)
    comptime CUDNN_ATTR_RESAMPLE_MODE = Self(1700)
    comptime CUDNN_ATTR_RESAMPLE_COMP_TYPE = Self(1701)
    comptime CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS = Self(1702)
    comptime CUDNN_ATTR_RESAMPLE_POST_PADDINGS = Self(1703)
    comptime CUDNN_ATTR_RESAMPLE_PRE_PADDINGS = Self(1704)
    comptime CUDNN_ATTR_RESAMPLE_STRIDES = Self(1705)
    comptime CUDNN_ATTR_RESAMPLE_WINDOW_DIMS = Self(1706)
    comptime CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION = Self(1707)
    comptime CUDNN_ATTR_RESAMPLE_PADDING_MODE = Self(1708)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC = Self(1710)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC = Self(1711)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = Self(1712)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA = Self(1713)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA = Self(1714)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC = Self(1716)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC = Self(1720)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC = Self(1721)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = Self(1722)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA = Self(1723)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA = Self(1724)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC = Self(1725)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC = Self(1726)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC = Self(1727)
    comptime CUDNN_ATTR_OPERATION_CONCAT_AXIS = Self(1800)
    comptime CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS = Self(1801)
    comptime CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = Self(1802)
    comptime CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC = Self(1803)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_MODE = Self(1900)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = Self(1901)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_VALUE = Self(1902)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_XDESC = Self(1903)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_YDESC = Self(1904)
    comptime CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC = Self(1950)
    comptime CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC = Self(1951)
    comptime CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC = Self(1952)
    comptime CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC = Self(1953)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_MODE = Self(2000)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_PHASE = Self(2001)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_XDESC = Self(2002)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC = Self(2003)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = Self(2004)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = Self(2005)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC = Self(2006)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = Self(2007)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = Self(2008)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = Self(2009)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = Self(2010)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = Self(2011)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = Self(2012)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_YDESC = Self(2013)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = Self(2014)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_MODE = Self(2100)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_XDESC = Self(2101)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC = Self(2102)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = Self(2103)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = Self(2104)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = Self(2105)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC = Self(2106)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = Self(2107)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC = Self(2108)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = Self(2109)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = Self(2110)
    comptime CUDNN_ATTR_OPERATION_RESHAPE_XDESC = Self(2200)
    comptime CUDNN_ATTR_OPERATION_RESHAPE_YDESC = Self(2201)
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC = Self(2250)
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC = Self(2251)
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH = Self(
        2252
    )
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH = Self(
        2253
    )
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS = Self(2254)
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE = Self(2255)
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC = Self(
        2256
    )
    comptime CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC = Self(
        2257
    )
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC = Self(2270)
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC = Self(2271)
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH = Self(
        2272
    )
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH = Self(
        2273
    )
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS = Self(2274)
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE = Self(2275)
    comptime CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE = Self(2276)
    comptime CUDNN_ATTR_RNG_DISTRIBUTION = Self(2300)
    comptime CUDNN_ATTR_RNG_NORMAL_DIST_MEAN = Self(2301)
    comptime CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = Self(2302)
    comptime CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM = Self(2303)
    comptime CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM = Self(2304)
    comptime CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY = Self(2305)
    comptime CUDNN_ATTR_OPERATION_RNG_YDESC = Self(2310)
    comptime CUDNN_ATTR_OPERATION_RNG_SEED = Self(2311)
    comptime CUDNN_ATTR_OPERATION_RNG_DESC = Self(2312)
    comptime CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC = Self(2313)
    comptime CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH = Self(2400)
    comptime CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED = Self(2401)
    comptime CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION = Self(2402)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC = Self(2500)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC = Self(2501)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC = Self(2502)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC = Self(2503)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE = Self(2504)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC = Self(2600)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC = Self(2601)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC = Self(2602)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC = Self(2603)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE = Self(2604)
    comptime CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE = Self(2605)
    comptime CUDNN_ATTR_DEVICEPROP_DEVICE_ID = Self(2700)
    comptime CUDNN_ATTR_DEVICEPROP_HANDLE = Self(2701)
    comptime CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION = Self(2702)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC = Self(2800)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC = Self(2801)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC = Self(2802)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC = Self(2803)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = Self(2804)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = Self(2805)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC = Self(2806)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC = Self(2807)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC = Self(2808)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = Self(2809)
    comptime CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = Self(2810)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC = Self(2851)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC = Self(2852)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC = Self(2853)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC = Self(2854)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC = Self(2855)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC = Self(2856)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC = Self(2857)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC = Self(2858)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC = Self(2859)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC = Self(2860)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC = Self(2861)
    comptime CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC = Self(2862)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE = Self(2900)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC = Self(2901)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC = Self(2902)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC = Self(2903)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC = Self(
        2904
    )
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC = Self(2905)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC = Self(
        2906
    )
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC = Self(2907)
    comptime CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K = Self(2908)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_ATTR_POINTWISE_MODE:
            return writer.write_string("CUDNN_ATTR_POINTWISE_MODE")
        if self is Self.CUDNN_ATTR_POINTWISE_MATH_PREC:
            return writer.write_string("CUDNN_ATTR_POINTWISE_MATH_PREC")
        if self is Self.CUDNN_ATTR_POINTWISE_NAN_PROPAGATION:
            return writer.write_string("CUDNN_ATTR_POINTWISE_NAN_PROPAGATION")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
            return writer.write_string("CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
            return writer.write_string("CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
            return writer.write_string(
                "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE"
            )
        if self is Self.CUDNN_ATTR_POINTWISE_ELU_ALPHA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_ELU_ALPHA")
        if self is Self.CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA")
        if self is Self.CUDNN_ATTR_POINTWISE_SWISH_BETA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_SWISH_BETA")
        if self is Self.CUDNN_ATTR_POINTWISE_AXIS:
            return writer.write_string("CUDNN_ATTR_POINTWISE_AXIS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_COMP_TYPE")
        if self is Self.CUDNN_ATTR_CONVOLUTION_CONV_MODE:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_CONV_MODE")
        if self is Self.CUDNN_ATTR_CONVOLUTION_DILATIONS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_DILATIONS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES")
        if self is Self.CUDNN_ATTR_CONVOLUTION_POST_PADDINGS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_POST_PADDINGS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_MODE:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_MODE")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_RESULTS:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_RESULTS")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_DEVICEPROP:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_DEVICEPROP")
        if self is Self.CUDNN_ATTR_ENGINECFG_ENGINE:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_ENGINE")
        if self is Self.CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO")
        if self is Self.CUDNN_ATTR_ENGINECFG_KNOB_CHOICES:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_KNOB_CHOICES")
        if self is Self.CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE")
        if self is Self.CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED:
            return writer.write_string(
                "CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_HANDLE:
            return writer.write_string("CUDNN_ATTR_EXECUTION_PLAN_HANDLE")
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
            return writer.write_string("CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE")
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
            return writer.write_string("CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP")
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID:
            return writer.write_string("CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID")
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_SIZE:
            return writer.write_string("CUDNN_ATTR_INTERMEDIATE_INFO_SIZE")
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS:
            return writer.write_string(
                "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS"
            )
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES:
            return writer.write_string(
                "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES"
            )
        if self is Self.CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE:
            return writer.write_string("CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE")
        if self is Self.CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR"
            )
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_BDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_BDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_DXDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_DXDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_DYDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_DYDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_TDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_TDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_HANDLE:
            return writer.write_string("CUDNN_ATTR_OPERATIONGRAPH_HANDLE")
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_OPS:
            return writer.write_string("CUDNN_ATTR_OPERATIONGRAPH_OPS")
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
            return writer.write_string(
                "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT"
            )
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED:
            return writer.write_string(
                "CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED"
            )
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY:
            return writer.write_string(
                "CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY"
            )
        if self is Self.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT:
            return writer.write_string("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT")
        if self is Self.CUDNN_ATTR_TENSOR_DATA_TYPE:
            return writer.write_string("CUDNN_ATTR_TENSOR_DATA_TYPE")
        if self is Self.CUDNN_ATTR_TENSOR_DIMENSIONS:
            return writer.write_string("CUDNN_ATTR_TENSOR_DIMENSIONS")
        if self is Self.CUDNN_ATTR_TENSOR_STRIDES:
            return writer.write_string("CUDNN_ATTR_TENSOR_STRIDES")
        if self is Self.CUDNN_ATTR_TENSOR_VECTOR_COUNT:
            return writer.write_string("CUDNN_ATTR_TENSOR_VECTOR_COUNT")
        if self is Self.CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION:
            return writer.write_string("CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION")
        if self is Self.CUDNN_ATTR_TENSOR_UNIQUE_ID:
            return writer.write_string("CUDNN_ATTR_TENSOR_UNIQUE_ID")
        if self is Self.CUDNN_ATTR_TENSOR_IS_VIRTUAL:
            return writer.write_string("CUDNN_ATTR_TENSOR_IS_VIRTUAL")
        if self is Self.CUDNN_ATTR_TENSOR_IS_BY_VALUE:
            return writer.write_string("CUDNN_ATTR_TENSOR_IS_BY_VALUE")
        if self is Self.CUDNN_ATTR_TENSOR_REORDERING_MODE:
            return writer.write_string("CUDNN_ATTR_TENSOR_REORDERING_MODE")
        if self is Self.CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC:
            return writer.write_string("CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_WORKSPACE:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_WORKSPACE")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS:
            return writer.write_string(
                "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS"
            )
        if self is Self.CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES:
            return writer.write_string(
                "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES"
            )
        if self is Self.CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES:
            return writer.write_string(
                "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES"
            )
        if self is Self.CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID:
            return writer.write_string("CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID")
        if self is Self.CUDNN_ATTR_LAYOUT_INFO_TYPES:
            return writer.write_string("CUDNN_ATTR_LAYOUT_INFO_TYPES")
        if self is Self.CUDNN_ATTR_KNOB_INFO_TYPE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_TYPE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_STRIDE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_STRIDE")
        if self is Self.CUDNN_ATTR_ENGINE_OPERATION_GRAPH:
            return writer.write_string("CUDNN_ATTR_ENGINE_OPERATION_GRAPH")
        if self is Self.CUDNN_ATTR_ENGINE_GLOBAL_INDEX:
            return writer.write_string("CUDNN_ATTR_ENGINE_GLOBAL_INDEX")
        if self is Self.CUDNN_ATTR_ENGINE_KNOB_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINE_KNOB_INFO")
        if self is Self.CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
            return writer.write_string("CUDNN_ATTR_ENGINE_NUMERICAL_NOTE")
        if self is Self.CUDNN_ATTR_ENGINE_LAYOUT_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINE_LAYOUT_INFO")
        if self is Self.CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
            return writer.write_string("CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE")
        if self is Self.CUDNN_ATTR_ENGINE_SM_COUNT_TARGET:
            return writer.write_string("CUDNN_ATTR_ENGINE_SM_COUNT_TARGET")
        if self is Self.CUDNN_ATTR_ENGINE_DEVICEPROP:
            return writer.write_string("CUDNN_ATTR_ENGINE_DEVICEPROP")
        if self is Self.CUDNN_ATTR_ENGINE_DISABLE_CLUSTER_COOPERATIVE:
            return writer.write_string(
                "CUDNN_ATTR_ENGINE_DISABLE_CLUSTER_COOPERATIVE"
            )
        if self is Self.CUDNN_ATTR_MATMUL_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_MATMUL_COMP_TYPE")
        if self is Self.CUDNN_ATTR_MATMUL_PADDING_VALUE:
            return writer.write_string("CUDNN_ATTR_MATMUL_PADDING_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_ADESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_ADESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_BDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_BDESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_CDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_CDESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_DESC")
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_REDUCTION_OPERATOR:
            return writer.write_string("CUDNN_ATTR_REDUCTION_OPERATOR")
        if self is Self.CUDNN_ATTR_REDUCTION_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_REDUCTION_COMP_TYPE")
        if self is Self.CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC:
            return writer.write_string("CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS"
            )
        if self is Self.CUDNN_ATTR_RESAMPLE_MODE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_MODE")
        if self is Self.CUDNN_ATTR_RESAMPLE_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_COMP_TYPE")
        if self is Self.CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS")
        if self is Self.CUDNN_ATTR_RESAMPLE_POST_PADDINGS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_POST_PADDINGS")
        if self is Self.CUDNN_ATTR_RESAMPLE_PRE_PADDINGS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_PRE_PADDINGS")
        if self is Self.CUDNN_ATTR_RESAMPLE_STRIDES:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_STRIDES")
        if self is Self.CUDNN_ATTR_RESAMPLE_WINDOW_DIMS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_WINDOW_DIMS")
        if self is Self.CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION")
        if self is Self.CUDNN_ATTR_RESAMPLE_PADDING_MODE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_PADDING_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_AXIS:
            return writer.write_string("CUDNN_ATTR_OPERATION_CONCAT_AXIS")
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_VALUE:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_PHASE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_PHASE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESHAPE_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESHAPE_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESHAPE_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESHAPE_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH"
            )
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH"
            )
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE"
            )
        if self is Self.CUDNN_ATTR_RNG_DISTRIBUTION:
            return writer.write_string("CUDNN_ATTR_RNG_DISTRIBUTION")
        if self is Self.CUDNN_ATTR_RNG_NORMAL_DIST_MEAN:
            return writer.write_string("CUDNN_ATTR_RNG_NORMAL_DIST_MEAN")
        if self is Self.CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION:
            return writer.write_string(
                "CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION"
            )
        if self is Self.CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM:
            return writer.write_string("CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM")
        if self is Self.CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM:
            return writer.write_string("CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM")
        if self is Self.CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY:
            return writer.write_string(
                "CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RNG_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_SEED:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_SEED")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC")
        if self is Self.CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH:
            return writer.write_string(
                "CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH"
            )
        if self is Self.CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED:
            return writer.write_string(
                "CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED"
            )
        if self is Self.CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION:
            return writer.write_string(
                "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE"
            )
        if self is Self.CUDNN_ATTR_DEVICEPROP_DEVICE_ID:
            return writer.write_string("CUDNN_ATTR_DEVICEPROP_DEVICE_ID")
        if self is Self.CUDNN_ATTR_DEVICEPROP_HANDLE:
            return writer.write_string("CUDNN_ATTR_DEVICEPROP_HANDLE")
        if self is Self.CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION:
            return writer.write_string(
                "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC")
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC
        ):
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K"
            )
        abort("invalid cudnnBackendAttributeName_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendAttributeName_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendDestroyDescriptor(
    descriptor: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendDestroyDescriptor",
        def(type_of(descriptor)) thin -> cudnnStatus_t,
    ]()(descriptor)
