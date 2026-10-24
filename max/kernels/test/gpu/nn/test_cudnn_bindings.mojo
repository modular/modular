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

from std.sys.info import size_of
from std.testing import TestSuite, assert_equal, assert_true

from _cudnn.backend import (
    cudnnBackendAttributeName_t,
    cudnnBackendAttributeType_t,
    cudnnBackendCreateDescriptor,
    cudnnBackendDescriptorType_t,
    cudnnBackendDestroyDescriptor,
    cudnnBackendFinalize,
    cudnnBackendSetAttribute,
)
from _cudnn.cnn_infer import (
    cudnnConvolutionFwdAlgoPerf_t,
    cudnnConvolutionMode_t,
    cudnnGetConvolutionForwardAlgorithm_v7,
    cudnnSetConvolution2dDescriptor,
)
from _cudnn.infer import (
    cudnnConvolutionFwdAlgo_t,
    cudnnDataType_t,
    cudnnDeterminism_t,
    cudnnGetErrorString,
    cudnnMathType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetTensor4dDescriptor,
    cudnnStatus_t,
    cudnnTensorFormat_t,
)
from max.gpu.host import DeviceContext
from nn.conv.conv import CuDNNConvMeta, check_cudnn_error


def test_enum_sizes() raises:
    # cuDNN enums are C ints.
    assert_equal(size_of[cudnnConvolutionFwdAlgo_t](), 4)
    assert_equal(size_of[cudnnConvolutionMode_t](), 4)
    assert_equal(size_of[cudnnDataType_t](), 4)
    assert_equal(size_of[cudnnDeterminism_t](), 4)
    assert_equal(size_of[cudnnMathType_t](), 4)
    assert_equal(size_of[cudnnStatus_t](), 4)
    assert_equal(size_of[cudnnTensorFormat_t](), 4)


def test_perf_struct_layout() raises:
    # Write a record at the offsets of the C struct and read it back.
    assert_equal(size_of[cudnnConvolutionFwdAlgoPerf_t](), 48)
    var raw_alloc = alloc[UInt8]({length = 48}).into_managed()
    var raw = raw_alloc.unsafe_ptr()
    for i in range(48):
        raw.unsafe_offset(i)[] = 0
    raw.unsafe_bitcast[Int32]()[] = 6
    raw.unsafe_offset(4).unsafe_bitcast[Int32]()[] = 3000
    raw.unsafe_offset(16).unsafe_bitcast[Int]()[] = 123456789
    raw.unsafe_offset(24).unsafe_bitcast[Int32]()[] = 1
    raw.unsafe_offset(28).unsafe_bitcast[Int32]()[] = 2

    ref perf = raw.unsafe_bitcast[cudnnConvolutionFwdAlgoPerf_t]()[]
    assert_true(
        perf.algo
        == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
    )
    assert_equal(perf.status._value, 3000)
    assert_equal(perf.memory, 123456789)
    assert_true(perf.determinism == cudnnDeterminism_t.CUDNN_DETERMINISTIC)
    assert_true(
        perf.mathType == cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
    )


def test_algo_count_out_param() raises:
    # cuDNN writes a C int through returned_algo_count. The word right after
    # it must survive the call.
    comptime guard = Int32(0x5A5A5A5A)
    with DeviceContext() as ctx:
        _ = ctx
        var meta = CuDNNConvMeta()
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                meta.ptr_input_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                cudnnDataType_t.CUDNN_DATA_BFLOAT16,
                1,
                8,
                16,
                16,
            )
        )
        check_cudnn_error(
            cudnnSetFilter4dDescriptor(
                meta.ptr_filter_desc,
                cudnnDataType_t.CUDNN_DATA_BFLOAT16,
                cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
                8,
                8,
                3,
                3,
            )
        )
        check_cudnn_error(
            cudnnSetConvolution2dDescriptor(
                meta.ptr_conv_desc,
                1,
                1,
                1,
                1,
                1,
                1,
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
            )
        )
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                meta.ptr_output_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                cudnnDataType_t.CUDNN_DATA_BFLOAT16,
                1,
                8,
                16,
                16,
            )
        )

        comptime max_algos = 8
        var perf_alloc = alloc[cudnnConvolutionFwdAlgoPerf_t](
            {length = max_algos}
        ).into_managed()
        var counts_alloc = alloc[Int32]({length = 2}).into_managed()
        var counts = counts_alloc.unsafe_ptr()
        counts[] = 0
        counts.unsafe_offset(1)[] = guard

        check_cudnn_error(
            cudnnGetConvolutionForwardAlgorithm_v7(
                meta.ptr_handle,
                meta.ptr_input_desc,
                meta.ptr_filter_desc,
                meta.ptr_conv_desc,
                meta.ptr_output_desc,
                max_algos,
                counts,
                perf_alloc.unsafe_ptr(),
            )
        )

        assert_equal(counts.unsafe_offset(1)[], guard)
        var returned = Int(counts[])
        assert_true(returned >= 1 and returned <= max_algos)
        for i in range(returned):
            var algo = Int(
                perf_alloc.unsafe_ptr().unsafe_offset(i)[].algo._value
            )
            assert_true(algo >= 0 and algo < max_algos)


def test_status_names_match_library() raises:
    # cuDNN names its own status codes, so a wrong value prints a wrong name.
    var statuses = [
        cudnnStatus_t.CUDNN_STATUS_SUCCESS,
        cudnnStatus_t.CUDNN_STATUS_NOT_INITIALIZED,
        cudnnStatus_t.CUDNN_STATUS_BAD_PARAM,
        cudnnStatus_t.CUDNN_STATUS_BAD_PARAM_NULL_POINTER,
        cudnnStatus_t.CUDNN_STATUS_NOT_SUPPORTED,
        cudnnStatus_t.CUDNN_STATUS_NOT_SUPPORTED_SHAPE,
        cudnnStatus_t.CUDNN_STATUS_INTERNAL_ERROR,
        cudnnStatus_t.CUDNN_STATUS_ALLOC_FAILED,
        cudnnStatus_t.CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
        cudnnStatus_t.CUDNN_STATUS_EXECUTION_FAILED,
    ]
    for status in statuses:
        assert_equal(
            String(status),
            String(
                unsafe_from_utf8_ptr=cudnnGetErrorString(status).unsafe_bitcast[
                    UInt8
                ]()
            ),
        )


def test_bad_param_status() raises:
    with DeviceContext() as ctx:
        _ = ctx
        var meta = CuDNNConvMeta()
        var status = cudnnSetTensor4dDescriptor(
            meta.ptr_input_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            -1,
            8,
            16,
            16,
        )
        assert_equal(String(status), "CUDNN_STATUS_BAD_PARAM")


def test_backend_convolution_attributes() raises:
    # cuDNN rejects an attribute name the descriptor does not have, so this
    # checks the attribute and descriptor values against the library.
    with DeviceContext() as ctx:
        _ = ctx
        var maybe_desc: Optional[OpaquePointer[AnyOrigin[mut=True]]] = None
        check_cudnn_error(
            cudnnBackendCreateDescriptor(
                cudnnBackendDescriptorType_t.CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                Pointer(to=maybe_desc).unsafe_bitcast[
                    OpaquePointer[AnyOrigin[mut=True]]
                ](),
            )
        )
        var desc = maybe_desc.value()

        var comp_type = cudnnDataType_t.CUDNN_DATA_FLOAT
        var mode = cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION
        var spatial_dims = Int64(2)
        var ones = [Int64(1), Int64(1)]
        var zeros = [Int64(0), Int64(0)]
        comptime Attr = cudnnBackendAttributeName_t
        comptime Type = cudnnBackendAttributeType_t
        check_cudnn_error(
            cudnnBackendSetAttribute(
                desc,
                Attr.CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                Type.CUDNN_TYPE_DATA_TYPE,
                1,
                Pointer(to=comp_type).unsafe_bitcast[NoneType](),
            )
        )
        check_cudnn_error(
            cudnnBackendSetAttribute(
                desc,
                Attr.CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                Type.CUDNN_TYPE_CONVOLUTION_MODE,
                1,
                Pointer(to=mode).unsafe_bitcast[NoneType](),
            )
        )
        check_cudnn_error(
            cudnnBackendSetAttribute(
                desc,
                Attr.CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                Type.CUDNN_TYPE_INT64,
                1,
                Pointer(to=spatial_dims).unsafe_bitcast[NoneType](),
            )
        )
        for attr in [
            Attr.CUDNN_ATTR_CONVOLUTION_DILATIONS,
            Attr.CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
        ]:
            check_cudnn_error(
                cudnnBackendSetAttribute(
                    desc,
                    attr,
                    Type.CUDNN_TYPE_INT64,
                    2,
                    ones.unsafe_ptr().unsafe_bitcast[NoneType](),
                )
            )
        for attr in [
            Attr.CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            Attr.CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
        ]:
            check_cudnn_error(
                cudnnBackendSetAttribute(
                    desc,
                    attr,
                    Type.CUDNN_TYPE_INT64,
                    2,
                    zeros.unsafe_ptr().unsafe_bitcast[NoneType](),
                )
            )
        check_cudnn_error(cudnnBackendFinalize(desc))
        check_cudnn_error(cudnnBackendDestroyDescriptor(desc))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
