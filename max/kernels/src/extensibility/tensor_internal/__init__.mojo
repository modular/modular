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
"""APIs to create and manage tensors in a graph."""

from .io_spec import (
    FusedInput,
    FusedOutput,
    Input,
    IOSpec,
    IOUnknown,
    MutableInput,
    Output,
    _FusedComputeOutput,
)
from .managed_tensor_slice import (
    DynamicTensor,
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    StaticTensorSpec,
    VariadicTensors,
    _FusedComputeOutputTensor,
    _input_fusion_hook_impl,
    _mixed_precision_input_fusion_hook_impl,
    _mixed_precision_output_fusion_hook_impl,
    _mixed_precision_compute_output_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    trace_slice_arg,
    view_copy_impl,
)
from .tensor_spec import RuntimeTensorSpec
from .operation_traits import (
    ElementwiseBinaryOp,
    ElementwiseBinaryComparisonOp,
    ElementwiseUnaryOp,
    ElementwiseUnaryMixedOp,
)
