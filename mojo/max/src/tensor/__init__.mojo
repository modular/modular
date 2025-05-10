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

# Don't put public documentation strings in this file. The docs build does not
# recognize these re-exported definitions, so it won't pick up an docs here.
# The docs are generated from the original implementations.

from tensor_internal import (
    DynamicTensor,
    ManagedTensorSlice,
    StaticTensorSpec,
    InputTensor,
    OutputTensor,
    IOSpec,
    Input,
    Output,
    MutableInput,
    VariadicTensors,
    InputVariadicTensors,
    _indexing,
    foreach,
)

# Note to make it hard to have circular dependencies, the impl lives elsewhere
from tensor_internal.tensor import Tensor
from tensor_internal.tensor_shape import TensorShape
from tensor_internal.tensor_spec import RuntimeTensorSpec, TensorSpec
