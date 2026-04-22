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

from .types import (
    Handle,
    TensorDescriptor,
    ConvolutionDescriptor,
    Status,
    DataType,
    TensorLayout,
    ConvolutionMode,
    ConvFwdAlgorithm,
    ConvAlgoPerf,
    ConvSolution,
    Problem,
    Solution,
    FindOptions,
    ProblemDirection,
    TensorArgumentId,
    TensorArgument,
)
from .miopen import (
    miopenCreate,
    miopenDestroy,
    miopenSetStream,
    miopenCreateTensorDescriptor,
    miopenDestroyTensorDescriptor,
    miopenSetTensorDescriptor,
    miopenSetNdTensorDescriptorWithLayout,
    miopenSet4dTensorDescriptorEx,
    miopenCreateConvolutionDescriptor,
    miopenDestroyConvolutionDescriptor,
    miopenInitConvolutionNdDescriptor,
    miopenSetConvolutionGroupCount,
    miopenConvolutionForwardGetWorkSpaceSize,
    miopenFindConvolutionForwardAlgorithm,
    miopenConvolutionForward,
    miopenConvolutionForwardGetSolutionCount,
    miopenConvolutionForwardGetSolution,
    miopenConvolutionForwardCompileSolution,
    miopenConvolutionForwardImmediate,
    miopenCreateConvProblem,
    miopenDestroyProblem,
    miopenSetProblemTensorDescriptor,
    miopenCreateFindOptions,
    miopenDestroyFindOptions,
    miopenSetFindOptionPreallocatedWorkspace,
    miopenFindSolutions,
    miopenGetSolutionWorkspaceSize,
    miopenRunSolution,
    miopenDestroySolution,
)
from .utils import check_error
