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
"""Surface needed to write GraphCompiler kernel entry points.

Provides the `@register` / `@register_internal` decorators, the
`ManagedTensorSlice` type and supporting tensor specs/IO enums, plus the
elementwise operation traits.

For registering [custom operations](/max/develop/custom-ops/), use the Mojo
[@compiler.register](https://mojolang.org/docs/reference/decorators/compiler-register/)
decorator instead.
"""

from .decorators import *
from .managed_tensor_slice import *
from .operation_traits import *

# Underscore-prefixed names are skipped by `import *`, but the kernels and
# kv_cache packages reference them by name. Re-export them explicitly.
from .managed_tensor_slice import (
    _FusedComputeOutput,
    _FusedComputeOutputTensor,
    _FusedInputTensor,
    _FusedInputVariadicTensors,
    _FusedOutputTensor,
    _FusedOutputVariadicTensors,
    _MutableInputTensor,
    _MutableInputVariadicTensors,
    _dot_prod,
    _shape_types_compatible,
    _slice_to_tuple,
)
