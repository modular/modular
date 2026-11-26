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

# DOC: develop/build-custom-ops.mdx

from pathlib import Path

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.functional import custom_op
from max.tensor import Tensor

mojo_kernels = Path(__file__).parent / "kernels"

# No input binds k, so the kernel's shape function computes it at run time.
n, k = custom_op.Symbols("n", "k")
keep_positive = custom_op.declare(
    "keep_positive",
    inputs={"x": custom_op.TemplateType(DType.float32, [n])},
    outputs=[custom_op.TemplateType(DType.float32, [k])],
    custom_extensions=[mojo_kernels],
)

if __name__ == "__main__":
    values = np.random.uniform(-1.0, 1.0, size=10).astype(np.float32)
    expected = values[values > 0]

    # The kernel is registered for CPUs only, so the input goes on the CPU.
    result = keep_positive(Tensor.from_dlpack(values).to(CPU()))
    assert isinstance(result, Tensor)
    result_values = result.to_numpy()

    assert result_values.shape == expected.shape, result_values.shape
    np.testing.assert_array_equal(result_values, expected)
    print("Input:")
    print(values)
    print()
    print("Positive values:")
    print(result_values)
