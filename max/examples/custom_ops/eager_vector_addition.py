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

from pathlib import Path

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random

if __name__ == "__main__":
    mojo_kernels = Path(__file__).parent / "kernels"

    vector_width = 10
    dtype = DType.float32

    # Place the op on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Fill input vectors with random values.
    lhs = random.uniform([vector_width], dtype=dtype, device=device)
    rhs = random.uniform([vector_width], dtype=dtype, device=device)

    # Run the custom vector_addition op directly on the eager Tensors.
    result = F.custom(
        name="vector_addition",
        device=device,
        values=[lhs, rhs],
        out_types=[lhs.type],
        custom_extensions=mojo_kernels,
    )[0]

    # Copy values back to the CPU to be read.
    lhs = lhs.to(CPU())
    rhs = rhs.to(CPU())
    result = result.to(CPU())

    print("Left-hand-side values:")
    print(lhs)
    print()

    print("Right-hand-side values:")
    print(rhs)
    print()

    print("Graph result:")
    print(result)
    print()

    print("Expected result:")
    print(lhs + rhs)
