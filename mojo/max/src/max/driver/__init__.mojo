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
"""DEPRECATED:
The `max.driver` Mojo API is being deprecated in favor of the open source
`gpu.host` API, which has better ergonomics, and is more feature complete.

APIs to interact with devices.

Although there are several modules in this `max.driver` package, you'll get
everything you need from this top-level `driver` namespace, so you don't need
to import each module.

For example, the basic code you need to create tensor on CPU looks like this:

```mojo
from max.driver import Tensor, cpu
from testing import assert_equal
from max.tensor import TensorShape

def main():
    tensor = Tensor[DType.float32, rank=2](TensorShape(1,2))
    tensor[0, 0] = 1.0

    # You can also explicitly set the devices.
    device = cpu()
    new_tensor = Tensor[DType.float32, rank=2](TensorShape(1,2), device)
    new_tensor[0, 0] = 1.0

    # You can also create slices of tensor
    subtensor = tensor[:, :1]
    assert_equal(subtensor[0, 0], tensor[0, 0])
```
"""

from max._tensor_utils import (
    DynamicTensor,
    ManagedTensorSlice,
    StaticTensorSpec,
    InputTensor,
    OutputTensor,
    IOSpec,
    Input,
    Output,
    MutableInput,
)
from max.tensor import RuntimeTensorSpec

from ._accelerator import (
    accelerator,
    accelerator_count,
    Accelerator,
    CompiledDeviceKernel,
)
from .anytensor import AnyMemory, AnyMojoValue, AnyTensor
from .device import Device, cpu
from .device_memory import DeviceMemory, DeviceTensor
from .tensor import Tensor
