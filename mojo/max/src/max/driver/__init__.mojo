# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to interact with devices.

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
    MutableInputTensor,
    OutputTensor,
    IOSpec,
    Input,
    Output,
    MutableInput,
)
from max.tensor import RuntimeTensorSpec

from ._accelerator import accelerator, Accelerator, CompiledDeviceKernel
from .anytensor import AnyMemory, AnyMojoValue, AnyTensor
from .device import Device, cpu
from .device_memory import DeviceMemory, DeviceTensor
from .tensor import Tensor
