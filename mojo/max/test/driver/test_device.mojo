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

# RUN: %mojo %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/test-device
# RUN: %mojo-build %s -o %t/test-device
# RUN: %t/test-device

from max.driver import DeviceMemory, DeviceTensor, cpu
from max.tensor import TensorSpec
from memory import UnsafePointer
from testing import assert_equal, assert_true


def test_device():
    dev = cpu()
    assert_true("Device(type=cpu" in String(dev))

    dev2 = dev
    assert_true("Device(type=cpu" in String(dev2))


def test_device_memory():
    dev = cpu()
    alias type = DType.float32

    dt1 = dev.allocate(TensorSpec(DType.float32, 2, 2))
    assert_true("DeviceTensor(Device(type=cpu" in String(dt1))
    assert_true("Spec(2x2xfloat32))" in String(dt1))

    dt2 = dev.allocate(TensorSpec(DType.float32, 3, 2))
    assert_true("DeviceTensor(Device(type=cpu" in String(dt2))
    assert_true("Spec(3x2xfloat32))" in String(dt2))

    dt3 = dev.allocate(TensorSpec(type, 3, 2), String("foo"))
    assert_true("DeviceTensor(foo,Device(type=cpu" in String(dt3))
    assert_true("Spec(3x2xfloat32))" in String(dt3))

    dt4 = dev.allocate(bytecount=128)
    assert_true("DeviceMemory(Device(type=cpu" in String(dt4))
    assert_true("Bytecount(128)" in String(dt4))

    dt5 = dev.allocate(TensorSpec(type, 2))
    ptr = dt5.unsafe_ptr()
    t5 = dt5^.to_tensor[type, 1]()
    t5[0] = 22
    assert_equal(rebind[UnsafePointer[Scalar[type]]](ptr).load(), t5[0])


def test_take():
    var tensors = List[DeviceTensor]()
    var cpu = cpu()
    for _ in range(2):
        tensors.append(cpu.allocate(TensorSpec(DType.float32, 2, 2)))

    def consume_and_check(owned t: DeviceTensor):
        assert_true("DeviceTensor(Device(type=cpu" in String(t))
        assert_true("Spec(2x2xfloat32))" in String(t))

    for tensor in tensors:
        consume_and_check(tensor[].take())


def test_kv_cache():
    cpu = cpu()
    alias type = DType.float32
    alias shape = (2, 2)
    allocs = List[DeviceTensor]()
    for _ in range(2):
        allocs.append(cpu.allocate(TensorSpec(type, shape)))

    for t in allocs:
        assert_true("DeviceTensor(Device(type=cpu" in String(t[]))
        assert_true("Spec(2x2xfloat32))" in String(t[]))


def main():
    test_device()
    test_device_memory()
    test_kv_cache()
    test_take()
