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

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext, DeviceMulticastBuffer


fn test_multicast_memory(contexts: List[DeviceContext]) raises:
    alias alloc_len = 128 * 1024
    alias dtype = DType.int32

    var multicast_buf = DeviceMulticastBuffer[dtype](contexts, alloc_len)

    for i in range(len(contexts)):
        var dev_buf = multicast_buf.unicast_buffer_for(contexts[i])
        with dev_buf.map_to_host() as host_buf:
            for i in range(len(host_buf)):
                host_buf[i] = i * 2

    print(multicast_buf.unicast_buffer_for(contexts[0]))


fn main() raises:
    var ctx0 = create_test_device_context(device_id=0)
    if not ctx0.supports_multicast():
        print("Multicast memory not supported")
        return

    var num_gpus = DeviceContext.number_of_devices(api="gpu")
    if num_gpus < 2:
        print("Too few devices")
        return

    var ctx1 = create_test_device_context(device_id=1)

    test_multicast_memory([ctx0, ctx1])
