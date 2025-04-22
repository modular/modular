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

from sys.param_env import env_get_string, is_defined

from gpu.host import DeviceContext


fn expect_eq[
    type: DType, size: Int, *Ts: Writable
](val: SIMD[type, size], expected: SIMD[type, size], *messages: *Ts) raises:
    if val != expected:
        var message = String(messages)
        raise Error("expect_eq failed: ", message)


fn expect_eq[*Ts: Writable](val: Bool, expected: Bool, *messages: *Ts) raises:
    if val != expected:
        var message = String(messages)
        raise Error("expect_eq failed: ", message)


fn api() -> String:
    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        alias api = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()

        @parameter
        if api == "gpu":
            return String(DeviceContext.device_api)
        return String(api)
    return "default"


fn create_test_device_context(*, device_id: Int = 0) raises -> DeviceContext:
    # Create an instance of the DeviceContext
    var test_ctx: DeviceContext

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        print("Using DeviceContext: V2 - " + api())
        test_ctx = DeviceContext(device_id=device_id, api=api())
    elif is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V1"]():
        raise Error("DeviceContextV1 is unsupported")
    else:
        print("Using DeviceContext: default")
        test_ctx = DeviceContext(device_id=device_id)

    return test_ctx
