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

from sys import external_call
from gpu.host.device_context import _CharPtr, _checked, _DeviceContextPtr


struct _MTLDevice:
    pass


alias MTLDevice = UnsafePointer[_MTLDevice]


# Accessor function to get access to the underlying MTLDevice from an abstract DeviceContext.
# Use `var metal_device: MTLDevice = metal_device(ctx)` where ctx is a `DeviceContext` to get access to the
# underlying MTLDevice.
@always_inline
fn metal_device(ctx: DeviceContext) raises -> MTLDevice:
    var result = MTLDevice()
    # const char *AsyncRT_DeviceContext_metal_device(MTL::Device **result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_metal_device",
            _CharPtr,
            UnsafePointer[MTLDevice],
            _DeviceContextPtr,
        ](
            UnsafePointer(to=result),
            ctx._handle,
        )
    )
    return result
