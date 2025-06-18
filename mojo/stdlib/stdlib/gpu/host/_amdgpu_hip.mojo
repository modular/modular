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

from gpu.host import DeviceContext, DeviceStream
from gpu.host.device_context import (
    _CharPtr,
    _checked,
    _DeviceContextPtr,
    _DeviceStreamPtr,
)


struct _ihipDevice_t:
    pass


struct _ihipStream_t:
    pass


alias hipDevice_t = UnsafePointer[_ihipDevice_t]
alias hipStream_t = UnsafePointer[_ihipStream_t]


# Accessor function to get access to the underlying hipDevice_t from an abstract DeviceContext.
# Use `var hip_dev: hipDevice_t = HIP(ctx)` where ctx is a `DeviceContext` to get access to the
# underlying hipDevice_t.
@always_inline
fn HIP(ctx: DeviceContext) raises -> hipDevice_t:
    var result = hipDevice_t()
    # const char *AsyncRT_DeviceContext_hip_device(hipDevice_t *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_hip_device",
            _CharPtr,
            UnsafePointer[hipDevice_t],
            _DeviceContextPtr,
        ](
            UnsafePointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying hipStream_t from an abstract DeviceStream.
# Use `var hip_stream: hipStream_t = HIP(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying hipStream_t.
@always_inline
fn HIP(stream: DeviceStream) raises -> hipStream_t:
    var result = hipStream_t()
    # const char *AsyncRT_DeviceStream_hip_stream(hipStream_t *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_hip_stream",
            _CharPtr,
            UnsafePointer[hipStream_t],
            _DeviceStreamPtr,
        ](
            UnsafePointer(to=result),
            stream._handle,
        )
    )
    return result
