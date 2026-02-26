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

from math import align_up
from sys import size_of
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils._utils import InitializationType

# 512 MiB — larger than 2x the infinity cache on MI300x (256 MiB)
# and larger than 2x the L2 cache on NVIDIA GPUs (A100=40MB, H100=50MB).
comptime CACHE_BUST_BYTES = 512 * 1024 * 1024


struct CacheBustingBuffer[dtype: DType](ImplicitlyCopyable):
    """Per-tensor cache busting buffer for GPU benchmarks.

    Owns a DeviceBuffer sized to exceed 2x the GPU cache. Each benchmark
    iteration uses a different offset into the buffer, preventing cache reuse
    and giving realistic bandwidth/latency numbers.

    When `enabled=False`, allocates only `tensor_size` elements and
    `offset()` always returns 0.
    """

    var _buf: DeviceBuffer[Self.dtype]
    var stride: Int
    var buffer_size: Int

    fn __init__(
        out self,
        tensor_size: Int,
        alignment: Int,
        ctx: DeviceContext,
        enabled: Bool = True,
    ) raises:
        self.stride = align_up(tensor_size, alignment)
        var full_buf_size = (
            align_up(CACHE_BUST_BYTES, self.stride * size_of[Self.dtype]())
            // size_of[Self.dtype]()
        )
        self.buffer_size = full_buf_size if enabled else self.stride
        var alloc = full_buf_size if enabled else tensor_size
        self._buf = ctx.enqueue_create_buffer[Self.dtype](alloc)

    @always_inline
    fn offset(self, iteration: Int) -> Int:
        """Element offset for a benchmark iteration. Returns 0 when disabled."""
        return (iteration * self.stride) % self.buffer_size

    @always_inline
    fn unsafe_ptr(self) -> DeviceBuffer[Self.dtype]._DevicePtr:
        """Raw device pointer to base of buffer."""
        return self._buf.unsafe_ptr()

    @always_inline
    fn offset_ptr(self, iteration: Int) -> DeviceBuffer[Self.dtype]._DevicePtr:
        """Device pointer offset to the window for this iteration."""
        return self._buf.unsafe_ptr() + self.offset(iteration)

    @always_inline
    fn device_buffer(self) -> DeviceBuffer[Self.dtype]:
        """Access underlying DeviceBuffer (copy, since DeviceBuffer is
        ImplicitlyCopyable)."""
        return self._buf

    @always_inline
    fn alloc_size(self) -> Int:
        """Number of elements allocated."""
        return len(self._buf)

    fn init_on_device(
        self, init_type: InitializationType, ctx: DeviceContext
    ) raises:
        """Initialize the entire buffer on the device."""
        from internal_utils._utils import init_vector_launch

        init_vector_launch[Self.dtype](
            self._buf, self.alloc_size(), init_type, ctx
        )
