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

# Implementation of DeviceContext backed by the HAL

from std.memory import ArcPointer
from std.os import getenv
from std.sys._hal import Driver, Device, Context, Stream, get_device_spec

from .info import GPUInfo


struct DeviceContext(ImplicitlyCopyable, RegisterPassable):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/develop/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/docs/manual/errors/#use-a-context-manager). For example:

    ```mojo
    from std.gpu.host import DeviceContext
    from std.gpu import thread_idx

    def kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives an opaque `DeviceContextPtr`, which provides
    a `get_device_context()` method to retrieve the device context:

    ```text
    from std.runtime.asyncrt import DeviceContextPtr
    from compiler import register

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        def execute(ctx_ptr: DeviceContextPtr) raises:
            var ctx = ctx_ptr.get_device_context()
            ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    comptime device_spec = get_device_spec[0]()

    comptime default_device_info = GPUInfo.from_target[
        Self.device_spec.target.value
    ]()

    var _driver: ArcPointer[Driver]
    var _device: ArcPointer[Device[Self.device_spec]]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _stream: ArcPointer[Stream[Self.device_spec]]

    @always_inline
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = String(Self.default_device_info.api),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Raises:
            If device initialization fails or the specified device is not available.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """

        var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
        if not plugin_spec:
            raise Error("MODULAR_DRIVER_PLUGINS not set")

        self._driver = Driver.create(plugin_spec)

        # Validate that the loaded plugin matches the requested api.
        var driver_name = self._driver[].get_name()
        if String(driver_name).lower() != String(api).lower():
            raise Error(
                String(
                    t"Requested API {api} not supported by driver {driver_name}"
                )
            )

        # TODO: DRIV-163 - Use real device_id
        self._device = self._driver[].get_device[0]()
        self._context = self._device[].get_context()
        self._stream = self._context[].create_stream()

    def __enter__(var self) -> Self:
        """Enables the use of `DeviceContext` in a `with` statement context manager.

        Returns:
            The `DeviceContext` instance to be used within the context manager block.
        """
        return self^

    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If the operation fails.
        """
        self._stream[].synchronize()

    def id(self) -> Int64:
        """Returns the ID associated with this device.

        Returns:
            The unique device ID as an `Int64`.
        """
        return self._device[].id
