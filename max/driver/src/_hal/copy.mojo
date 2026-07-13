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
"""Unified, transport-dispatching copy between HAL buffers."""

from .buffer import Buffer
from .device import DeviceSpec
from .plugin import RawDriver, QueueHandle
from .status import STATUS_INVALID_ARG, HALError
from std.memory import ArcPointer, memcpy


# ===----------------------------------------------------------------------=== #
# Transport dispatch
# ===----------------------------------------------------------------------=== #


def _enqueue_copy[
    device_spec: DeviceSpec
](
    raw: ArcPointer[RawDriver],
    queue: QueueHandle,
    queue_device_id: Int64,
    *,
    dst: Buffer[device_spec],
    src: Buffer[device_spec],
) raises HALError:
    """Validates and enqueues a buffer-to-buffer copy on `queue`.

    Transfers exactly `src.byte_size` bytes into the front of `dst`, routing
    by the residency of each operand: same-device device-to-device,
    cross-device device-to-device (a peer copy), pinned-host-to-device, or
    device-to-pinned-host. A purely host-side (pinned-to-pinned) copy has no
    queue transport and raises; use the synchronous free `copy` for it.

    The transfer runs on `queue`, so the device-resident operand it touches
    must live on the queue's device: `dst` for a to-device or same-device copy,
    `src` for a device-to-pinned-host copy. A pinned host operand is only a
    host pointer and may come from any device's context. A cross-device (peer)
    copy does not order itself against work in flight on other queues: the
    caller must ensure the source is ready before the copy and must not free
    or overwrite the source until `queue` is synchronized.

    Parameters:
        device_spec: The compilation target the buffers' memory lives on.

    Args:
        raw: The loaded driver plugin.
        queue: The queue to enqueue the transfer on.
        queue_device_id: Runtime id of the device backing `queue`, used to
            validate that the copy's device-resident operand lives on it.
        dst: Destination buffer.
        src: Source buffer.
    """
    if dst.is_host_pinned and src.is_host_pinned:
        raise HALError(
            STATUS_INVALID_ARG,
            message=String(
                "host-to-host copy cannot be enqueued; use the synchronous"
                " free `copy`"
            ),
        )
    if dst.byte_size < src.byte_size:
        raise HALError(
            STATUS_INVALID_ARG,
            message=String(
                t"copy overflow: src is {src.byte_size} bytes but dst holds"
                t" only {dst.byte_size} bytes"
            ),
        )
    var n = src.byte_size
    if n == 0:
        return
    if src.is_host_pinned:
        if dst._device_id() != queue_device_id:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"copy requires the device operand to reside on the"
                    t" queue's device (operand device {dst._device_id()},"
                    t" queue device {queue_device_id})"
                ),
            )
        var src_ptr = src._context[].memory_get_host_address[ImmutAnyOrigin](
            src
        )
        raw[].copy_to_device(
            queue, dst.view(byte_offset=0, byte_size=n)._view, src_ptr
        )
    elif dst.is_host_pinned:
        if src._device_id() != queue_device_id:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"copy requires the device operand to reside on the"
                    t" queue's device (operand device {src._device_id()},"
                    t" queue device {queue_device_id})"
                ),
            )
        var dst_ptr = dst._context[].memory_get_host_address[MutAnyOrigin](dst)
        raw[].copy_from_device(queue, dst_ptr, src.view()._view)
    elif dst._device_id() == src._device_id():
        raw[].copy_intra_device(
            queue,
            dst.view(byte_offset=0, byte_size=n)._view,
            src.view()._view,
        )
    else:
        # Both device-resident on different devices: a peer copy. The plugin
        # uses `queue`'s context as the destination context, so `dst` must live
        # on the queue's device; `src`'s context is passed so the plugin can
        # resolve the peer (source) device.
        if dst._device_id() != queue_device_id:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"copy requires the device operand to reside on the"
                    t" queue's device (operand device {dst._device_id()},"
                    t" queue device {queue_device_id})"
                ),
            )
        raw[].copy_inter_device(
            queue,
            dst.view(byte_offset=0, byte_size=n)._view,
            src.view()._view,
            src._context[]._handle,
        )


def _sync_copy[
    device_spec: DeviceSpec
](
    raw: ArcPointer[RawDriver],
    *,
    dst: Buffer[device_spec],
    src: Buffer[device_spec],
) raises HALError:
    """Validates and performs a blocking buffer-to-buffer copy.

    Routes by residency to the plugin's synchronous, stream-less copy ops:
    pinned-host-to-device, device-to-pinned-host, same-device device-to-device,
    or cross-device device-to-device (a peer copy). Blocks until the transfer
    completes; unlike the queue path it creates no stream. Each transport runs
    on its device-resident operand's context — the destination for a to-device
    or same-device copy, the source for a device-to-pinned-host copy — so a
    pinned host operand may come from any device's context, and no pair of
    operands is required to share a device.

    Parameters:
        device_spec: The compilation target the buffers' memory lives on.

    Args:
        raw: The loaded driver plugin (shared by both operands' contexts).
        dst: Destination buffer.
        src: Source buffer.
    """
    if dst.byte_size < src.byte_size:
        raise HALError(
            STATUS_INVALID_ARG,
            message=String(
                t"copy overflow: src is {src.byte_size} bytes but dst holds"
                t" only {dst.byte_size} bytes"
            ),
        )
    var n = src.byte_size
    if n == 0:
        return
    if src.is_host_pinned:
        var src_ptr = src._context[].memory_get_host_address[ImmutAnyOrigin](
            src
        )
        raw[].copy_to_device_sync(
            dst._context[]._handle,
            dst.view(byte_offset=0, byte_size=n)._view,
            src_ptr,
        )
    elif dst.is_host_pinned:
        var dst_ptr = dst._context[].memory_get_host_address[MutAnyOrigin](dst)
        raw[].copy_from_device_sync(
            src._context[]._handle, dst_ptr, src.view()._view
        )
    elif dst._device_id() == src._device_id():
        raw[].copy_intra_device_sync(
            dst._context[]._handle,
            dst.view(byte_offset=0, byte_size=n)._view,
            src.view()._view,
        )
    else:
        raw[].copy_inter_device_sync(
            dst._context[]._handle,
            dst.view(byte_offset=0, byte_size=n)._view,
            src.view()._view,
            src._context[]._handle,
        )


# ===----------------------------------------------------------------------=== #
# Synchronous free `copy`
# ===----------------------------------------------------------------------=== #


def copy[
    device_spec: DeviceSpec
](*, dst: Buffer[device_spec], src: Buffer[device_spec]) raises HALError:
    """Synchronously copies buffer `src` into the front of buffer `dst`.

    Transfers exactly `src.byte_size` bytes; `dst` must be at least that
    large. The transport follows each operand's residency and blocks until it
    completes: two pinned host buffers are copied with a plain `memcpy` (they
    must not overlap); every other combination runs through the plugin's
    synchronous, stream-less copy ops (no queue is created), on the
    device-resident operand's context. Operands need not share a device —
    device-to-device across GPUs uses a peer transfer, and a pinned host buffer
    (only a host pointer) may pair with a device buffer on any GPU.

    Parameters:
        device_spec: The compilation target the buffers' memory lives on.

    Args:
        dst: Destination buffer.
        src: Source buffer.
    """
    if dst.is_host_pinned and src.is_host_pinned:
        if dst.byte_size < src.byte_size:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"copy overflow: src is {src.byte_size} bytes but dst"
                    t" holds only {dst.byte_size} bytes"
                ),
            )
        var n = src.byte_size
        if n == 0:
            return
        var dst_ptr = dst._context[].memory_get_host_address[MutAnyOrigin](dst)
        var src_ptr = src._context[].memory_get_host_address[ImmutAnyOrigin](
            src
        )
        memcpy(dest=dst_ptr, src=src_ptr, count=Int(n))
        return

    _sync_copy(dst._context[]._raw, dst=dst, src=src)
