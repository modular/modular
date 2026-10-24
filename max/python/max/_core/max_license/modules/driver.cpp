//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// Driver module: exports Buffer class (renamed from Tensor)
#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/DLPack/Utils.h"
#include "_core/Driver/Buffer.h"
#include "_core/Driver/ChunkedStagingRegion.h"
#include "_core/Driver/CompletionFlag.h"
#include "_core/Driver/Device.h"
#include "_core/Driver/SymmetricPool.h"
#include "_core/Graph/Bindings.h"
#include "_core/Profiler/Trace.h"
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/optional.h"   // IWYU pragma: keep (type caster)
#include "nanobind/stl/pair.h"       // IWYU pragma: keep (type caster)
#include "nanobind/stl/shared_ptr.h" // IWYU pragma: keep (type casters)
#include "nanobind/stl/string.h"     // IWYU pragma: keep (type casters)
#include "nanobind/stl/vector.h"     // IWYU pragma: keep (type casters)
#include <dlpack/dlpack.h>
#include <stdexcept>

namespace nb = nanobind;

// For named argument shorthand.
// See:
// https://nanobind.readthedocs.io/en/stable/basics.html#keyword-and-default-arguments
using namespace nanobind::literals;
using namespace M;
using namespace Driver;

#if MODULAR_HAL_DEVICE_CONTEXT
extern "C" const char *MojoHAL_DeviceContext_enableAllPeerAccess();
extern "C" void MojoHAL_strfree(const char *s);
#else
extern "C" const char *AsyncRT_DeviceContext_enableAllPeerAccess();
#endif

// This function wraps the internals of the overloads of Tensor.to().
// In each of these, the method will return the same Python object if the
// requested device is the same as the one the device is already on, otherwise
// it will return a new tensor on the target device.
nb::typed<nb::object, Python::Buffer> static tensorTo(
    nb::pointer_and_handle<Python::Buffer> self,
    Python::PyDeviceTy &currentDevice, Python::PyDeviceTy &targetDevice,
    const Python::DeviceEvent *after) {
  // Pinned host memory and device memory can have the same
  // device context, but they're different memory spaces (host RAM
  // vs GPU VRAM). Always copy if the source is pinned memory.
  if (*currentDevice->ctx == *targetDevice->ctx && !self.p->isPinned()) {
    return nb::borrow(self.h);
  } else {
    return nb::cast(self.p->to(targetDevice, after));
  }
}

static const Python::DeviceEvent *
eventOrNull(const std::optional<Python::PyDeviceEventTy> &after) {
  return after ? after->get() : nullptr;
}

void M::Graph::Python::bindDriverModule(nb::module_ m) {
  namespace Python = M::Driver::Python;
  m.doc() = R"nbdoc(
        MAX Driver Python bindings.

        Provides low-level access to hardware devices and memory management.
    )nbdoc";

  nb::class_<Python::Device>(m, "Device", R"nbdoc(
      Represents a compute device available for tensor operations.

      This is the base class for :class:`CPU` and :class:`Accelerator`.
      Do not instantiate this class directly; use :class:`CPU` for host
      devices and :class:`Accelerator` for any hardware accelerator.

      .. invisible-code-block: python

          from max import driver

      .. code-block:: python

          from max import driver

          cpu = driver.CPU()
          device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
  )nbdoc")
      .def(
          "can_access",
          [](Python::Device &self, Python::Device &other) -> bool {
            return self.canAccess(other);
          },
          "other"_a,
          R"nbdoc(
         Checks if this device can directly access memory of another device.

         .. invisible-code-block: python

              from max import driver

         .. skip: next if(driver.accelerator_count() < 2, reason="requires two accelerators")

         .. code-block:: python

              from max import driver

              gpu0 = driver.Accelerator(id=0)
              gpu1 = driver.Accelerator(id=1)

              if gpu0.can_access(gpu1):
                  print("GPU0 can directly access GPU1 memory.")

         Args:
             other: The other device to check peer access against.

         Returns:
             ``True`` if peer access is possible, ``False`` otherwise.
         )nbdoc")
      .def("synchronize", &Python::Device::synchronize, R"nbdoc(
        Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def_prop_ro("is_host", &Python::Device::isHost, R"nbdoc(
        Whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device.is_host
      )nbdoc")
      .def_prop_ro("is_host_unified", &Python::Device::isHostUnified, R"nbdoc(
        Whether this device and the host draw from one physical memory pool.

        Reports hardware topology, so it does not predict whether a particular
        buffer is readable from the host.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device.is_host_unified
      )nbdoc")
      .def_prop_ro(
          "stats",
          [](Python::Device &self) {
            auto json = nb::module_::import_("json");
            return nb::borrow<nb::typed<nb::mapping, nb::str, nb::any>>(
                json.attr("loads")(self.getStats()));
          },
          R"nbdoc(
        Returns utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            stats = device.stats

        Returns:
            A dictionary containing device utilization statistics.
      )nbdoc")
      .def_prop_ro("max_single_alloc_size",
                   &Python::Device::getMaxSingleAllocSize,
                   "Largest single contiguous allocation, in bytes.")
      .def("_allocation_range", &Python::Device::getAllocationRange,
           nb::arg("address"),
           "Returns ``(base, size)`` of the driver allocation that contains "
           "``address``.")
      .def_prop_ro("label", &Python::Device::getLabel, R"nbdoc(
        Returns device label.

        Possible values are:

        - ``cpu`` for host devices.
        - ``gpu`` for accelerators.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device.label
      )nbdoc")
      .def_prop_ro("api", &Python::Device::getApi, R"nbdoc(
        Returns the API used to program the device.

        Possible values are:

        - ``cpu`` for host devices.
        - ``cuda`` for NVIDIA GPUs.
        - ``hip`` for AMD GPUs.
        - ``metal`` for Apple GPUs.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device.api
      )nbdoc")
      .def_prop_ro("architecture_name", &Python::Device::getArchName,
                   R"nbdoc(
        Returns the architecture name of the device.

        Examples of possible values:

        - ``gfx90a``, ``gfx942`` for AMD GPUs.
        - ``sm_80``, ``sm_86`` for NVIDIA GPUs.
        - CPU devices raise an exception.

        .. invisible-code-block: python

            from max import driver

        .. skip: next if(driver.accelerator_count() == 0, reason="requires an accelerator")

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.architecture_name
      )nbdoc")
      .def_prop_ro("model_name", &Python::Device::getModelName, R"nbdoc(
        Returns the model name of the device.

        Examples of possible values:

        - ``NVIDIA H100 80GB HBM3`` for an H100.
        - ``NVIDIA B200`` for a B200.
        - ``AMD Instinct MI300X`` for an MI300X.
        - ``cpu`` for a CPU device.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device.model_name
      )nbdoc")
      .def_prop_ro("id", &Python::Device::getId, R"nbdoc(
        Returns a zero-based device id.

        For a CPU device this is always 0.
        For GPU accelerators this is the ID of the device relative to this host.
        Along with the ``label``, an ID can uniquely identify a device,
        such as ``gpu:0`` or ``gpu:1``.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            device_id = device.id

        Returns:
            The device ID.
      )nbdoc")
      .def_prop_ro("default_queue", &Python::Device::defaultStream, R"nbdoc(
        Returns the default queue for this device.

        The default queue is initialized when the device object is created.

        Returns:
            The default execution queue for this device.
      )nbdoc")
      .def_prop_ro("is_compatible", &Python::Device::isCompatible, R"nbdoc(
        Returns whether this device is compatible with MAX.

        Returns:
            ``True`` if the device is compatible with MAX, ``False``
            otherwise.
      )nbdoc")
      .def(
          "__unsafe_enqueue_py_host_func",
          [](Python::Device &self, nb::callable &&fn) {
            self.defaultStream()->enqueueHostFunc(std::move(fn));
          },
          "fn"_a,
          R"nbdoc(
          Enqueues a Python callable to run on the host after preceding work.

          The callable runs on a driver thread once the device's default
          queue reaches this point, after all previously enqueued work has
          completed. It must not call any device APIs (per the
          ``cuLaunchHostFunc`` contract). Currently only supported on CUDA
          devices.

          This API is intentionally namespaced with a ``__unsafe_`` prefix
          to discourage casual use: there is no safety net for callbacks
          that capture state that outlives the compiled graph, and
          misuse can deadlock or corrupt memory.

          Args:
              fn (Callable[[], None]): A zero-argument callable.

          Raises:
              RuntimeError: If the underlying device does not support host
                  callbacks, or if the driver rejects the enqueue.
      )nbdoc")
      .def(
          "__unsafe_enqueue_async_py_host_func",
          [](Python::Device &self, nb::callable &&fn,
             const Python::CompletionFlag &flag, uint64_t value,
             const Python::CPU &cpu) {
            self.defaultStream()->enqueueAsyncHostFunc(std::move(fn), flag,
                                                       value, cpu);
          },
          "fn"_a, "flag"_a, "value"_a, "cpu"_a,
          R"nbdoc(
          Async kickoff variant of ``__unsafe_enqueue_py_host_func``.

          Like ``__unsafe_enqueue_py_host_func``, except the kickoff host
          node dispatches ``fn`` onto ``cpu``'s AsyncRT worker pool and
          returns immediately, so the GPU queue can proceed to
          subsequent nodes concurrently with ``fn`` running on an
          AsyncRT worker thread.

          When ``fn`` finishes, the worker atomic-stores ``value``
          (release ordering) to the 64-bit memory at ``flag``. Pair with
          ``DeviceQueue.wait_for_host_value(flag, value)`` on the same
          queue to gate the downstream consumer kernel.

          The trampoline keeps refcounts on ``flag``'s underlying MLRT
          allocation AND on ``cpu``'s AsyncRT CPUDevice, so neither can
          be released out from under the in-flight task even if the
          Python wrappers are GC'd between enqueue and signal.

          Args:
              fn (Callable[[], None]): A zero-argument callable.
              flag (CompletionFlag): The completion flag the worker will
                  signal when ``fn`` returns.
              value (int): The 64-bit value to store on completion
                  (matched against the
                  ``wait_for_host_value(flag, value)`` consumer).
              cpu (CPU): The CPU device whose AsyncRT worker pool will
                  execute ``fn``. Typically just ``max.driver.CPU()``;
                  callers needing fine-grained scheduling can pass a
                  specific CPU device.

          Raises:
              RuntimeError: If the underlying device does not support
                  host callbacks, if the supplied ``cpu`` has no
                  associated AsyncRT CPUDevice, or if the driver rejects
                  the enqueue.
      )nbdoc")
      .def("__str__", &Python::Device::getAsString)
      .def("__repr__", &Python::Device::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__hash__", &Python::Device::hash)
      .def(
          "_device_context_ptr",
          [](Python::Device &self) {
            auto *ctx = self.ctx.getPointer();
#if MODULAR_HAL_DEVICE_CONTEXT
            // Under the HAL backend the Mojo `DeviceContext` is the source of
            // truth and the C++ object merely wraps it. Mojo consumers
            // reconstruct via `DeviceContext._from_unsafe_cpp_handle`, which
            // expects the boxed Mojo handle - so hand out `mojoHandle()`
            // rather than the C++ wrapper.
            return reinterpret_cast<uintptr_t>(ctx->mojoHandle());
#else
            return reinterpret_cast<uintptr_t>(ctx);
#endif
          },
          "Gets the device context pointer.")
      .def_static("cpu", &Python::getCpuDevice, "id"_a = 0,
                  R"nbdoc(
        Creates a CPU device.

        Args:
            id: The device ID. Currently ignored.
      )nbdoc");

  // Note the use of __new__ here instead of __init__. This allows us to use a
  // factory function on the C++ side to produce the Accelerator type from a
  // cache. We could also use __init__ here if we changed the C++
  // implementation.
  nb::class_<Python::Accelerator, Python::Device>(m, "Accelerator", R"nbdoc(
        Represents any hardware accelerator attached to the host.

        The graph compiler reaches CUDA, HIP, Metal and plugin-provided
        backends alike through this one device class. Use
        ``isinstance(device, Accelerator)`` to mean "any non-CPU device",
        and :attr:`api` to tell the concrete backends apart.

        .. invisible-code-block: python

            from max import driver

        .. skip: next if(driver.accelerator_count() == 0, reason="requires an accelerator")

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            first = driver.Accelerator(id=0)

        Args:
            id: The zero-based ID of the accelerator to select. Defaults to
                -1, which selects the first available accelerator.

        Raises:
            ValueError: If the device cannot be created, for example when no
                compatible accelerator is available.
        )nbdoc")
      .def(nb::new_(&Python::getAccelerator), "id"_a = -1);

  nb::class_<Python::CPU, Python::Device>(m, "CPU", R"nbdoc(
          A host (CPU) device.

          .. code-block:: python

              from max import driver

              device = driver.CPU()
              device_id = device.id

          Args:
              id: The device ID. Defaults to 0.
       )nbdoc")
      .def(nb::new_(&Python::getCpuDevice), "id"_a = 0);

  nb::class_<Python::CompletionFlag>(m, "CompletionFlag", R"nbdoc(
          An 8-byte completion flag in pinned host memory mapped into a device's address space.

          Lets a CPU thread signal a GPU queue (or vice versa) by writing a
          64-bit value to a single location that is visible to both. Pair
          with :meth:`DeviceQueue.wait_for_host_value` or the
          ``mo.wait_host_value`` graph op to gate downstream GPU work on a
          host-produced result without a second queue or a blocking host
          callback.

          Currently requires a CUDA-backed :class:`Device`; constructing
          against any other backend raises ``RuntimeError``.

          .. invisible-code-block: python

              from max import driver

          .. skip: next if(driver.accelerator_count() == 0 or driver.Accelerator().api != "cuda", reason="requires a CUDA device")

          .. code-block:: python

              from max.driver import Accelerator, CompletionFlag

              flag = CompletionFlag(Accelerator())
              assert flag.load() == 0  # initialized to zero

          Args:
              device: A CUDA-backed device. Other backends raise
                  ``RuntimeError``.
       )nbdoc")
      .def(nb::init<const Python::Device &>(), "device"_a)
      .def_prop_ro("device_ptr", &Python::CompletionFlag::devicePtr,
                   R"nbdoc(
              Device-visible 64-bit address of the 8-byte slot.

              Suitable for passing to graph ops or queue APIs that wait
              on a memory value.
            )nbdoc")
      .def("reset", &Python::CompletionFlag::reset,
           R"nbdoc(
              Clears the flag back to ``0`` with a relaxed atomic store.

              Safe to call before any consumer has observed the address.
            )nbdoc")
      .def("signal", &Python::CompletionFlag::signal, "value"_a,
           R"nbdoc(
              Release-ordered store of ``value`` to the flag.

              Pairs with the GPU-side ``cuStreamWaitValue64`` (or a
              host-side acquire ``load``).

              Primary intended use is priming the flag at setup time so
              the first captured-graph replay's ``mo.wait_host_value``
              passes immediately, before any async kickoff has run.
              Direct Python signalling on the hot path is usually a
              mistake; prefer the async-host-func trampoline, which
              signals from its AsyncRT worker.

              Args:
                  value: The 64-bit value to store.
            )nbdoc")
      .def("load", &Python::CompletionFlag::load,
           R"nbdoc(
              Acquire-ordered load of the current flag value.

              Pairs with a release-ordered store on the producer side.

              Returns:
                  The current 64-bit flag value.
            )nbdoc")
      .def_prop_ro(
          "_unsafe_ptr",
          [](const Python::CompletionFlag &self) {
            return reinterpret_cast<std::uintptr_t>(self.ref().getPointer());
          },
          R"nbdoc(
              Raw 64-bit address of the underlying ``M::Driver::CompletionFlag``.

              Intended for packing into a graph-op payload buffer
              (such as for ``mo.wait_host_value``); parallels the
              trampoline/user_data pointers returned by
              ``__unsafe_pack_py_host_func`` for ``mo.launch_host_func``.

              The caller must keep this ``CompletionFlag`` Python object
              alive for the duration of any graph execution that
              references the pointer; the underlying allocation is
              freed when the last ref to the wrapper is dropped. The
              leading underscore marks this as an escape hatch with no
              safety net.
            )nbdoc");

  nb::class_<Python::DeviceEvent>(m, "DeviceEvent", R"nbdoc(
          Provides access to an event object.

          An event can be used to wait for the GPU execution to reach a certain
          point on the given queue.

          For example, create an event, record it on the default queue, and
          wait for the queue to reach it:

          .. code-block:: python

              from max import driver

              device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
              event = driver.DeviceEvent(device)
              device.default_queue.record_event(event)
              event.synchronize()

          Args:
              device: The device on which to create the event.
              enable_timing: If ``True``, enables GPU timing on this event.
                  Events created with ``enable_timing=True`` can be used with
                  :meth:`elapsed_time` to measure GPU execution time.
                  Defaults to ``False``.

          Raises:
              ValueError: If event creation failed.
  )nbdoc")
      .def(nb::init<Python::Device &, bool>(), "device"_a,
           "enable_timing"_a = false)
      .def("synchronize", &Python::DeviceEvent::synchronize, R"nbdoc(
          Ensures all operations on this queue complete before returning.

          Raises:
              ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def("is_ready", &Python::DeviceEvent::isReady, R"nbdoc(
        Returns whether this event is ready.

        Returns:
            ``True`` if the event is complete, ``False`` otherwise.

        Raises:
            ValueError: If querying the event status returned an error.
      )nbdoc")
      .def("elapsed_time", &Python::DeviceEvent::elapsedTime, "end_event"_a,
           R"nbdoc(
        Returns the elapsed GPU time in milliseconds between this event and ``end_event``.

        Both events must have been created with ``enable_timing=True``
        and recorded on a queue before calling this method. The end
        event must be synchronized before calling this method.

        For example, time the GPU work enqueued between two recorded events:

        .. invisible-code-block: python

            from max import driver

        .. skip: next if(driver.accelerator_count() == 0 or driver.Accelerator().api not in ("cuda", "hip"), reason="requires a CUDA or HIP device")

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            start = driver.DeviceEvent(device, enable_timing=True)
            end = driver.DeviceEvent(device, enable_timing=True)

            queue = device.default_queue
            queue.record_event(start)
            # ... GPU work ...
            queue.record_event(end)
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)

        Args:
            end_event: The ending event.

        Returns:
            The elapsed time in milliseconds.

        Raises:
            RuntimeError: If either event was not created with timing
                enabled, or if the events have not been recorded.
      )nbdoc")
      .def("__str__", &Python::DeviceEvent::getAsString)
      .def("__repr__", &Python::DeviceEvent::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"));
  auto launchTraceEntry = nb::class_<M::Driver::KernelLaunchMetadata>(
      m, "LaunchTraceEntry", R"nbdoc(
          One operation recorded by ``max.driver.begin_launch_trace``.

          Describes a kernel launch, memory copy, or memset enqueued on a
          device stream. Only the field group matching ``kind`` is
          meaningful; the other groups hold zero values. ``stream_index``
          identifies which stream enqueued the operation, so entries from
          different streams can be told apart in the single enqueue-ordered
          list. ``semantic_hash`` is a deterministic hash of the launch
          parameters that excludes memory addresses, so it is stable across
          runs and suitable for change-detection in tests.
  )nbdoc");
  nb::enum_<M::Driver::KernelLaunchMetadata::OperationKind>(
      launchTraceEntry, "OperationKind",
      "The kind of operation a trace entry describes.")
      .value("KERNEL_LAUNCH",
             M::Driver::KernelLaunchMetadata::OperationKind::KernelLaunch,
             "A compute kernel launched on the device.")
      .value("MEMCPY", M::Driver::KernelLaunchMetadata::OperationKind::Memcpy,
             "A memory copy between device or host locations.")
      .value("MEMSET", M::Driver::KernelLaunchMetadata::OperationKind::Memset,
             "A region of memory filled with a value.");
  nb::enum_<M::Driver::KernelLaunchMetadata::MemcpyKind>(
      launchTraceEntry, "MemcpyKind", "The direction of a memcpy entry.")
      .value("NONE", M::Driver::KernelLaunchMetadata::MemcpyKind::None,
             "Not a memcpy; the entry is a kernel launch or memset.")
      .value("HTOD", M::Driver::KernelLaunchMetadata::MemcpyKind::HtoD,
             "Host-to-device copy.")
      .value("DTOH", M::Driver::KernelLaunchMetadata::MemcpyKind::DtoH,
             "Device-to-host copy.")
      .value("DTOD", M::Driver::KernelLaunchMetadata::MemcpyKind::DtoD,
             "Device-to-device copy.");
  launchTraceEntry
      .def_ro("kind", &M::Driver::KernelLaunchMetadata::kind,
              "The kind of operation this entry represents.")
      .def_ro("name", &M::Driver::KernelLaunchMetadata::kernelName,
              "The kernel name, or the driver API name for copies/memsets"
              " (such as ``cuMemcpyHtoD``).")
      .def_ro("semantic_hash", &M::Driver::KernelLaunchMetadata::semanticHash,
              "Deterministic, address-free hash of the operation parameters.")
      .def_ro("grid_x", &M::Driver::KernelLaunchMetadata::gridX,
              "X dimension of the launch grid, in blocks.")
      .def_ro("grid_y", &M::Driver::KernelLaunchMetadata::gridY,
              "Y dimension of the launch grid, in blocks.")
      .def_ro("grid_z", &M::Driver::KernelLaunchMetadata::gridZ,
              "Z dimension of the launch grid, in blocks.")
      .def_ro("block_x", &M::Driver::KernelLaunchMetadata::blockX,
              "X dimension of each thread block.")
      .def_ro("block_y", &M::Driver::KernelLaunchMetadata::blockY,
              "Y dimension of each thread block.")
      .def_ro("block_z", &M::Driver::KernelLaunchMetadata::blockZ,
              "Z dimension of each thread block.")
      .def_ro("shared_mem_bytes",
              &M::Driver::KernelLaunchMetadata::sharedMemBytes,
              "Bytes of dynamic shared memory requested by the launch.")
      .def_ro("stream_index", &M::Driver::KernelLaunchMetadata::streamIndex,
              "Identifies the stream this operation was enqueued on, assigned"
              " in first-seen order within a trace.")
      .def_ro("memcpy_kind", &M::Driver::KernelLaunchMetadata::memcpyKind,
              "The copy direction; ``NONE`` unless ``kind`` is ``MEMCPY``.")
      .def_ro("memcpy_byte_size",
              &M::Driver::KernelLaunchMetadata::memcpyByteSize,
              "Bytes copied; zero unless ``kind`` is ``MEMCPY``.")
      .def_ro("memset_byte_size",
              &M::Driver::KernelLaunchMetadata::memsetByteSize,
              "Bytes written; zero unless ``kind`` is ``MEMSET``.")
      .def_ro("memset_value", &M::Driver::KernelLaunchMetadata::memsetValue,
              "The fill value; zero unless ``kind`` is ``MEMSET``.")
      .def_ro("memset_value_size",
              &M::Driver::KernelLaunchMetadata::memsetValueSize,
              "Element size of the fill value in bytes; zero unless ``kind``"
              " is ``MEMSET``.")
      .def("__repr__", [](const M::Driver::KernelLaunchMetadata &self) {
        const char *kind = "KERNEL_LAUNCH";
        if (self.kind == M::Driver::KernelLaunchMetadata::OperationKind::Memcpy)
          kind = "MEMCPY";
        else if (self.kind ==
                 M::Driver::KernelLaunchMetadata::OperationKind::Memset)
          kind = "MEMSET";
        return std::string("LaunchTraceEntry(kind=") + kind +
               ", name=" + self.kernelName +
               ", stream_index=" + std::to_string(self.streamIndex) + ")";
      });
  nb::class_<Python::DeviceStream>(m, "DeviceQueue", R"nbdoc(
          Provides access to a queue of execution on a device.

          A queue represents a sequence of operations that will be executed in order.
          Multiple queues on the same device can execute concurrently.

          .. code-block:: python

              from max import driver
              device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
              queue = device.default_queue
              new_queue = driver.DeviceQueue(device)
  )nbdoc")
      .def(nb::new_(&Python::DeviceStream::createStream), "device"_a, R"nbdoc(
          Creates a new queue of execution associated with the device.

          Args:
              device: The device to create the queue on.
      )nbdoc")
      .def("synchronize", &Python::DeviceStream::synchronize, R"nbdoc(
          Ensures all operations on this queue complete before returning.

          Raises:
              ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def("record_event",
           static_cast<Python::PyDeviceEventTy (Python::DeviceStream::*)()>(
               &Python::DeviceStream::enqueueEvent),
           R"nbdoc(
          Records an event on this queue.

          Returns:
              A new event that will be signaled when all operations
              submitted to this queue before this call have completed.

          Raises:
              ValueError: If recording the event failed.
      )nbdoc")
      .def("record_event",
           static_cast<void (Python::DeviceStream::*)(
               const Python::DeviceEvent &) const>(
               &Python::DeviceStream::enqueueEvent),
           "event"_a,
           R"nbdoc(
          Records an existing event on this queue.

          Args:
              event: The event to record on this queue.

          Raises:
              ValueError: If recording the event failed.
      )nbdoc")
      .def("wait_for", &Python::DeviceStream::waitFor, "stream"_a, R"nbdoc(
          Ensures all operations on the other queue complete before future work submitted to this queue is scheduled.

          Args:
              stream: The queue to wait for.
      )nbdoc")
      .def("wait_for", &Python::DeviceStream::waitForDevice, "device"_a,
           R"nbdoc(
          Ensures all operations on the device's default queue complete before future work submitted to this queue is scheduled.

          Args:
              device: The device whose default queue to wait for.
      )nbdoc")
      .def("wait_for_host_value", &Python::DeviceStream::waitForHostValue,
           "flag"_a, "value"_a,
           R"nbdoc(
          Stalls the queue until ``flag``'s 64-bit value equals ``value``.

          Wraps the MLRT ``DeviceStream::enqueueWaitOnHostValue`` primitive
          (CUDA's ``cuStreamWaitValue64``). Typically paired with
          ``Device.__unsafe_enqueue_async_py_host_func`` to gate
          downstream GPU work on a host-side AsyncRT task that signals
          ``flag`` when it finishes -- a queue-internal sync that
          avoids a host ``synchronize()`` and captures cleanly into a
          CUDA graph as a wait-value node.

          Args:
              flag: The completion flag to wait on. The queue observes
                  ``flag.device_ptr`` via the pinned device-mapped alias.
              value: The 64-bit value to wait for (equality).

          Raises:
              RuntimeError: If the underlying device does not support
                  stream memory ops, or if the driver rejects the
                  enqueue.
      )nbdoc")
      .def(
          "__unsafe_enqueue_async_py_host_func",
          [](Python::DeviceStream &self, nb::callable &&fn,
             const Python::CompletionFlag &flag, uint64_t value,
             const Python::CPU &cpu) {
            self.enqueueAsyncHostFunc(std::move(fn), flag, value, cpu);
          },
          "fn"_a, "flag"_a, "value"_a, "cpu"_a,
          R"nbdoc(
          Queue-targeted variant of ``Device.__unsafe_enqueue_async_py_host_func``.

          Enqueues a kickoff host node on **this** queue that dispatches ``fn``
          onto ``cpu``'s AsyncRT worker pool and returns immediately. When ``fn``
          finishes, the worker atomic-stores ``value`` (release ordering) to the
          64-bit memory at ``flag``. Pair with
          ``DeviceQueue.wait_for_host_value(flag, value)`` on a consumer queue
          to gate downstream GPU work.

          Use this overload when you need the host callback to run on a side
          queue concurrently with the model queue's forward pass; the
          ``Device`` overload always targets the default queue and therefore
          serializes against any other default-queue work. As of this
          writing no production caller dispatches via this queue
          overload -- ``StructuredOutputOverlapState.enqueue_async_callback``
          intentionally lands on the device default queue so the
          trampoline's ``flag.reset()`` is naturally ordered against the
          next iter's captured-graph wait. The queue variant is exposed
          as future-facing API and exercised by the GPU integration test
          (``test_structured_output_overlap_gpu.py``).

          Args:
              fn (Callable[[], None]): A zero-argument callable.
              flag (CompletionFlag): The completion flag the worker will
                  signal when ``fn`` returns.
              value (int): The 64-bit value to store on completion.
              cpu (CPU): The CPU device whose AsyncRT worker pool will
                  execute ``fn``.

          Raises:
              RuntimeError: If the underlying device does not support host
                  callbacks, if the supplied ``cpu`` has no associated AsyncRT
                  CPUDevice, or if the driver rejects the enqueue.
      )nbdoc")
      .def_prop_ro("device", &Python::DeviceStream::getDevice,
                   "The device this queue is executing on.")
      .def(
          "_device_context_ptr",
          [](Python::DeviceStream &self) {
            auto *ctx = self.stream->ctx.getPointer();
#if MODULAR_HAL_DEVICE_CONTEXT
            // See Device._device_context_ptr: hand out the boxed Mojo handle.
            return reinterpret_cast<uintptr_t>(ctx->mojoHandle());
#else
            return reinterpret_cast<uintptr_t>(ctx);
#endif
          },
          "Gets the AsyncRT DeviceContext pointer for this specific queue.")
      .def_prop_ro("native_stream_handle",
                   &Python::DeviceStream::nativeStreamHandle, R"nbdoc(
        The native stream handle as an integer, or ``0`` if there is none.

        The handle is the CUDA ``CUstream`` / HIP ``hipStream_t``; ``0`` means
        the stream has no native handle (such as a CPU device). Native code
        outside MLRT can use the handle to order its own work against this
        stream, for example by recording a CUDA event on it. The handle
        remains owned by this stream; do not destroy it.

        Returns:
            The native stream handle, or ``0`` if there is none.
      )nbdoc")
      .def("__str__", &Python::DeviceStream::getAsString)
      .def("__repr__", &Python::DeviceStream::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"));

  m.def("accelerator_count", &Python::acceleratorCount,
        "Returns the number of accelerator devices available.");

  m.def(
      "begin_launch_trace",
      []() {
        nb::gil_scoped_release release;
        M::Driver::beginGlobalLaunchTrace();
      },
      R"nbdoc(
      Starts a process-global recording of enqueued device operations.

      Records kernel launches, memory copies, and memsets across **all**
      streams into one enqueue-ordered list, clearing any previous trace.
      No stream or device handle is needed, so work enqueued on streams the
      caller does not hold (such as a compiled graph's internal stream) is still
      captured. Only CUDA and HIP devices record entries; on other devices the
      trace is always empty. Intended for tests and debugging: pair with
      ``take_launch_trace`` to assert which device work a code path enqueues
      and on which stream.
  )nbdoc");

  m.def(
      "take_launch_trace",
      []() {
        std::vector<M::Driver::KernelLaunchMetadata> trace;
        {
          nb::gil_scoped_release release;
          trace = M::Driver::takeGlobalLaunchTrace();
        }
        return trace;
      },
      R"nbdoc(
      Stops the global recording and returns the recorded entries.

      Returns:
          list[LaunchTraceEntry]: The operations enqueued since
          ``begin_launch_trace``, in enqueue order across all streams.
          Each entry's ``stream_index`` identifies its stream.
  )nbdoc");

  m.def("__unsafe_pack_py_host_func", &Python::packHostFunc, "fn"_a,
        R"nbdoc(
      Packs a Python callable into a `(trampoline_ptr, user_data_ptr)` pair.

      The returned integers are suitable for passing to
      ``MLRT::DeviceStream::enqueueHostFunc`` (and, by extension, to the
      ``mo.launch_host_func`` custom op). The trampoline is a C ABI
      ``void(void*)`` function; ``user_data_ptr`` owns a heap allocation
      holding the Python callable.

      Ownership transfers to the caller: the user-data pointer MUST be
      consumed by exactly one ``enqueue_host_func`` invocation, which
      causes the trampoline to free it after calling the callable.
      Discarding the pair without enqueueing leaks the callable.

      Args:
          fn (Callable[[], None]): A zero-argument callable.

      Returns:
          tuple[int, int]: ``(trampoline_ptr, user_data_ptr)`` as integers.
      )nbdoc");

  m.def("set_virtual_device_count", &M::AsyncRT::setVirtualDeviceCount,
        "count"_a,
        R"nbdoc(
        Sets the number of virtual devices for device creation.

        When ``count`` is greater than 0, device creation returns virtual
        devices for GPU APIs, and device enumeration reports this count. This
        allows creating devices for GPU configurations beyond the current
        hardware.

        Args:
            count: The number of virtual devices. Set to 0 to disable
                virtual device mode.
    )nbdoc");

  m.def("get_virtual_device_count", &M::AsyncRT::getVirtualDeviceCount, R"nbdoc(
        Gets the current virtual device count.

        Returns:
            The number of virtual devices, or 0 if virtual device mode is
            disabled.
    )nbdoc");

  m.def("is_virtual_device_mode", &M::AsyncRT::isVirtualDeviceMode, R"nbdoc(
        Checks if virtual device mode is currently enabled.

        Returns:
            ``True`` if virtual device mode is enabled (``count > 0``),
            ``False`` otherwise.
    )nbdoc");

  m.def(
      "enable_all_peer_access",
      []() {
#if MODULAR_HAL_DEVICE_CONTEXT
        const char *err = MojoHAL_DeviceContext_enableAllPeerAccess();
        if (err) {
          std::string message(err);
          MojoHAL_strfree(err);
          throw std::runtime_error(message);
        }
#else
        const char *err = AsyncRT_DeviceContext_enableAllPeerAccess();
        if (err)
          throw std::runtime_error(err);
#endif
      },
      R"nbdoc(
        Enables peer-to-peer memory access between all available GPU pairs.

        This must be called before any collective operations (allreduce,
        broadcast, and similar) that require direct GPU-to-GPU memory access.
        It is safe to call multiple times; the underlying runtime caches
        the result after the first successful enablement.

        Raises:
            RuntimeError: If P2P access cannot be enabled between any GPU pair.
    )nbdoc");

  m.def("set_virtual_device_api", &M::AsyncRT::setVirtualDeviceAPI, "api"_a,
        R"nbdoc(
        Sets the target API for virtual devices in compile-only mode.

        This specifies which GPU API virtual devices use for compilation,
        such as ``"cuda"``, ``"hip"``, or ``"metal"``. Must be called
        before creating virtual devices via
        :func:`~max.driver.set_virtual_device_count`.

        Args:
            api: The target API string, such as ``"cuda"`` for NVIDIA,
                ``"hip"`` for AMD, or ``"metal"`` for Apple.
    )nbdoc");

  m.def("get_virtual_device_api", &M::AsyncRT::getVirtualDeviceAPI,
        R"nbdoc(
        Gets the current target API for virtual devices.

        Returns:
            The target API string, or an empty string if not set.
    )nbdoc");

  m.def("set_virtual_device_target_arch",
        &M::AsyncRT::setVirtualDeviceTargetArch, "arch"_a,
        R"nbdoc(
        Sets the target GPU architecture for virtual devices in compile-only mode.

        This specifies the GPU architecture that virtual devices target
        when compiling code, such as ``"sm_80"`` or ``"sm_90"``. Must be
        called before creating virtual devices via
        :func:`~max.driver.set_virtual_device_count`.

        Args:
            arch: The target GPU architecture string, such as ``"sm_80"``
                for Ampere/A100 or ``"sm_90"`` for Hopper/H100.
    )nbdoc");

  m.def("get_virtual_device_target_arch",
        &M::AsyncRT::getVirtualDeviceTargetArch,
        R"nbdoc(
        Gets the current target GPU architecture for virtual devices.

        Returns:
            The target GPU architecture string, or an empty string if not
            set.
    )nbdoc");

  m.def("set_virtual_cpu_target", &M::Driver::setVirtualCpuTarget, "cpu"_a,
        R"nbdoc(
        Sets the CPU target for host-independent kernel codegen.

        Set this before any CPU kernel is compiled. CPU kernels compiled
        afterwards target this CPU, so the kernel cache can be shipped to,
        and reused on, a host with a different CPU. Mirrors
        :func:`set_virtual_device_target_arch` for GPUs.

        Args:
            cpu: An LLVM target-CPU name, such as ``x86-64-v3`` or
                ``neoverse-n1``; ``generic``, for the most portable baseline
                of the host architecture family (``x86-64`` on x86_64, the
                armv8-a baseline on AArch64, an error for other families);
                or an empty string, which restores host-CPU codegen.
                ``native`` is rejected because it would compile for the
                host's CPU.
    )nbdoc");

  m.def("get_virtual_cpu_target", &M::Driver::getVirtualCpuTarget,
        R"nbdoc(
        Gets the current virtual CPU target.

        Returns:
            The CPU target string, or an empty string if not set (host CPU).
    )nbdoc");

  nb::enum_<Python::Usage>(m, "Usage", nb::is_flag(), R"nbdoc(
          Allocation-intent descriptor for :obj:`Buffer`.

          Flags compose with ``|`` and are tested with ``in``. ``max.driver``
          owns the flag set and its per-backend mapping.
      )nbdoc")
      .value("DEFAULT", Python::Usage::Default,
             "The allocation Buffer performs today: device memory for a "
             "non-host device, ordinary host memory for the CPU.")
      .value("STAGING", Python::Usage::Staging,
             "Host memory for staging transfers to and from the given "
             "device. May be page-locked, depending on the backend.")
      .value("UNTRACKED", Python::Usage::Untracked,
             "Memory opting out of host side hazard tracking. Useless unless "
             "paired with STAGING.");

  nb::exception<Python::HostHazardError>(m, "HostHazardError",
                                         PyExc_RuntimeError);

  nb::class_<Python::HostHazardCompletion>(m, "HostHazardCompletion", R"nbdoc(
          A completion shared by host hazard tracked resources.

          Internal to max.driver's hazard tracking. Facilitates ordering of host
          access to resources.

          .. code-block:: python

              import numpy as np

              from max import driver
              from max.dtype import DType

              device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
              staging = driver.Buffer(
                  DType.float32, [4], device=device, usage=driver.Usage.STAGING
              )
              host = np.zeros(4, dtype=np.float32)
              source = driver.Buffer.from_numpy(host).to(device)

              # Record a token on the queue that will signal the copy, then
              # stamp it onto every buffer that submission touched.
              staging.inplace_copy_from(source)

              # The host read waits on that token without draining the device.
              value = staging[0].item()
    )nbdoc")
      .def_static("record_on", &Python::HostHazardCompletion::recordOn,
                  "queue"_a, nb::call_guard<nb::gil_scoped_release>(),
                  R"nbdoc(
          Records an event on ``queue`` for the work already submitted to it.

          Args:
              queue (DeviceQueue): The queue whose submitted work to record.

          Returns:
              HostHazardCompletion: A completion that resolves when that work
              finishes.
      )nbdoc")
      .def_static("poisoned", &Python::HostHazardCompletion::poisoned,
                  "queue"_a, "error"_a, R"nbdoc(
          Creates a completion that raises instead of waiting.

          For a producer that failed after enqueuing work, so a later host
          access reports the failure rather than trusting unfinished bytes.

          Args:
              queue (DeviceQueue): The queue the failed work was submitted to.
              error (str): The producer's rendered error.

          Returns:
              HostHazardCompletion: A poisoned completion.
      )nbdoc")
      .def("is_ready", &Python::HostHazardCompletion::isReady, R"nbdoc(
          Returns whether the recorded work has finished.

          A poisoned completion reports ready: waiting on it raises rather
          than blocking, so it can never hold up a caller.

          Returns:
              bool: Whether the work has finished.
      )nbdoc")
      .def("wait", &Python::HostHazardCompletion::wait, R"nbdoc(
          Blocks until the recorded work finishes.

          Raises:
              HostHazardError: If the completion is poisoned, carrying the
                  producer's error.
      )nbdoc")
      .def("poison", &Python::HostHazardCompletion::poison, "message"_a,
           R"nbdoc(
          Marks the completion failed, so every later wait raises.

          Args:
              message (str): The producer's error, reported by the raise.
      )nbdoc");

  nb::class_<Python::Buffer>(m, "Buffer", R"nbdoc(
          A multi-dimensional buffer allocated on a device.

          Allocates memory onto a given device with the provided shape and dtype.
          Buffers can be sliced to provide strided views of the underlying memory,
          but any buffers input into model execution must be contiguous.

          Supports numpy-style slicing but does not currently support setting
          items across multiple indices.

          .. code-block:: python

              from max import driver
              from max.dtype import DType

              cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32)

          .. code-block:: python

              from max import driver
              from max.dtype import DType

              device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
              device_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32, device=device)

          Args:
              dtype: The data type of buffer elements.
              shape: The buffer shape as a sequence of positive, non-zero
                  integers.
              device: The device to allocate the buffer on. Defaults to the
                  CPU.
              stream: The queue to associate the buffer with.
              usage: The allocation intent, see :obj:`Usage`. Defaults to
                  ``Usage.DEFAULT``.
      )nbdoc")
      // Constructors
      .def(nb::init<M::DType, const std::vector<size_t> &,
                    std::optional<Python::PyDeviceTy>, Python::Usage>(),
           "dtype"_a, "shape"_a, "device"_a = nb::none(),
           "usage"_a = Python::Usage::Default)
      .def(nb::init<M::DType, const std::vector<size_t> &,
                    Python::PyDeviceStreamTy, Python::Usage>(),
           "dtype"_a, "shape"_a, "stream"_a, "usage"_a = Python::Usage::Default)
      .def(nb::init<nb::ndarray<nb::ro>, Python::PyDeviceTy>(), "shape"_a,
           "device"_a)
      // Properties
      .def_prop_ro("device", &Python::Buffer::getDevice,
                   "The device on which the buffer is resident.")
      .def_prop_ro("stream", &Python::Buffer::getStream,
                   "The stream to which the buffer is bound.")
      .def_prop_ro("dtype", &Python::Buffer::getDType,
                   "The data type of the buffer elements.")
      .def_prop_ro(
          "element_size",
          [](Python::Buffer &self) { return self.getDType().getSizeInBytes(); },
          "The size of the element type in bytes.")
      .def_prop_ro("is_contiguous", &Python::Buffer::isContiguous, R"nbdoc(
            Whether the buffer is contiguously allocated in memory.

            Returns ``False`` if the buffer is a non-contiguous slice.

            Currently, some situations that are contiguous are considered
            non-contiguous for the purposes of the engine, such as when a
            buffer has negative steps.
      )nbdoc")
      .def_prop_ro("strides", &Python::Buffer::getStrides, R"nbdoc(
            How far apart consecutive elements are, per dimension.

            Counted in elements, not bytes.

            Slicing does not move any data, so a slice keeps its parent's
            strides. Narrow a 10-wide buffer to 6 columns and it still takes
            10 elements to reach the next row:

            .. code-block:: python

                parent = driver.Buffer(shape=[8, 10], dtype=DType.uint8)
                rows = parent[:, :6]
                rows.shape          # (8, 6)
                rows.strides        # (10, 1) -- still 10 to the next row
                rows.is_contiguous  # False

            A buffer must be contiguous to be passed into model execution.
          )nbdoc")
      .def_prop_ro("is_host", &Python::Buffer::isHost, R"nbdoc(
            Whether the buffer is host-resident.

            Returns ``False`` for GPU buffers and ``True`` for CPU buffers.

            .. code-block:: python

                from max import driver
                from max.dtype import DType

                cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())

                print(cpu_buffer.is_host)
      )nbdoc")
      .def_prop_ro("num_elements", &Python::Buffer::getNumElements, R"nbdoc(
            Returns the number of elements in this buffer.

            Rank-0 buffers have 1 element by convention.
      )nbdoc")
      .def_prop_ro("rank", &Python::Buffer::getRank, "The rank of the buffer.")
      .def_prop_ro("shape", &Python::Buffer::getShape,
                   "The shape of the buffer.")
      // Methods (including statics)
      .def(
          "contiguous", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def contiguous(self) -> Buffer"),
          "Creates a contiguous copy of the buffer.")
      .def(
          "copy",
          [](Python::Buffer &self,
             std::optional<Python::PyDeviceTy> maybeDevice) {
            return self.copyTo(maybeDevice ? std::move(maybeDevice.value())
                                           : self.getDevice());
          },
          "device"_a = nb::none(),
          R"nbdoc(
        Creates a deep copy on an optionally given device.

        If ``device`` is ``None`` (the default), a copy is created on the
        same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            cpu_copy = cpu_buffer.copy()

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            device = driver.Accelerator() if driver.accelerator_count() else driver.CPU()
            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            device_copy = cpu_buffer.copy(device=device)

        Args:
            device: The device to create the copy on. Defaults to ``None``
                (same device).

        Returns:
            A new buffer that is a copy of this buffer.
      )nbdoc")
      .def_static(
          "mmap", []() { /* Replaced in Python */ },
          nb::sig("def mmap("
                  "filename: os.PathLike, "
                  "dtype: max._core.dtype.DType, "
                  "shape: Sequence[int], "
                  "mode: numpy._MemMapModeKind = \"copyonwrite\", "
                  "offset: int = 0) -> Buffer"),
          R"nbdoc(
          Creates a memory-mapped buffer from a binary file on disk.

          The constructor argument semantics follow those of
          ``numpy.memmap``.
      )nbdoc")
      .def(
          "inplace_copy_from",
          [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def inplace_copy_from(self, src: Buffer) -> None"),
          R"nbdoc(
          Copies the contents of another buffer into this one.

          These buffers may be on different devices.
          Requires that both buffers are contiguous and have the same size.
      )nbdoc")
      .def_static(
          "from_dlpack", []() { /* Replaced in Python */ },
          nb::sig("def from_dlpack(array: Any, *, copy: bool | None = None) -> "
                  "Buffer"),
          R"nbdoc(
        Creates a buffer from an object implementing the DLPack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.

        Args:
            array: An object that implements the DLPack protocol.
            copy: Whether to create a copy of the data. Defaults to ``None``.

        Returns:
            A new buffer that views or copies the DLPack data.
      )nbdoc")
      .def_static(
          "from_numpy", []() { /* Replaced in Python */ },
          nb::sig("def from_numpy(arr: numpy.ndarray) -> Buffer"),
          R"nbdoc(
        Creates a buffer from a NumPy array on the host device.

        The underlying data is not copied unless the array is noncontiguous.
        If it is, a contiguous copy is returned.

        Args:
            arr: The NumPy array to convert.

        Returns:
            A new buffer that views or copies the NumPy array data.
      )nbdoc")
      .def("item", &Python::Buffer::item, R"nbdoc(
        Returns the scalar value at a given location.

        Currently implemented only for zero-rank buffers. The return type is
        converted to a Python built-in type.
      )nbdoc")
      .def_static(
          "scalar",
          [](nb::any value, M::DType dtype,
             std::optional<Python::PyDeviceTy> device) {
            auto cpu = Python::getCpuDevice(0);
            Python::Buffer t{dtype, {}, cpu};
            t.set(nb::cast(0), value);
            if (device && device.value() != cpu) {
              // We can't directly set GPU memory, so
              // we just have to copy the tensor over.
              return t.copyTo(std::move(device.value()));
            } else {
              return t;
            }
          },
          "value"_a, "dtype"_a, "device"_a = nb::none(), R"nbdoc(
        Creates a scalar value of a given dtype and value.

        Args:
            value: The scalar value.
            dtype: The data type of the scalar.
            device: The device to allocate the buffer on. Defaults to
                ``None``, which allocates the buffer on the CPU.

        Returns:
            A new rank-0 buffer holding the scalar value.
      )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> self,
             Python::PyDeviceTy device,
             std::optional<Python::PyDeviceEventTy> after) {
            return tensorTo(self, self.p->getStream()->stream, device,
                            eventOrNull(after));
          },
          "device"_a, nb::kw_only(), "after"_a = nb::none(),
          R"nbdoc(
        Returns a buffer that's guaranteed to be on the given device.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident. A buffer on another
        queue of the same device becomes a view on the default queue, ordered
        after ``after``.

        Args:
            device (Device): The device to move the buffer to.
            after (DeviceEvent, optional): Recorded on the buffer's current
                queue. Required when the result is a view.

        Raises:
            ValueError: If the result is a view and ``after`` is not given.
      )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> self,
             std::vector<Python::PyDeviceTy> devices,
             std::optional<Python::PyDeviceEventTy> after) {
            std::vector<nb::typed<nb::object, Python::Buffer>> tensors;
            for (Python::PyDeviceTy &device : devices) {
              tensors.emplace_back(tensorTo(self, self.p->getStream()->stream,
                                            device, eventOrNull(after)));
            }
            return tensors;
          },
          "devices"_a, nb::kw_only(), "after"_a = nb::none(),
          R"nbdoc(
            Returns a list of buffers that are guaranteed to be on the given devices.

            The buffers are only copied if the requested devices are different from the
            device upon which the buffer is already resident.
          )nbdoc")
      .def(
          "to_numpy", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def to_numpy(self) -> numpy.ndarray"),
          R"nbdoc(
        Converts the buffer to a numpy array.

        If the buffer is on the host (CPU), the NumPy array aliases the
        existing memory. Otherwise, it is copied to the host device.

        Returns:
            A NumPy array containing the buffer data.
      )nbdoc")
      .def_prop_ro(
          "pinned", &Python::Buffer::isPinned,
          "Whether the allocation landed in the device's host memory space. "
          "Ask ``usage`` for what was requested.")
      .def_prop_ro(
          "usage", &Python::Buffer::getUsage,
          "Allocation intent. Slices and views report their parent's usage.")
      .def_prop_ro(
          "_host_hazard_tracked", &Python::Buffer::isHostHazardTracked,
          "Whether the allocation is instrumented for host hazard tracking.")
      .def(
          "view", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def view(self, dtype: max._core.dtype.DType, shape: "
                  "Sequence[int] | None = None) -> Buffer"),
          R"nbdoc(
        Returns a new buffer with the given type and shape that shares the underlying memory.

        If ``shape`` is not given, it is deduced by resizing the last axis to fit
        the new dtype. A rank-0 buffer has no axis to resize, so it keeps shape
        ``()`` and only accepts a dtype of the same size. A ``ValueError`` is raised
        when no shape can be deduced.
      )nbdoc")
      .def("memset", &Python::Buffer::memset, "value"_a, R"nbdoc(
        Fills the buffer's bytes in place with a repeated byte value.

        The fill is enqueued on the buffer's device stream. Contiguous slices
        are supported and only their bytes are changed. The value is a byte
        pattern, not a typed element value; zero clears every byte.

        Args:
            value: The byte value, between 0 and 255 inclusive.

        Raises:
            ValueError: If the buffer is not contiguous, is a sub-byte slice
                that does not start and end on whole bytes, or the value is
                outside the byte range.
      )nbdoc")
      .def_static(
          "zeros",
          [](const std::vector<size_t> &shape, M::DType dtype,
             std::optional<Python::PyDeviceTy> device, Python::Usage usage) {
            Python::Buffer t{dtype, shape, std::move(device), usage};
            t.zeros();
            return t;
          },
          "shape"_a, "dtype"_a, "device"_a = nb::none(),
          "usage"_a = Python::Usage::Default,
          R"nbdoc(
        Allocates a buffer with all elements initialized to zero.

        Args:
            shape: The shape of the buffer.
            dtype: The data type of the buffer.
            device: The device to allocate the buffer on. Defaults to
                ``None`` (CPU).
            usage: The allocation intent, see :obj:`Usage`. Defaults to
                ``Usage.DEFAULT``.

        Returns:
            A new buffer filled with zeros.
      )nbdoc")
      // Dunders
      .def("__dlpack__", &Python::Buffer::dlPack, nb::kw_only(),
           "stream"_a = nb::none(), "kwargs"_a,
           "Implements part of the DLPack contract.")
      .def("__dlpack_device__", &Python::Buffer::dlPackDevice,
           "Implements part of the DLPack contract.")
#define ShapeType "int | slice | Sequence[int | slice]"
      .def("__getitem__", &Python::Buffer::get, "index"_a, R"nbdoc(
          Gets a buffer slice.

          Supports full numpy-style slicing. Invocations
          using only integer-based indexes will return zero-rank buffers.
      )nbdoc",
           nb::sig("def __getitem__(self, idx: " ShapeType ") -> Buffer"))
      .def("__setitem__", &Python::Buffer::set, "index"_a, "value"_a,
           "Sets an item in the buffer.",
           nb::sig("def __setitem__(self, idx: " ShapeType
                   ", value: Any) -> None"))
#undef ShapeType
      // Private methods
      .def(
          "_aligned",
          [](Python::Buffer &self, std::optional<int> maybe_align) {
            const auto alignment =
                maybe_align ? *maybe_align : self.getDType().getAlignment();
            return self.aligned(alignment);
          },
          "alignment"_a = nb::none(),
          "Returns whether the buffer is aligned to the desired alignment.")
      .def_static("_from_dlpack", &Python::Buffer::fromDLPack)
      .def_static("_from_dlpack",
                  [](nb::capsule dlpack, Python::PyDeviceTy device,
                     nb::int_ externalStream) {
                    if (!PyCapsule_IsValid(dlpack.ptr(), "dltensor")) {
                      throw std::runtime_error(
                          "malformed capsule in dlpack implementation");
                    }
                    PyCapsule_SetName(dlpack.ptr(), "used_dltensor");
                    return Python::Buffer::fromDLManagedTensor(
                        DLPack::DLManagedTensorWrapper{
                            static_cast<DLManagedTensor *>(dlpack.data())},
                        device, nb::cast<intptr_t>(externalStream));
                  })
      .def_static("_iterate_indices",
                  [](Python::Buffer &) { /* Replaced in Python*/ })
      .def("_view", &Python::Buffer::view, "dtype"_a, "shape"_a)
      .def("_inplace_copy_from", &Python::Buffer::inplaceCopyFrom, "src"_a)
      .def(
          "_data_ptr",
          [](Python::Buffer &self) {
            return reinterpret_cast<uintptr_t>(self.data());
          },
          "Gets the memory address of the buffer data. Internal use only.")
      .def("_stamp_read", &Python::Buffer::stampRead, "completion"_a, R"nbdoc(
          Add a read completion to the buffer. Internal use only.
      )nbdoc")
      .def("_stamp_write", &Python::Buffer::stampWrite, "completion"_a, R"nbdoc(
          Add a write completion to the buffer. Internal use only.
      )nbdoc");

  m.def(
      "_batch_inplace_copy",
      [](nb::sequence dsts, nb::sequence srcs) {
        std::vector<Python::Buffer *> dstPtrs, srcPtrs;
        for (nb::handle h : dsts)
          dstPtrs.push_back(nb::cast<Python::Buffer *>(h));
        for (nb::handle h : srcs)
          srcPtrs.push_back(nb::cast<Python::Buffer *>(h));
        Python::Buffer::batchInplaceCopy(dstPtrs, srcPtrs);
      },
      "dsts"_a, "srcs"_a,
      nb::sig("def _batch_inplace_copy(dsts: Sequence[Buffer], srcs: "
              "Sequence[Buffer]) -> None"),
      R"nbdoc(
      Batched copy of ``srcs`` into ``dsts``.

      Sources may be host, pinned, same-device or peer memory in any mix. One
      submission only orders the writes on its own stream, so destinations are
      grouped by device and submitted one batch per device. Identical pairs
      (``dst is src``) are skipped. All buffers must have matching sizes.
  )nbdoc");

  nb::class_<Python::ChunkedStagingRegion>(m, "_ChunkedStagingRegion",
                                           R"nbdoc(
      A host mapping made device-accessible in chunks.

      Page-locking a multi-hundred-GiB range in one call is slow, and above
      1 TiB it fails outright. This maps the range once, faults it in across
      parallel workers, and registers it in chunks while they run ahead.

      Not a :obj:`Buffer`: a ``Buffer`` promises a range the device addresses
      in a single transfer, and a transfer crossing a chunk boundary here
      fails. Callers work from :obj:`address` and keep each transfer inside
      one ``row_bytes`` row, which no chunk boundary splits.

      With ``driver_alloc``, each chunk is instead its own pinned host
      allocation from the GPU driver, which the kernel cannot migrate during
      memory compaction. The chunks are then not contiguous, so callers work
      from :obj:`chunk_addresses`.

      Releases on garbage collection: drains the device, unregisters every
      chunk, then unmaps (or frees every driver-allocated chunk).

      Internal to ``max``; no compatibility guarantee.

      Args:
          byte_size (int): Total bytes to map. Must be a positive multiple of ``row_bytes``.
          row_bytes (int): Row width no chunk boundary may fall inside.
          device (Device): GPU/Accelerator the range is made accessible to. Must not be CPU.
          threads (int, optional): Parallel page-touch workers. Defaults to 16. Unused with ``driver_alloc``.
          chunk_bytes (int, optional): Registration granularity, rounded up to whole rows and pages. Defaults to 512 MiB.
          driver_alloc (bool, optional): Allocate each chunk as pinned host memory from the GPU driver instead of registering one host mapping. Defaults to False.
          mlock_pages (bool, optional): ``mlock`` the registered mapping. Not allowed with ``driver_alloc``. Defaults to False.

      Raises:
          ValueError: If ``device`` is a CPU device, any size is zero, or ``byte_size`` is not a multiple of ``row_bytes``.
          RuntimeError: If registering or allocating a chunk fails.
  )nbdoc")
      .def(nb::init<size_t, size_t, Python::PyDeviceTy, unsigned, size_t, bool,
                    bool>(),
           "byte_size"_a, "row_bytes"_a, "device"_a,
           "threads"_a = Python::kDefaultTouchThreads,
           "chunk_bytes"_a = Python::kDefaultChunkBytes,
           "driver_alloc"_a = false, "mlock_pages"_a = false)
      .def_prop_ro("address", &Python::ChunkedStagingRegion::address,
                   "Base host address, valid until this is released. Raises "
                   "for a ``driver_alloc`` region, which has no single base.")
      .def_prop_ro("chunk_addresses",
                   &Python::ChunkedStagingRegion::chunkAddresses,
                   "Host address of each chunk, in order; chunk ``i`` holds "
                   "region bytes ``[i * chunk_bytes, (i + 1) * "
                   "chunk_bytes)``.")
      .def_prop_ro("driver_alloc", &Python::ChunkedStagingRegion::driverAlloc)
      .def_prop_ro("mlocked", &Python::ChunkedStagingRegion::mlocked)
      .def_prop_ro("byte_size", &Python::ChunkedStagingRegion::byteSize)
      .def_prop_ro("row_bytes", &Python::ChunkedStagingRegion::rowBytes)
      .def_prop_ro("chunk_bytes", &Python::ChunkedStagingRegion::chunkBytes,
                   "The requested ``chunk_bytes`` rounded up to whole rows "
                   "and pages.")
      .def_prop_ro("num_chunks", &Python::ChunkedStagingRegion::numChunks)
      .def_prop_ro("device", &Python::ChunkedStagingRegion::getDevice);

  m.def("_release_buffers_to_borrowed", &Python::Buffer::releaseAllToBorrowed,
        "buffers"_a,
        "Convert owning buffers into borrowed wrappers over the same storage.");

  nb::class_<Python::SymmetricPool>(m, "SymmetricPool", R"nbdoc(
        A driver-owned symmetric multicast pool shared by a set of devices.

        The pool is a single fixed-capacity allocation, created once per device
        set and registered in the driver. It outlives this handle: dropping the
        Python object does not free device memory, and two handles for the same
        devices refer to the same allocation.

        Not constructible directly. Use :meth:`allocate()` once during session
        setup, and :meth:`lookup()` afterwards.
        )nbdoc")
      .def_static("allocate", &Python::SymmetricPool::allocate, "devices"_a,
                  "byte_size"_a, R"nbdoc(
            Allocates the pool over ``devices`` and registers it in the driver.

            Args:
                devices (Sequence[Device]): The devices to span, at least two,
                    in any order.
                byte_size (int): Usable bytes per device. The driver rounds
                    the real allocation up to the multicast granularity.

            Returns:
                SymmetricPool: The newly registered pool.

            Raises:
                RuntimeError: If a pool is already registered for these
                    devices, fewer than two devices are given, a device lacks
                    multicast support, or the allocation fails.
            )nbdoc")
      .def_static("lookup", &Python::SymmetricPool::lookup, "devices"_a,
                  R"nbdoc(
            Returns the pool registered for ``devices``, or ``None``.

            A registry read, never an allocation.

            Args:
                devices (Sequence[Device]): The devices the pool spans.

            Returns:
                SymmetricPool | None: The registered pool, if any.
            )nbdoc")
      .def_static("is_supported", &Python::SymmetricPool::isSupported,
                  "devices"_a,
                  R"nbdoc(
            Returns whether ``devices`` can back a multicast pool.

            Answers before any memory is committed. A device that cannot
            report its capability counts as unsupported.

            Args:
                devices (Sequence[Device]): The devices a pool would span.

            Returns:
                bool: True when every device supports multicast.
            )nbdoc")
      .def_prop_ro("byte_size", &Python::SymmetricPool::getByteSize,
                   "Usable bytes per device; the views are exactly this long.")
      .def("unicast_view", &Python::SymmetricPool::unicastView, "device"_a,
           R"nbdoc(
            Returns the device's own view of the pool as a flat ``uint8`` buffer.

            The buffer is non-owning; the driver owns the memory. Retype a
            sub-region with :meth:`Buffer.view()`.

            Args:
                device (Device): One of the devices the pool spans.

            Returns:
                Buffer: A flat ``uint8`` view, ``byte_size`` bytes long.

            Raises:
                RuntimeError: If ``device`` is not one the pool spans.
            )nbdoc")
      .def("multicast_view", &Python::SymmetricPool::multicastView, "device"_a,
           R"nbdoc(
            Returns the multicast alias the switch reduces through.

            Only meaningful to a kernel issuing multimem instructions; host
            loads and stores through it are undefined.

            Args:
                device (Device): One of the devices the pool spans.

            Returns:
                Buffer: A flat ``uint8`` view, ``byte_size`` bytes long.
            )nbdoc");
  M::Profiler::Python::addTracingToNanobindModuleClassMethods(
      m, "max._core.driver.");
}
