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

#include "_core/AsyncRT/Bindings.h"

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Support/Error.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/typing.h"
#include <future>

namespace nb = nanobind;

namespace {

// Resolve the current MLRT CPU device or throw a Python-visible error.
M::AsyncRT::CompactCPUDevicePtr requireCurrentCPUDevice() {
  auto *cpuDevice = M::AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull();
  if (!cpuDevice)
    throw nb::value_error(
        "AsyncValue requires an MLRT CPUDevice on the current thread");
  return cpuDevice->getCompactPtr();
}

// Build an EncodedDiagnostic from a Python exception object.
M::AsyncRT::EncodedDiagnostic diagnosticFromException(nb::handle exc) {
  // Use Python's `str(exc)` so subclasses with custom `__str__` work
  // correctly. `nb::str(handle)` can hit a bad_cast for some inputs.
  nb::object strFn = nb::module_::import_("builtins").attr("str");
  std::string msg = nb::cast<std::string>(strFn(exc));
  return M::AsyncRT::UnknownLocationDecoder::getDiagnostic(
      M::Error(msg.c_str()));
}

// Raise `asyncio.InvalidStateError` to mirror `asyncio.Future`'s
// behavior on premature `exception()` / `set_result` / etc.
[[noreturn]] void raiseInvalidStateError(const char *msg) {
  nb::object excClass =
      nb::module_::import_("asyncio").attr("InvalidStateError");
  PyErr_SetString(excClass.ptr(), msg);
  throw nb::python_error();
}

// ``PyAsyncValue`` holds an ``nb::object pyException`` that can close a
// reference cycle (av -> exception -> traceback -> frame -> av). Without
// GC support, the cycle never collects and the instance leaks. Provide
// ``tp_traverse`` / ``tp_clear`` slots so Python's cyclic GC can see
// through the C++ struct member and break the cycle.
int pyAsyncValueTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (nb::inst_ready(nb::handle(self))) {
    auto *cppSelf =
        nb::inst_ptr<::M::AsyncRT::Nanobind::PyAsyncValue>(nb::handle(self));
    Py_VISIT(cppSelf->pyException.ptr());
  }
  return 0;
}

int pyAsyncValueClear(PyObject *self) {
  if (nb::inst_ready(nb::handle(self))) {
    auto *cppSelf =
        nb::inst_ptr<::M::AsyncRT::Nanobind::PyAsyncValue>(nb::handle(self));
    cppSelf->pyException = nb::none();
  }
  return 0;
}

PyType_Slot pyAsyncValueSlots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(&pyAsyncValueTraverse)},
    {Py_tp_clear, reinterpret_cast<void *>(&pyAsyncValueClear)},
    {0, nullptr},
};

} // namespace

nb::object M::AsyncRT::Nanobind::dispatchPyObject(AnyAsyncValueRef &ref,
                                                  nb::handle /*self*/) {
  auto *payload =
      static_cast<nb::object *>(ref.getPointer()->getUnderlyingPtr());
  return *payload;
}

::M::AsyncRT::Nanobind::PyAsyncValue::PyAsyncValue()
    : PyAsyncValue(::M::AsyncRT::AsyncValueRef<nb::object>::allocate(
                       requireCurrentCPUDevice()),
                   &::M::AsyncRT::Nanobind::dispatchPyObject, nb::none()) {}

// Resolve a Python-payload AsyncValue to the error state, recording the
// Python exception on the wrapper so `result()`/`exception()` re-raise it
// exactly. The GIL must be held.
void resolveWithException(::M::AsyncRT::Nanobind::PyAsyncValue &av,
                          nb::object exc) {
  av.pyException = exc;
  av.ref.copy().setToError(diagnosticFromException(exc));
}

M::AsyncRT::Nanobind::PyAsyncValue::~PyAsyncValue() {
  if (ref && !ref.isReady()) {
    std::move(ref).setToError(M::AsyncRT::UnknownLocationDecoder::getDiagnostic(
        M::Error("AsyncValue dropped before being resolved")));
  }
}

void M::AsyncRT::Nanobind::bindMLRT(nb::module_ m) {
  m.attr("T") = nb::type_var("T");

  nb::class_<PyAsyncValue>(m, "AsyncValue", nb::is_generic(),
                           nb::sig("class AsyncValue(typing.Generic[T])"),
                           nb::type_slots(pyAsyncValueSlots),
                           R"nbdoc(
        Reference-counted handle to an asynchronous value.

        Maps to `AsyncRT::AsyncValueRef<T>`, but duck-types like
        `asyncio.Future`: `done()` / `result()` / `exception()`,
        `set_result()` / `set_exception()`, `add_done_callback()`,
        and `await` are all supported.

        `AsyncValue` instances handed in from C++ are read-only from
        Python — `set_result` / `set_exception` are rejected. Use
        the no-arg constructor or `from_future` to get a Python-owned
        `AsyncValue[object]` that the caller resolves.

        Example — read a result a C++ API handed back:

            compiled = session.compile(graph)         # AsyncValue[CompiledModels]
            models = await compiled._compiled         # suspends until ready

        Example — fulfill from Python and await:

            av: AsyncValue[object] = AsyncValue()
            asyncio.create_task(produce(av))          # calls av.set_result(...)
            value = await av

        Example — bridge an `asyncio.Future`:

            fut: asyncio.Future[int] = loop.create_future()
            av = AsyncValue.from_future(fut)          # AsyncValue[object]
            ...
            fut.set_result(42)                        # av resolves shortly after

        Example — fire-and-forget callback:

            av.add_done_callback(lambda x: log(x.result()))

        Note:
            Constructing an unresolved `AsyncValue` requires an MLRT
            `CPUDevice` on the current thread (any `InferenceSession`
            initializes one).
      )nbdoc")
      // Default-construct an unresolved Python-payload AsyncValue. Users
      // resolve it via `set_result` / `set_exception`.
      .def(
          "__init__", [](PyAsyncValue *self) { new (self) PyAsyncValue(); },
          "Allocate an unresolved AsyncValue with a Python-object payload.")
      .def(
          "done", [](PyAsyncValue &self) { return self.ref.isReady(); },
          "True if the value (or an error) is available.")
      .def(
          "is_error", [](PyAsyncValue &self) { return self.ref.isError(); },
          "True if the value is in the error state.")
      .def(
          "result",
          [](nb::handle self_handle) -> nb::object {
            auto &self = nb::cast<PyAsyncValue &>(self_handle);
            if (!self.ref.isReady())
              raiseInvalidStateError("AsyncValue result is not yet available");
            if (self.ref.isError()) {
              // Re-raise the Python exception verbatim if one was set via
              // `set_exception`; otherwise synthesize a `RuntimeError`
              // from the C++ diagnostic message.
              if (!self.pyException.is_none()) {
                PyErr_SetObject(reinterpret_cast<PyObject *>(
                                    Py_TYPE(self.pyException.ptr())),
                                self.pyException.ptr());
              } else {
                PyErr_SetString(PyExc_RuntimeError,
                                self.ref.getDiagnostic().getMessage().get());
              }
              throw nb::python_error();
            }
            return self.dispatch(self.ref, self_handle);
          },
          nb::sig("def result(self) -> T"),
          "Return the held value. Raises `asyncio.InvalidStateError` if "
          "not yet done; re-raises the stored exception (or a synthesized "
          "`RuntimeError` for C++-side errors) if errored. Mirrors "
          "`asyncio.Future.result`.")
      .def(
          "wait",
          [](PyAsyncValue &self) {
            nb::gil_scoped_release release;
            M::AsyncRT::await(
                llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>(self.ref));
          },
          "Block the calling thread until this AsyncValue is done.")
      .def(
          "__await__",
          [](nb::handle /*self*/) -> nb::object {
            throw nb::value_error(
                "AsyncValue.__await__ requires `import max.engine` to "
                "install the asyncio integration.");
          },
          nb::sig("def __await__(self) -> Generator[Any, None, T]"),
          "Yield control to the event loop until this AsyncValue resolves.")
      .def(
          "exception",
          [](PyAsyncValue &self) -> nb::object {
            if (!self.ref.isReady())
              raiseInvalidStateError("AsyncValue is not yet done");
            if (!self.ref.isError())
              return nb::none();
            // Return the original Python exception if `set_exception` was
            // used; otherwise synthesize a `RuntimeError` from the
            // diagnostic message.
            if (!self.pyException.is_none())
              return self.pyException;
            return nb::module_::import_("builtins")
                .attr("RuntimeError")(
                    nb::str(self.ref.getDiagnostic().getMessage().get()));
          },
          nb::sig("def exception(self) -> BaseException | None"),
          "Return the exception, or `None` if the AsyncValue completed "
          "successfully. Raises `asyncio.InvalidStateError` if the "
          "AsyncValue is not yet done. Mirrors `asyncio.Future.exception`.")
      .def(
          "set_result",
          [](PyAsyncValue &self, nb::object value) {
            if (self.dispatch != &dispatchPyObject)
              throw nb::value_error(
                  "set_result is only supported on AsyncValues created from "
                  "Python (AsyncValue() or AsyncValue.from_future)");
            if (self.ref.isReady())
              raiseInvalidStateError("AsyncValue is already done");
            // `emplace` consumes the ref it's called on; emplace on a
            // copy so the persistent Python-held ref stays valid.
            self.ref.copy().emplace<nb::object>(std::move(value));
          },
          "Mark this AsyncValue as done with the given value. Raises "
          "`asyncio.InvalidStateError` if already done.")
      .def(
          "set_exception",
          [](PyAsyncValue &self, nb::object exc) {
            if (self.dispatch != &dispatchPyObject)
              throw nb::value_error(
                  "set_exception is only supported on AsyncValues created "
                  "from Python (AsyncValue() or AsyncValue.from_future)");
            if (self.ref.isReady())
              raiseInvalidStateError("AsyncValue is already done");
            resolveWithException(self, exc);
          },
          nb::sig("def set_exception(self, exc: BaseException, /) -> None"),
          "Mark this AsyncValue as errored with the given exception. Raises "
          "`asyncio.InvalidStateError` if already done.")
      .def(
          "and_then",
          [](nb::handle self_handle, nb::object cb) -> nb::object {
            auto &self = nb::cast<PyAsyncValue &>(self_handle);
            // The returned AsyncValue owns the callback's result or
            // exception, like chaining futures.
            nb::object resultObj = nb::cast(PyAsyncValue());
            self.ref.andThenAsync([cb = std::move(cb),
                                   selfObj =
                                       nb::borrow<nb::object>(self_handle),
                                   resultObj]() mutable {
              nb::gil_scoped_acquire gil;
              // Move the Python captures into locals so they are
              // destroyed while the GIL is held; the lambda itself may
              // be destroyed by the work queue off-GIL.
              nb::object localCb = std::move(cb);
              nb::object localSelf = std::move(selfObj);
              nb::object localResult = std::move(resultObj);
              auto &result = nb::cast<PyAsyncValue &>(localResult);
              try {
                result.ref.copy().emplace<nb::object>(localCb(localSelf));
              } catch (nb::python_error &e) {
                resolveWithException(result, nb::borrow<nb::object>(e.value()));
              }
            });
            return resultObj;
          },
          nb::sig("def and_then(self, fn: object, /) -> AsyncValue"),
          "Run fn(self) when this AsyncValue resolves (promptly if "
          "already resolved) and return a new AsyncValue that resolves to "
          "the callback's return value, or to the exception it raises. "
          "The callback is invoked with the GIL held on a runtime worker "
          "thread (not the registering thread), concurrently with other "
          "Python threads.")
      .def(
          "add_done_callback",
          [](nb::handle self_handle, nb::object cb) {
            auto &self = nb::cast<PyAsyncValue &>(self_handle);
            self.ref.andThenAsync(
                [cb = std::move(cb),
                 selfObj = nb::borrow<nb::object>(self_handle)]() mutable {
                  nb::gil_scoped_acquire gil;
                  // Move the Python captures into locals so they are
                  // destroyed while the GIL is held; the lambda itself may
                  // be destroyed by the work queue off-GIL.
                  nb::object localCb = std::move(cb);
                  nb::object localSelf = std::move(selfObj);
                  try {
                    localCb(localSelf);
                  } catch (nb::python_error &e) {
                    e.discard_as_unraisable("AsyncValue.add_done_callback");
                  }
                });
          },
          "Run fn(self) when this AsyncValue resolves, like "
          "`concurrent.futures.Future.add_done_callback`. The callback is "
          "invoked with the GIL held on a runtime worker thread (not the "
          "registering thread), concurrently with other Python threads. "
          "Exceptions it raises are reported via `sys.unraisablehook`.")
      .def_static(
          "from_future",
          [](nb::object fut) -> nb::object {
            nb::object avObj = nb::cast(PyAsyncValue());
            nb::object doneCb = nb::cpp_function([avObj](nb::object fut_done) {
              auto &av = nb::cast<PyAsyncValue &>(avObj);
              nb::object exc = fut_done.attr("exception")();
              if (!exc.is_none())
                return resolveWithException(av, exc);
              auto result = fut_done.attr("result")();
              av.ref.copy().emplace<nb::object>(result);
            });
            fut.attr("add_done_callback")(doneCb);
            return avObj;
          },
          nb::sig("def from_future(fut: object) -> AsyncValue[object]"),
          "Wrap an asyncio.Future in an AsyncValue that resolves when the "
          "Future does.");
}
