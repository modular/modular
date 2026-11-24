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

#ifndef ASYNCRT_NANOBIND_BINDINGS_H
#define ASYNCRT_NANOBIND_BINDINGS_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "nanobind/nanobind.h"

namespace M::AsyncRT::Nanobind {

/// A type-erased AsyncValueRef<T>. AnyAsyncValueRef itself doesn't know
/// its type. The underlying type is stored via a TypeID, but
/// - no TypeID is stored until the AsyncValue is resolved, so
///   unresolved / error futures typecheck as any future
/// - it's not easy to use that TypeID to correctly cast back to a python
/// instance
///
/// PyAsyncValue stores the AnyAsyncValueRef plus a type-erased caster
/// function for T, allowing T to be known at compile time without
/// dynamic TypeID info.
struct PyAsyncValue {
  /// Dispatch receives the Python handle of the owning `AsyncValue` so
  /// it can install a strong-ref keep_alive between the returned typed
  /// wrapper and the AsyncValue (for class payloads aliasing the
  /// AsyncValue's storage). The caster specialization decides whether
  /// keep_alive is needed based on the static `T`.
  using DispatchFn = nanobind::object (*)(AnyAsyncValueRef &ref,
                                          nanobind::handle self);

  /// Allocates an unresolved Python-payload AsyncValue on the current
  /// thread's CPU device; resolve it through `ref`
  /// (`emplace<nb::object>` / `setToError`).
  PyAsyncValue();
  PyAsyncValue(AnyAsyncValueRef ref, DispatchFn dispatch,
               nanobind::object pyException)
      : ref(std::move(ref)), dispatch(dispatch),
        pyException(std::move(pyException)) {}
  PyAsyncValue(PyAsyncValue &&) = default;

  AnyAsyncValueRef ref;
  DispatchFn dispatch;
  /// For Python-set exceptions (via `set_exception`): preserves the
  /// original Python exception object so `exception()` returns it
  /// verbatim. C++-emitted errors (via `setToError`) leave this as
  /// `None` and `exception()` synthesizes a `RuntimeError` from
  /// the diagnostic message.
  nanobind::object pyException;

  /// MLRT requires AsyncValues to reach a terminal state before they
  /// can be destroyed. If an unresolved AsyncValue is dropped (e.g. a
  /// Python-constructed one the user never fulfilled), error it here so
  /// cleanup is well-defined. Any pending `add_done_callback` waiters
  /// fire with that error.
  ~PyAsyncValue();
};

/// Dispatch for Python-payload AsyncValues (those created via the
/// `AsyncValue()` constructor or `AsyncValue.from_future`). The held
/// value is an `nb::object` placed directly into the AsyncValue's
/// storage; we return it without aliasing tricks.
nanobind::object dispatchPyObject(AnyAsyncValueRef &ref, nanobind::handle self);

/// Registers MLRT bindings (the `AsyncValue` class) into \p m. Call
/// once, from the extension module entry point.
void bindMLRT(nanobind::module_ m);

} // namespace M::AsyncRT::Nanobind

#endif // ASYNCRT_NANOBIND_BINDINGS_H
