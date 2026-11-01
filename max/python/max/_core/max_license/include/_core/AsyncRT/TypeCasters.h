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

#ifndef ASYNCRT_NANOBIND_TYPECASTERS_H
#define ASYNCRT_NANOBIND_TYPECASTERS_H

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "_core/AsyncRT/Bindings.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace NB_NAMESPACE {
namespace detail {

/// Bidirectional type caster between `M::AsyncRT::AsyncValueRef<T>` and
/// the Python `AsyncValue` class (see `_core/AsyncRT/Bindings`).
///
/// The Python wrapper carries a function pointer that knows how to
/// surface the held payload as the Python class for `T`. The dispatch
/// is set here at the cast site where `T` is known at compile time.
template <typename T>
struct type_caster<::M::AsyncRT::AsyncValueRef<T>> {
  using PayloadCaster = make_caster<T>;
  NB_TYPE_CASTER(::M::AsyncRT::AsyncValueRef<T>,
                 const_name("max._core.mlrt.AsyncValue[") +
                     PayloadCaster::Name + const_name("]"))

  static nb::object dispatchPayload(::M::AsyncRT::AnyAsyncValueRef &ref,
                                    nb::handle self) {
    T *payload = static_cast<T *>(ref.getPointer()->getUnderlyingPtr());
    // Use reference_internal to bind the lifetime of the returned value to
    // the AsyncValue's storage.
    return nb::cast(payload, nb::rv_policy::reference_internal, self);
  }

  static handle from_cpp(::M::AsyncRT::AsyncValueRef<T> &&result,
                         rv_policy /*policy*/,
                         cleanup_list * /*cleanup*/) noexcept {
    auto *pyav = new ::M::AsyncRT::Nanobind::PyAsyncValue{
        std::move(result), &dispatchPayload, nb::none()};
    return nb::cast(pyav, nb::rv_policy::take_ownership).release();
  }

  bool from_python(handle src, uint8_t /*flags*/,
                   cleanup_list * /*cleanup*/) noexcept {
    ::M::AsyncRT::Nanobind::PyAsyncValue *pyav = nullptr;
    if (!nb::try_cast<::M::AsyncRT::Nanobind::PyAsyncValue *>(src, pyav))
      return false;

    // Don't accept AsyncValues to other types.
    // As a consequence, AsyncValue[T] is _invariant_, where intuitively
    // and ergonomically it probably wants to be _covariant_.
    if (pyav->dispatch != &dispatchPayload)
      return false;

    // Copy to leave the python instance intact for future use.
    value = ::M::AsyncRT::AsyncValueRef<T>(pyav->ref.copy());
    return true;
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

#endif // ASYNCRT_NANOBIND_TYPECASTERS_H
