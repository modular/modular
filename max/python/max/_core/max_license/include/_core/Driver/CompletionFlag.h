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

#ifndef SDK_DRIVERAPI_PYTHON_COMPLETIONFLAG_H
#define SDK_DRIVERAPI_PYTHON_COMPLETIONFLAG_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "_core/Driver/Device.h"
#include "nanobind/nanobind.h"

#include <cstdint>

namespace nb = nanobind;

namespace M::Driver::Python {

/// Python-facing wrapper around `M::Driver::CompletionFlag`. Holds an
/// `RCRef` to the underlying refcounted MLRT object; the host and device
/// pointers are properties for callers (the async-host-func and
/// wait-for-host-value bindings extract them internally).
///
/// Construction allocates a fresh 8-byte device-mapped pinned u64 via
/// `DeviceContext::allocateCompletionFlag`. The allocation is released
/// when the last `CompletionFlagRef` is dropped (typically when the
/// Python object is garbage collected). RAII via C++ destructor is
/// deterministic; we do not rely on Python `__del__`.
struct CompletionFlag {
  /// Allocates a fresh CompletionFlag bound to the given device. The
  /// device must be CUDA-backed; raises `RuntimeError` otherwise.
  explicit CompletionFlag(const Device &device);
  ~CompletionFlag() = default;

  CompletionFlag(const CompletionFlag &) = delete;
  CompletionFlag &operator=(const CompletionFlag &) = delete;

  /// Device-visible address of the 8-byte flag, suitable for
  /// `DeviceStream::enqueueWaitOnHostValue`.
  std::uint64_t devicePtr() const { return ref_->devicePtr(); }

  /// Convenience: relaxed store of 0 to the flag.
  void reset() { ref_->reset(); }

  /// Convenience: release-ordered store of `value`. Pairs with the
  /// GPU-side ``cuStreamWaitValue64`` (or a host-side acquire ``load``)
  /// so that any stores the caller performed before ``signal`` are
  /// visible to whichever side first observes ``value``. Primary use
  /// from Python is priming the flag at setup time so the first
  /// captured-graph replay's ``mo.wait_host_value`` passes immediately,
  /// before any async kickoff has run.
  void signal(std::uint64_t value) { ref_->signal(value); }

  /// Convenience: acquire load of the flag.
  std::uint64_t load() const { return ref_->load(); }

  /// Internal accessor for retyped bindings: lets them grab the
  /// underlying ref to keep the allocation alive for the duration of
  /// the work they enqueue.
  const M::Driver::CompletionFlagRef &ref() const { return ref_; }

private:
  M::Driver::CompletionFlagRef ref_;
};

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_COMPLETIONFLAG_H
