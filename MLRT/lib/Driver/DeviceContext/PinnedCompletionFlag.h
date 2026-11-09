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

#ifndef MLRT_DRIVER_DEVICECONTEXT_PINNEDCOMPLETIONFLAG_H
#define MLRT_DRIVER_DEVICECONTEXT_PINNEDCOMPLETIONFLAG_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

namespace M::Driver {

/// The CUDA and HIP flag: a pinned host slot mapped into the device, which
/// the stream polls.
class PinnedCompletionFlag final : public CompletionFlag {
public:
  PinnedCompletionFlag(DeviceContext &context, void *hostPtr,
                       uint64_t devicePtr);
  ~PinnedCompletionFlag() override;

  void signal(uint64_t value) override;
  uint64_t load() const override;
  void reset() override;

  uint64_t devicePtr() const { return devicePtr_; }

  /// Every flag a CUDA or HIP context allocates is a `PinnedCompletionFlag`.
  static const PinnedCompletionFlag &cast(const CompletionFlag &flag) {
    return static_cast<const PinnedCompletionFlag &>(flag);
  }

private:
  DeviceContextRef context_;
  void *hostPtr_;
  uint64_t devicePtr_;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_PINNEDCOMPLETIONFLAG_H
