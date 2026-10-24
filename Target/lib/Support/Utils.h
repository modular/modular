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
#ifndef TARGET_LIB_UTILS_UTILS_H
#define TARGET_LIB_UTILS_UTILS_H

#include "HAL/M_driver_common.h"
#include "StandardStatus.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>

namespace M::Target {

template <typename Stringish>
void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       Stringish const &srcMsg) {
  assert(dstMsgBufSize > 0);
  auto srcBeginIter = std::begin(srcMsg);
  auto srcEndIter = (int64_t(srcMsg.size()) >= (dstMsgBufSize - 1))
                        ? (srcBeginIter + (dstMsgBufSize - 1))
                        : std::end(srcMsg);
  auto dstEndIter = std::copy(srcBeginIter, srcEndIter, dstMsgBuf);
  *dstEndIter = '\0';
}

void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       const char *srcMsg);
void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       llvm::Twine const &srcMsg);

/// Use the CustomStatus struct as a data member in a M_driver_driver struct to
/// manage custom (plugin specific) and ephemeral (one-shot dynamically
/// generated) status messages.
struct CustomStatus {
  /// Construct the CustomStatus struct.
  ///
  /// @param numCustomStatusCodes Number of custom status messages.
  CustomStatus(uint64_t numCustomStatusCodes)
      : numCustomStatusCodes(numCustomStatusCodes),
        customStatusCodeNext(M_DRIVER_STATUS_CUSTOM_CODES_BASE -
                             numCustomStatusCodes) {}

  /// Small buffer optimized storage for a small number of custom statuses.
  /// Access is guarded by `mutex`;
  llvm::SmallDenseMap<M_driver_status_t, llvm::SmallString<64>> messages;
  /// Number of custom status messages.
  int64_t numCustomStatusCodes;
  /// Status code for the next ephemeral status message.
  M_driver_status_t customStatusCodeNext;
  /// Serializes access to `messages` and `customStatusCodeNext`.
  std::mutex mutex;

  /// Create an ephemeral status code from the given message.
  ///
  /// @param message Message associated with the new ephemeral status.
  /// @return Unique custom status code, value of customStatusCodeNext.
  M_driver_status_t create(llvm::Twine message);

  /// Get a standard or custom status message from the given code.
  ///
  /// If the status code is ephemeral (less than or equal to
  /// M_DRIVER_STATUS_CUSTOM_CODES_BASE - numCustomStatusCodes), it will be
  /// removed from the messages store after being copied into the buffer.
  ///
  /// @param status Status code to get the message for.
  /// @param buffer User provided storage to write the message into.
  /// @param size Size in bytes of buffer.
  /// @return Either M_DRIVER_STATUS_SUCCESS on success or
  /// M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE when buffer or size are invalid or
  /// when the status code is not known.
  M_driver_status_t getMessage(M_driver_status_t status, char *buffer,
                               int64_t size);
};

} // namespace M::Target

#endif // TARGET_LIB_UTILS_UTILS_H
