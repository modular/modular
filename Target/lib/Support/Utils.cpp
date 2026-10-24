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
#include "Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstring>

namespace M::Target {

void copyStatusMessage(char *dstMessageBuffer, int64_t dstMessageBufferSize,
                       const char *srcErrorString) {
  int64_t messageLength = (int64_t)strlen(srcErrorString);
  if (messageLength >= dstMessageBufferSize)
    messageLength = dstMessageBufferSize - 1;
  [[maybe_unused]] auto _ =
      strncpy(dstMessageBuffer, srcErrorString, messageLength);
  dstMessageBuffer[messageLength] = '\0';
}

void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       llvm::Twine const &srcMsg) {
  llvm::SmallVector<char, 128> tmp;
  llvm::StringRef str = srcMsg.toStringRef(tmp);
  int64_t strSize = int64_t(str.size());
  int64_t msgSize =
      strSize < (dstMsgBufSize - 1) ? strSize : (dstMsgBufSize - 1);
  [[maybe_unused]] auto _ = strncpy(dstMsgBuf, str.data(), msgSize);
  dstMsgBuf[msgSize] = '\0';
}

M_driver_status_t CustomStatus::create(llvm::Twine message) {
  std::lock_guard<std::mutex> lock(mutex);
  M_driver_status_t status = customStatusCodeNext--;
  llvm::SmallString<64> &storage = messages[status];
  message.toVector(storage);
  return status;
}

M_driver_status_t CustomStatus::getMessage(M_driver_status_t status,
                                           char *buffer, int64_t size) {
  if (size <= 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  if (status > 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  if (status > -M_DRIVER_NUM_STANDARD_STATUS_CODES) {
    copyStatusMessage(buffer, size, standardStatusMessages[-status]);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (status > M_DRIVER_STATUS_CUSTOM_CODES_BASE)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;

  std::lock_guard<std::mutex> lock(mutex);

  auto iter = messages.find(status);
  if (iter != messages.end()) {
    copyStatusMessage(buffer, size, iter->second);
    if (status <= M_DRIVER_STATUS_CUSTOM_CODES_BASE - numCustomStatusCodes)
      messages.erase(iter);
  } else
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;

  return M_DRIVER_STATUS_SUCCESS;
}

} // namespace M::Target
