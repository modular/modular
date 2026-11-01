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

#include "_core/Driver/CompletionFlag.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include <stdexcept>
#include <string>

using namespace M::Driver;

Python::CompletionFlag::CompletionFlag(const Device &device) {
  M::Driver::DeviceContextRef ctx = device.ctx;
  // Non-CUDA backends return Error("allocateCompletionFlag is not
  // supported on this device") from the MLRT virtual default; propagate
  // that message verbatim to the Python caller.
  M::ErrorOr<M::Driver::CompletionFlagRef> flagOr =
      ctx->allocateCompletionFlag();
  if (flagOr.isError()) {
    throw std::runtime_error(std::string("CompletionFlag allocation failed: ") +
                             std::string(flagOr.getError()));
  }
  ref_ = std::move(*flagOr);
}
