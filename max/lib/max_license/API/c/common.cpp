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

#include "max/c/common.h"
#include "Config/Version.h"
#include "max/c/symbol_export.h"
#include "max/c/types.h"
#include "max/max_license/API/c/detail/types.h"

using namespace M;

MODULAR_API_EXPORT const char *M_version() { return getModularVersionString(); }

MODULAR_API_EXPORT M_Status *M_newStatus() {
  auto *status = new M_Status();
  status->status = success();
  return status;
}

MODULAR_API_EXPORT const char *M_getError(const M_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.getError();
}

MODULAR_API_EXPORT int M_isError(const M_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.isError();
}

MODULAR_API_EXPORT void M_freeStatus(M_Status *status) { delete status; }
