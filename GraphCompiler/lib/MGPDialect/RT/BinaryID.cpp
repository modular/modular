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

#include "GraphCompiler/MGPDialect/RT/BinaryID.h"
#include "Support/BinaryID.h"

MODULAR_CXX_EXPORT std::string M::MGPRT::getMGPRTBinaryID() {
  // The ID of the shared library containing this code, so MEF caches are
  // invalidated when it changes.
  return M::getBinaryID();
}
