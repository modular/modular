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

#include "max/c/weights.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "Support/ErrorOr.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"
#include "llvm/ADT/bit.h"

MODULAR_API_EXPORT M_WeightsRegistry *M_newWeightsRegistry(const char **names,
                                                           const void **data,
                                                           size_t numWeights,
                                                           M_Status *status) {
  if (!status)
    return nullptr;

  if (numWeights > 0 && (!names || !data)) {
    M::Detail::setError(status, M::Error::getStaticString(
                                    "names and data must be provided when the "
                                    "number of weights is greater than zero"));
    return nullptr;
  }

  M::ErrorOr<M::WeightsRegistryRef> weightsOr = M::WeightsRegistry::create(
      {names, numWeights},
      {llvm::bit_cast<const std::byte *const *>(data), numWeights});
  if (weightsOr.isError()) {
    M::Detail::setError(status, weightsOr.takeError());
    return nullptr;
  }

  M::Detail::setOK(status);
  return new M_WeightsRegistry{weightsOr.takeValue()};
}

MODULAR_API_EXPORT void
M_freeWeightsRegistry(M_WeightsRegistry *weightsRegistry) {
  delete weightsRegistry;
}
