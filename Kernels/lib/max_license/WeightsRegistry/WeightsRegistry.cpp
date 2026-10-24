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

#include "Kernels/max_license/WeightsRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"

using namespace M;

static constexpr int kResultSuccess = 0;

ErrorOr<WeightsRegistryRef>
WeightsRegistry::create(ArrayRef<const char *> names,
                        ArrayRef<const std::byte *> data) {
  if (names.size() != data.size())
    return Error("expected equal length names (" + Twine(names.size()) +
                 ") and data (" + Twine(data.size()) + ") arrays");

  void *weights;
  if (create_weights_registry(&weights, (void *)names.data(),
                              (void *)data.data(),
                              data.size()) == kResultSuccess) {
    return WeightsRegistryRef::create(
        MojoWeightsRegistryPtr(weights, destroy_weights_registry));
  }
  return Error("unable to create weights registry");
}

ErrorOr<const std::byte *> WeightsRegistry::operator[](const char *name) const {
  void *data;
  if (weights_registry_lookup(&data, mojoWeights.get(),
                              const_cast<char *>(name)) == kResultSuccess)
    return llvm::bit_cast<const std::byte *>(data);

  return Error("unable to look up " + Twine(name));
}
