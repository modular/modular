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

#ifndef KERNELS_WEIGHTSREGISTRY_H
#define KERNELS_WEIGHTSREGISTRY_H

#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "weights_registry_c_api.h"
#include <cstddef>

namespace M {

using MojoWeightsRegistryPtr =
    std::unique_ptr<void, decltype(&destroy_weights_registry)>;

class WeightsRegistry;
using WeightsRegistryRef = RCRef<WeightsRegistry>;

class WeightsRegistry : public ReferenceCounted<WeightsRegistry> {
public:
  /// Fallible constructor for the weights registry.
  static ErrorOr<WeightsRegistryRef> create(ArrayRef<const char *> names,
                                            ArrayRef<const std::byte *> data);

  explicit WeightsRegistry(MojoWeightsRegistryPtr weights)
      : mojoWeights(std::move(weights)) {}

  /// Frees the owned mojo weights registry on destruction.
  /// Note that the weights themselves are borrowed, not owned by this class.
  ~WeightsRegistry() = default;

  /// Deleted move operations due to ReferenceCounted.
  WeightsRegistry(WeightsRegistry &&other) = delete;
  WeightsRegistry &operator=(WeightsRegistry &&other) = delete;

  /// Looks up the registered weight data for name.
  ErrorOr<const std::byte *> operator[](const char *name) const;

  /// Return opaque pointer to mojo type.
  void *opaque() const { return mojoWeights.get(); }

private:
  /// Disallow copying since this owns the weights registry (and frees it).
  WeightsRegistry(const WeightsRegistry &) = delete;
  WeightsRegistry &operator=(const WeightsRegistry &) = delete;

  /// A void pointer to the Mojo weights registry struct.
  MojoWeightsRegistryPtr mojoWeights;
};

} // namespace M

#endif // KERNELS_WEIGHTSREGISTRY_H
