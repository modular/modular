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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_STATICRESHAPEDIMENSIONSMAPPINGANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_STATICRESHAPEDIMENSIONSMAPPINGANALYSIS_H

#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>

namespace M::MO {

/// Maps input to output dimensions across a reshape. Dynamic dimensions are
/// ignored; fails when dynamic dimensions mix with static ones.
/// EXAMPLE:
/// Works: D0, 4, 6, 1 -> D0, 2, 3, 2, 1, 2
///   Forward: [[], [1, 2], [2, 3, 5]]
///   Backward: [[], [1], [1, 2], [2], [2]]
/// Does not work: D0, 4, 6, 1 -> 4, D0, 6, 1
///
/// 1s map to nothing, since they could map to any neighbor.
///
/// A `before` dim maps to the `after` dims it contributes to, which lets e.g.
/// a `mo.split` axis move across a `mo.reshape` without touching other dims.
class StaticReshapeDimensionsMappingAnalysis {
public:
  // Before := shape before the reshape
  // After := shape after the reshape
  StaticReshapeDimensionsMappingAnalysis(llvm::ArrayRef<mlir::TypedAttr> before,
                                         llvm::ArrayRef<mlir::TypedAttr> after);
  ~StaticReshapeDimensionsMappingAnalysis() = default;

  inline operator bool() const { return isSupported; }

  using DimensionsMapping = llvm::SmallVector<llvm::SmallVector<int64_t>>;

  // Maps before dimensions to after dimensions.
  // Example:
  // Before  : [1, 2, 3, 4]
  // After   : [1, 2, 3, 2, 2]
  // Forward : [[], [1], [2], [3, 4]]
  inline const DimensionsMapping &forwardMapping() const {
    assert(isSupported);
    return forward;
  }

  // Maps after dimensions to before dimensions.
  // Example:
  // Before   : [1, 2, 3, 4]
  // After    : [1, 2, 3, 2, 2]
  // Backward : [[], [1], [2], [3], [3]]
  inline const DimensionsMapping &backwardMapping() const {
    assert(isSupported);
    return backward;
  }

private:
  DimensionsMapping forward;
  DimensionsMapping backward;
  bool isSupported;
};

} // namespace M::MO

#endif
