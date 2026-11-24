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

#include "GraphCompiler/MODialect/Analysis/StaticReshapeDimensionsMappingAnalysis.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"

namespace M::MO {

StaticReshapeDimensionsMappingAnalysis::StaticReshapeDimensionsMappingAnalysis(
    llvm::ArrayRef<mlir::TypedAttr> before,
    llvm::ArrayRef<mlir::TypedAttr> after)
    : forward(before.size()), backward(after.size()) {

  if (before.empty() || after.empty()) {
    isSupported = false;
    return;
  }

  const auto calculateStaticVolume = [](llvm::ArrayRef<mlir::TypedAttr> dims) {
    int64_t total = 1;
    for (auto dimAttr : dims) {
      if (auto dimInt = tryGetConstantInt(dimAttr)) {
        total *= dimInt.value();
      }
    }

    return total;
  };

  const int64_t beforeTotalVolume = calculateStaticVolume(before);
  const int64_t afterTotalVolume = calculateStaticVolume(after);
  // Early check for mixing of static and dynamic dimensions.
  if (beforeTotalVolume != afterTotalVolume) {
    isSupported = false;
    return;
  }

  int64_t beforeVolume = 1;
  int64_t afterVolume = 1;

  // Enabled means we can increment it.
  bool beforeEnabled = true;
  bool afterEnabled = true;

  /// TODO: Apply the algorithm per range of static dimensions; today it fails
  /// whenever dynamic and static dimensions alternate, even without mixing.
  /// Works: 3, 5, 6, D0 -> 15, 3, 2, D0
  /// Works: D0, 3, 5, 6 -> D0, 15, 3, 2
  /// Does not work: 3, D0, 5, 6 -> 3, D0, 30
  /// Does not work: D0, 3, 5, 6, D1 -> D0, 15, 3, 2, D1
  /// Does not work: D0, 3, 5, 6 -> D2, D3, 3, 5, 6, D4
  const bool startsWithStaticDimension =
      llvm::isa<mlir::IntegerAttr>(before.front());
  const int64_t iterationOrder = startsWithStaticDimension ? 1 : -1;
  uint64_t beforeIdx = startsWithStaticDimension ? 0 : before.size() - 1;
  uint64_t afterIdx = startsWithStaticDimension ? 0 : after.size() - 1;

  while (beforeVolume != beforeTotalVolume || afterVolume != afterTotalVolume) {
    auto beforeIntegerAttr = tryGetConstantInt(before[beforeIdx]);
    auto afterIntegerAttr = tryGetConstantInt(after[afterIdx]);

    // Mixing of static and dynamic dimensions.
    if (bool(beforeIntegerAttr) ^ bool(afterIntegerAttr)) {
      isSupported = false;
      return;
    }

    if (beforeIntegerAttr) {
      int64_t beforeDim = beforeIntegerAttr.value();
      int64_t afterDim = afterIntegerAttr.value();

      // 0 dims make the mapping ambiguous:
      // [0, 1, 2] -> [5000, 0, 500, 42, 0].
      if (beforeDim == 0 || afterDim == 0) {
        isSupported = false;
        return;
      }

      if (beforeEnabled)
        beforeVolume *= beforeDim;
      if (afterEnabled)
        afterVolume *= afterDim;

      beforeEnabled = beforeVolume <= afterVolume;
      afterEnabled = afterVolume <= beforeVolume;

      if (beforeDim != 1 && afterDim != 1) {
        forward[beforeIdx].push_back(afterIdx);
        backward[afterIdx].push_back(beforeIdx);
      }
    }

    beforeIdx += iterationOrder * beforeEnabled;
    afterIdx += iterationOrder * afterEnabled;
  }

  // Mapping should be sorted.
  if (!startsWithStaticDimension) {
    for (auto &dimMapping : forward)
      std::reverse(dimMapping.begin(), dimMapping.end());

    for (auto &dimMapping : backward)
      std::reverse(dimMapping.begin(), dimMapping.end());
  }

  isSupported = true;
}

} // namespace M::MO
