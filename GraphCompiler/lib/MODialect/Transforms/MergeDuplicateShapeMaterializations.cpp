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

#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "llvm/ADT/STLExtras.h"

#include <unordered_map>

using namespace M;
using namespace MO;

#define DEBUG_TYPE "merge-duplicate-shape-materializations"

namespace std {
template <>
struct hash<::mlir::ValueRange> {
  std::size_t operator()(const ::mlir::ValueRange &s) const noexcept {
    return ::llvm::hash_combine_range(s.begin(), s.end());
  }
};

template <>
struct hash<std::pair<::llvm::StringRef, ::mlir::ValueRange>> {
  std::size_t operator()(const std::pair<::llvm::StringRef, ::mlir::ValueRange>
                             &s) const noexcept {
    return ::llvm::hash_combine(::llvm::hash_value(s.first),
                                std::hash<::mlir::ValueRange>{}(s.second));
  }
};
} // namespace std
namespace M::MO {

#define GEN_PASS_DEF_MERGEDUPLICATESHAPEMATERIALIZATIONS

#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {
/// Maps (op name, operands) to the primary op and the parameters it defines,
/// so duplicates' parameters can be remapped onto the primary's.
using OpCacheTy = std::unordered_map<
    std::pair<llvm::StringRef, ValueRange>,
    std::pair<Operation *, SmallVector<KGEN::ParamDeclRefAttr>>>;

bool isOpDefiningOpForAllParams(Operation *op,
                                ArrayRef<KGEN::ParamDeclRefAttr> references,
                                const KGEN::ParameterUseDefGraph &useDefGraph) {

  SmallVector<KGEN::ParamDefinition> definitions;
  for (const auto &reference : references) {
    if (auto it = useDefGraph.defs.find(reference.getName());
        it != useDefGraph.defs.end() && it->second.defOp == op)
      definitions.emplace_back(it->second);
    else
      return false;
  }
  return llvm::all_of(definitions, [&](const KGEN::ParamDefinition &def) {
    return isOpDefiningOpForAllParams(op, def.uses, useDefGraph);
  });
}
} // namespace

/// Erases all but the first of each set of shape materialization ops with the
/// same name and operands that define all their parameters, remapping the
/// erased ops' parameters to the first's. CSE can't merge these itself because
/// their parameter references differ.
struct MergeDuplicateShapeMaterializationsPass
    : public MO::impl::MergeDuplicateShapeMaterializationsBase<
          MergeDuplicateShapeMaterializationsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    auto context = &getContext();
    IRRewriter rewriter(context);

    OpCacheTy opCache;

    auto paramCache = KGEN::ParameterCollector::Analysis(graph);
    KGEN::ParameterUseDefGraph useDefGraph(graph.getRegion());
    useDefGraph.calculate(paramCache);

    llvm::DenseMap<StringAttr, TypedAttr> paramMappings;
    llvm::SmallVector<Operation *> toErase;
    for (const auto &opUse : useDefGraph.opUses) {
      Operation *op = opUse.first;
      const SmallVector<KGEN::ParamDeclRefAttr> &references = opUse.second;

      if (!isa<ShapeMaterialization>(op))
        continue;

      auto definesAllParams =
          isOpDefiningOpForAllParams(op, references, useDefGraph);

      if (!definesAllParams)
        continue;

      auto [it, inserted] = opCache.try_emplace(
          {op->getName().getStringRef(), ValueRange(op->getOperands())}, opUse);

      if (inserted)
        continue;

      for (auto [thisOpReference, primaryOpReference] :
           llvm::zip(references, it->second.second)) {
        paramMappings[thisOpReference.getName()] = primaryOpReference;
      }

      op->replaceAllUsesWith(it->second.first);
      toErase.emplace_back(op);
    }

    MOGG::remapTypesAndAttrs(graph, paramMappings);

    for (auto op : toErase)
      op->erase();
  }
};
