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

#include "GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace M::MOGG {

static TypedAttr
tryReplace(KGEN::ParamDeclRefAttr attr,
           const llvm::DenseMap<StringAttr, TypedAttr> &oldNameToNew) {
  while (true) {
    auto itr = oldNameToNew.find(attr.getName());
    if (itr == oldNameToNew.end() || itr->second == attr)
      return attr;

    auto ref = dyn_cast<KGEN::ParamDeclRefAttr>(itr->second);
    if (!ref)
      return itr->second;

    attr = ref;
  }
}

// Any reference to a parameter must be checked against our map and remapped if
// it is something we should be remapping.
void remapTypesAndAttrs(
    Operation *op, const llvm::DenseMap<StringAttr, TypedAttr> &oldNameToNew,
    bool skipRootOp) {
  mlir::AttrTypeReplacer walker;
  walker.addReplacement(
      [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
        return tryReplace(attr, oldNameToNew);
      });

  if (!skipRootOp) {
    walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  } else {
    // Not using walk or else we will double walk (recursivelyReplaceElementsIn
    // will do it too).
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Operation &subOp : block.getOperations()) {
          walker.recursivelyReplaceElementsIn(&subOp, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
        }
      }
    }
  }
}

// Scoping aware name mappings for the parameter uniquing process.
using DeclMap =
    llvm::ScopedHashTable<KGEN::ParamDeclRefAttr, KGEN::ParamDeclRefAttr>;
using DeclMapScope =
    llvm::ScopedHashTableScope<KGEN::ParamDeclRefAttr, KGEN::ParamDeclRefAttr>;

// Walk IR recursively, renaming parameters in a predictable away to help dedup
// compilation of identical code.
//
// `paramRenaming` holds the renaming for in-scope parameters; keeping it up to
// date is most of the complexity. `totalParamsSeen` generates unique names.
//
// Parameters are in scope if they are declared in the same block as an
// operation using it:
//
//  kgen.param.declare A = ...
//  kgen.param.declare B = A + 1 // A is in scope
//
// Or if they are declared as input parameters of an enclosing operation, such
// as a `mo.graph`. Each such operation opens its own scope, so two sibling
// graphs may declare parameters of the same name independently.
static void uniqueAllParamsHelper(Operation *current, DeclMap &paramRenaming,
                                  size_t &totalParamsSeen) {

  // Gives `oldDecl` a new name and records it in the given list.
  auto getNewDecl = [&](KGEN::ParamDeclAttr oldDecl) -> KGEN::ParamDeclAttr {
    auto oldRef = KGEN::ParamDeclRefAttr::get(oldDecl);
    std::string newName = "__P" + std::to_string(totalParamsSeen++);
    auto newDecl = KGEN::ParamDeclAttr::get(newName, oldDecl.getType());
    ASSERT_STREAM(!paramRenaming.count(oldRef),
                  << "Shadowing of parameter names detected! This is not "
                     "supported right now.");
    paramRenaming.insert(oldRef, KGEN::ParamDeclRefAttr::get(newDecl));
    return newDecl;
  };

  auto getNewDecls = [&](ArrayRef<KGEN::ParamDeclAttr> oldDecls) {
    return llvm::map_to_vector(oldDecls, getNewDecl);
  };

  DeclMapScope regionScope(paramRenaming);

  // Pre-order: an op's declared input parameters are in scope for it. Ideally
  // any DeclInterface op, but the interface can't update its declared params.
  if (auto op = dyn_cast<MO::GraphOp>(current)) {
    auto inputDecls = op.getInputParameters();
    auto newInputDecls = getNewDecls(inputDecls);
    op.setInputParameters(newInputDecls);
  }

  // Conduct renaming with renamings in scope
  mlir::AttrTypeReplacer replaceRefs;
  replaceRefs.addReplacement(
      [&](KGEN::ParamDeclRefAttr oldRef) -> std::optional<TypedAttr> {
        if (auto newRef = paramRenaming.lookup(oldRef))
          return newRef;
        return {};
      });
  replaceRefs.replaceElementsIn(current, /*replaceAttrs=*/true,
                                /*replaceLocs=*/false,
                                /*replaceTypes=*/true);

  for (Region &region : current->getRegions()) {
    for (Block &block : region.getBlocks()) {
      DeclMapScope blockScope(paramRenaming);

      // All parameters in the same block are in scope of each other.
      for (auto paramDecl : block.getOps<KGEN::ParamOpInterface>()) {
        SmallVector<ParamDeclAttr> newParamDecls;
        paramDecl.walkDeclarations([&](ParamDeclAttr decl) {
          newParamDecls.push_back(getNewDecl(decl));
        });

        paramDecl.renameDeclarations(newParamDecls);
      }

      // `paramRenaming` is now accurate, begin recursive descent.
      for (Operation &next : block.getOperations())
        uniqueAllParamsHelper(&next, paramRenaming, totalParamsSeen);
    }
  }
}

// Unique all the parameters to be less sensitive to the name of the parameter.
void uniqueAllParams(Operation *graph) {
  // unordered_map (open chaining) over DenseMap (open addressing) because
  // deletions are common.
  DeclMap paramRenaming;

  size_t totalParamsRenamed = 0;
  uniqueAllParamsHelper(graph, paramRenaming, totalParamsRenamed);
}

void normalizeAllNames(ModuleOp module) {
  auto ctx = module.getContext();

  llvm::DenseMap<SymbolRefAttr, FlatSymbolRefAttr> newNames;

  for (auto [index, symbol] :
       llvm::enumerate(module.getOps<mlir::SymbolOpInterface>())) {
    auto oldName = symbol.getNameAttr();
    auto newName = "__S" + std::to_string(index);
    symbol.setSymbolName(newName);

    newNames[FlatSymbolRefAttr::get(oldName)] =
        FlatSymbolRefAttr::get(ctx, newName);
  }

  auto ops = llvm::to_vector(llvm::make_pointer_range(module.getOps()));

  mlir::parallelForEach(ctx, ops, [&](Operation *op) {
    uniqueAllParams(op);

    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](FlatSymbolRefAttr sym) { return newNames.lookup_or(sym, {}); });

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  });
}

} // namespace M::MOGG
