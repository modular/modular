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

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

#define GEN_PASS_DEF_REWRITEOPAQUETYPES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct RewriteOpaqueTypesPass
    : public impl::RewriteOpaqueTypesBase<RewriteOpaqueTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    // Simple -> fully-qualified mapping persisted by `mogg-import-packages`.
    // Absence is a valid no-op.
    auto mappingAttr =
        module->getAttrOfType<mlir::DictionaryAttr>(kOpaqueTypeMappingAttrName);
    // Safety net for the kernel-struct exclusion in `buildOpaqueTypeMapping`:
    // an `!mo.opaque<"Foo">` naming a registered kernel struct fails here
    // instead of far from the cause during later lowering.
    llvm::StringSet<> kernelLeafNames;
    auto leafOf = [](llvm::StringRef name) -> llvm::StringRef {
      return name.rsplit('.').second.empty() ? name : name.rsplit('.').second;
    };
    for (auto decl : module.getOps<KernelDeclOp>())
      kernelLeafNames.insert(leafOf(decl.getSymName()));
    for (auto decl : module.getOps<PrimitiveDeclOp>())
      kernelLeafNames.insert(leafOf(decl.getSymName()));

    bool sawKernelStructMisuse = false;
    if (mappingAttr && !mappingAttr.empty()) {
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](MO::OpaqueType opaqueTy) -> std::optional<mlir::Type> {
            llvm::StringRef symbol = opaqueTy.getSymbol().getValue();
            auto qualified = mappingAttr.getNamed(symbol);
            if (!qualified) {
              // Kernel structs are resolved by registration, not opaque data
              // types. Qualified or external opaque types have dotted names,
              // so this fires only on that misuse.
              if (kernelLeafNames.contains(symbol)) {
                module.emitError()
                    << "opaque type '" << symbol
                    << "' refers to a registered kernel struct; kernel "
                       "structs are implementations resolved by registration, "
                       "not opaque data types";
                sawKernelStructMisuse = true;
              }
              return std::nullopt;
            }
            auto qualifiedStr =
                mlir::dyn_cast<mlir::StringAttr>(qualified->getValue());
            if (!qualifiedStr)
              return std::nullopt;
            return MO::OpaqueType::get(qualifiedStr, opaqueTy.getParameters());
          });
      replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
      if (sawKernelStructMisuse)
        return signalPassFailure();
    }

    // Downstream passes use fully-qualified names only.
    module->removeAttr(kOpaqueTypeMappingAttrName);
  }
};

} // namespace

} // namespace M::MOGG
