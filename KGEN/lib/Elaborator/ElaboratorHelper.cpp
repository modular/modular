//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ElaboratorHelper.h"
#include "IREvaluatorContext.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace KGEN;

ErrorTreeOr<std::pair<StringAttr, GeneratorOp>>
KGEN::getExpectedMangledName(Location errorLoc, StringRef errorContext,
                             TypedAttr symCst, SymbolTable &symTab,
                             bool allowParametric, bool sanitize) {
  auto symbol = extractSymbolConstantAttr(symCst);
  if (!symbol) {
    return ErrorTree(
        errorLoc,
        "'" + errorContext +
            "' function argument did not resolve to a concrete function");
  }
  if (!symbol.getType().isFullyBound()) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "'" << errorContext << "' function is not fully bound: "
       << symbol.getSymbol().getLeafReference().getValue() << " missing "
       << symbol.getType().getInputParamTypes().size()
       << " parameter binding(s)";
    return ErrorTree(errorLoc, errMsg);
  }
  auto func = symTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr());

  if (!func) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "'" << errorContext
       << "' expected a valid generator reference, but got "
       << symbol.getSymbol().getLeafReference().getValue() << "\n";
    return ErrorTree(errorLoc, errMsg);
  }

  // If the generator has a constant linkage name, that is the final name.
  StringAttr baseName;
  if (auto linkageName =
          dyn_cast_if_present<StringAttr>(func.getLinkageNameAttr())) {
    baseName = linkageName;
  } else {
    baseName =
        StringAttr::get(func.getContext(),
                        mangleParameterValues(func, symbol.getParamValues()));
  }
  if (sanitize)
    baseName = sanitizeSymbolToAlnum(baseName);

  return std::make_pair(baseName, func);
}

static void
replaceSymNames(Operation *op,
                const DenseMap<SymbolRefAttr, StringAttr> &symToRename) {
  if (symToRename.empty())
    return;

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&symToRename](SymbolConstantAttr attr) {
    auto iter = symToRename.find(attr.getSymbol());
    if (iter != symToRename.end())
      return SymbolConstantAttr::get(iter->second, attr.getType(),
                                     attr.getParamValues());
    return attr;
  });

  replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);
}

void KGEN::renameFunctions(mlir::ModuleOp theModule, bool isGPU, bool &failed) {
  DenseMap<SymbolRefAttr, StringAttr> symToRename;
  DenseMap<StringAttr, FuncOp> renamedTo;

  for (auto func : theModule.getOps<FuncOp>()) {
    StringAttr newName;

    if (auto linkageNameAttr = func.getLinkageNameAttr()) {
      if (!isa<StringAttr>(linkageNameAttr)) {
        failed = true;
        mlir::emitError(func.getLoc()) << "unable to resolve `linkageName` '"
                                       << linkageNameAttr << "' to a string";
        continue;
      }
      // Convert from a !kgen.string-typed StringAttr to a built-in StringAttr.
      newName = StringAttr::get(theModule.getContext(),
                                cast<StringAttr>(linkageNameAttr).getValue());
      func.removeLinkageNameAttr();
    }

    if (isGPU)
      newName =
          sanitizeSymbolToAlnum(newName ? newName : func.getSymNameAttr());

    if (!newName || newName == func.getSymNameAttr())
      continue;

    auto [it, inserted] = renamedTo.try_emplace(newName, func);
    if (!inserted) {
      failed = true;
      mlir::emitError(func.getLoc()) << "duplicate functions named " << newName;
      mlir::emitRemark(it->second.getLoc()) << "existing function here";
      continue;
    }
    symToRename[FlatSymbolRefAttr::get(func.getSymNameAttr())] = newName;
    func.setSymName(newName);
    DebugInfo::updateSubprogram(func, newName);
  }

  replaceSymNames(theModule, symToRename);
}
