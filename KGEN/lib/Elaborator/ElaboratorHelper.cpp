//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ElaboratorHelper.h"
#include "IREvaluatorContext.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
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
                             bool sanitize) {
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

  // Determine the final base name from an explicit linkage name or the
  // auto-mangled parameter values.
  // TODO: When mangle=true, hash the auto-mangled parameter values into the
  // prefix to guarantee uniqueness across instantiations (e.g.
  // "my_kernel_a3f2c1b0"). For now both mangle=true and mangle=false
  // use the prefix verbatim; PTX sanitization is applied afterwards
  // when sanitize=true.
  StringAttr baseName;
  if (auto lna = func.getLinkageNameAttr()) {
    if (auto prefix = dyn_cast<StringAttr>(lna.getName()))
      baseName = StringAttr::get(func.getContext(), prefix.getValue());
  }
  if (!baseName)
    baseName =
        StringAttr::get(func.getContext(),
                        mangleParameterValues(func, symbol.getParamValues()));
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

  for (FuncOp func : theModule.getOps<FuncOp>()) {
    StringAttr newName;

    if (auto linkageNameAttr = func.getLinkageNameAttr()) {
      auto prefixAttr = dyn_cast<StringAttr>(linkageNameAttr.getName());
      if (!prefixAttr) {
        failed = true;
        mlir::emitError(func.getLoc())
            << "unable to resolve `linkageName` '" << linkageNameAttr.getName()
            << "' to a string before renaming";
        func.removeLinkageNameAttr();
        continue;
      }
      StringRef prefix = prefixAttr.getValue();
      // Use the linkage name verbatim. For GPU functions, the isGPU block
      // below will apply sanitizeSymbolToAlnum, just like any other function.
      // TODO: When mangle=true, hash the auto-mangled parameter values into
      // the prefix to guarantee uniqueness across instantiations.
      newName = StringAttr::get(theModule.getContext(), prefix);
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
