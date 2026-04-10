//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ElaboratorHelper.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/Support/NameMangling.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace KGEN;

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

/// Apply the @__name linkage name formula for GPU targets.
///
/// This function is called from two places that must produce identical names:
///
///   1. renameFunctions (post-elaboration, GPU side): renames the concrete
///      FuncOp in the PTX module to its final symbol.
///
///   2. applyLinkageName (during parameter evaluation, host side): computes
///      the name that get_linkage_name will return so the host runtime knows
///      what PTX symbol to dispatch to.
///
/// Centralizing the formula here ensures the PTX symbol written into the
/// binary and the name the host looks up are always identical.
StringAttr KGEN::applyGPULinkageName(StringAttr resolved, LinkageNameAttr lna,
                                     StringRef symName,
                                     mlir::FunctionType funcType) {
  if (!lna.getMangle().getValue())
    return sanitizeSymbolToUnderscores(resolved);
  // mangle=true: sanitize the @__name prefix, then append a uniqueness suffix
  // derived from symName and the printed function type. This matches the PTX
  // symbol produced by the GPU rename loop (renameFunctions) and the name
  // looked up by the host (get_linkage_name / applyLinkageName).
  std::string funcTypeStr;
  llvm::raw_string_ostream os(funcTypeStr);
  funcType.print(os);
  StringAttr sanitized =
      sanitizeSymbolToUnderscores(resolved, std::numeric_limits<size_t>::max());
  return appendAutoMangledSuffix(sanitized, symName, funcTypeStr);
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
      StringAttr userName = StringAttr::get(theModule.getContext(), prefix);
      if (isGPU) {
        newName =
            applyGPULinkageName(userName, linkageNameAttr, func.getSymName(),
                                func.getFunctionType());
      } else {
        // mangle=false + non-GPU: use the linkage name verbatim.
        newName = StringAttr::get(theModule.getContext(), prefix);
      }
      func.removeLinkageNameAttr();
    }

    if (isGPU) {
      // When @__name is set, newName is already set above.
      // When @__name is absent, sanitize the auto-mangled sym_name as before.
      if (!newName)
        newName = sanitizeSymbolToAlnum(func.getSymNameAttr());
    }

    if (!newName || newName == func.getSymNameAttr())
      continue;

    auto [it, inserted] = renamedTo.try_emplace(newName, func);
    if (!inserted) {
      failed = true;
      mlir::emitError(func.getLoc()) << "duplicate functions named " << newName;
      mlir::emitRemark(it->second.getLoc()) << "existing function here";
      continue;
    }
    StringAttr oldSym = func.getSymNameAttr();
    symToRename[FlatSymbolRefAttr::get(oldSym)] = newName;
    func.setSymName(newName);
    DebugInfo::updateSubprogram(func, newName);

    // On host targets, also rename the companion populate_captures stub when
    // a host function carries @__name. GPU kernels live only in standalone GPU
    // modules and never have stubs in the host module, so this is a no-op for
    // GPU targets.
    if (!isGPU) {
      MLIRContext *ctx = theModule.getContext();
      SmallString<128> stubOldStr(oldSym.getValue());
      stubOldStr += "_populate_captures";
      SmallString<128> stubNewStr(newName.getValue());
      stubNewStr += "_populate_captures";
      if (Operation *stubOp = mlir::SymbolTable::lookupSymbolIn(
              theModule, StringRef(stubOldStr))) {
        // populate_captures stubs are always FuncOps — created as such in
        // evaluateCompileOffloadClosureAttr. A non-FuncOp here is a bug.
        auto stubFunc = cast<FuncOp>(stubOp);
        StringAttr stubOldAttr = StringAttr::get(ctx, StringRef(stubOldStr));
        StringAttr stubNewAttr = StringAttr::get(ctx, StringRef(stubNewStr));
        symToRename[FlatSymbolRefAttr::get(stubOldAttr)] = stubNewAttr;
        stubFunc.setSymName(stubNewAttr);
      }
    }
  }

  replaceSymNames(theModule, symToRename);
}
