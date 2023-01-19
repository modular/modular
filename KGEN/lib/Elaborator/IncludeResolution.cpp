//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Elaborator.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

using namespace M;
using namespace KGEN;

/// Compare constraints ignoring their location.
static bool areSameConstraints(ArrayRef<ConstraintAttr> lhsConstraints,
                               ArrayRef<ConstraintAttr> rhsConstraints) {
  if (lhsConstraints.size() != rhsConstraints.size())
    return false;
  for (auto [lhs, rhs] : llvm::zip(lhsConstraints, rhsConstraints)) {
    if (std::make_pair(lhs.getExpr(), lhs.getMessage()) !=
        std::make_pair(rhs.getExpr(), rhs.getMessage()))
      return false;
  }
  return true;
}

/// When including files, some symbols may be duplicated. Attempt to reconcile
/// the `included` symbol with the current `symbol`. Returns failure if the
/// duplicate symbols could not be reconciled.
static LogicalResult reconcileDuplicateSymbol(StringRef name, Operation *symbol,
                                              Operation *included) {
  // Make sure they're the same kind of op.
  if (symbol->getName() != included->getName()) {
    InFlightDiagnostic diag = symbol->emitError("redefinition of symbol @")
                              << name << " as a " << symbol->getName();
    return diag.attachNote(included->getLoc())
           << "previously defined as a " << included->getName() << " here";
  }

  // If the symbol is a generator or function, it cannot be redefined.
  if (isa<FuncOp, GeneratorOp>(symbol)) {
    InFlightDiagnostic diag =
        symbol->emitError("redefinition of ")
        << (isa<FuncOp>(symbol) ? "function" : "generator") << " @" << name;
    return diag.attachNote(included->getLoc())
           << "see previous definition here";
  }

  // Interfaces can be redeclared. Ensure the declarations match.
  if (auto iface = dyn_cast<GeneratorInterfaceOp>(symbol)) {
    auto incIface = cast<GeneratorInterfaceOp>(included);

    // Emit nice diagnostics for the obvious possible differences: signatures
    // and constraints.
    if (failed(verifyDeclSignaturesMatch(
            "interface redeclaration", iface.getSignature(), iface.getLoc(),
            "previous interface declaration", incIface.getSignature(),
            incIface.getLoc())))
      return failure();

    // Compare the constraints ignoring the location. Take one of the sets of
    // constraints.
    if (!areSameConstraints(iface.getConstraints(),
                            incIface.getConstraints())) {
      return (symbol->emitError("interface @")
              << name << " was redeclared with different constraints")
                 .attachNote(included->getLoc())
             << "previous declaration here";
    }
    iface.setConstraintsAttr(incIface.getConstraintsAttr());

    // Just check the attributes now.
    if (symbol->getAttrDictionary() != included->getAttrDictionary()) {
      InFlightDiagnostic diag =
          symbol->emitError("redeclaration of interface @")
          << name << " has different attributes";
      return diag.attachNote(included->getLoc())
             << "see previous declaration here";
    }

    // Identical interface declarations can be reconciled if they match.
    return success();
  }

  return included->emitError("included symbol @")
         << name
         << " is something other than a function, generator, interface, or "
            "type";
}

/// Recursively resolve included files according to the provided search paths,
/// appending the included IR to the main module. Include each file at most
/// once.
static LogicalResult
resolveInclude(SymbolTable &symtab, IncludeOp include,
               ArrayRef<std::filesystem::path> searchPaths,
               DenseSet<StringAttr> &loadedFiles,
               SmallVectorImpl<std::string> *includedFiles) {
  if (!loadedFiles.insert(include.getFileNameAttr()).second) {
    include->erase();
    return success();
  }

  std::string modulePath;
  if (std::filesystem::path(include.getFileName().str()).is_absolute()) {
    modulePath = include.getFileName().str();
  } else {
    for (const auto &p : searchPaths) {
      auto testPath = p / std::filesystem::path(include.getFileName().str());
      if (!std::filesystem::exists(testPath))
        continue;

      modulePath = testPath.string();
      break;
    }
    if (modulePath.empty())
      return include->emitError("could not find file '")
             << include.getFileName() << "'";
  }

  // Record this file if requested.
  if (includedFiles)
    includedFiles->push_back(modulePath);

  auto includedModule =
      mlir::parseSourceFile<ModuleOp>(modulePath, include->getContext());
  if (!includedModule)
    return mlir::emitError(include.getLoc(),
                           "failed to parse included source file");

  // Recursively resolve transitive includes.
  for (auto inc :
       llvm::make_early_inc_range(includedModule->getOps<IncludeOp>()))
    if (failed(resolveInclude(symtab, inc, searchPaths, loadedFiles,
                              includedFiles)))
      return failure();

  // Prepend all the ops to the main module.
  for (Operation &included :
       llvm::make_early_inc_range(includedModule->getOps())) {
    auto name =
        included.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (!name)
      return included.emitError(
          "unexpected top-level operation that is not a symbol");
    Operation *symbol = symtab.lookup(name);
    // If there is no conflict, just add it.
    if (!symbol) {
      included.remove();
      symtab.insert(&included);
      included.moveAfter(included.getBlock(), included.getBlock()->begin());
      continue;
    }
    if (failed(reconcileDuplicateSymbol(name, symbol, &included)))
      return failure();
    // The symbols match, so just ignore the include.
  }

  include->erase();
  return success();
}

LogicalResult M::resolveIncludes(SymbolTable &symtab,
                                 ArrayRef<std::filesystem::path> searchPaths,
                                 SmallVectorImpl<std::string> *includedFiles) {
  DenseSet<StringAttr> loadedFiles;
  for (auto include : llvm::make_early_inc_range(
           cast<ModuleOp>(symtab.getOp()).getOps<IncludeOp>()))
    if (failed(resolveInclude(symtab, include, searchPaths, loadedFiles,
                              includedFiles)))
      return failure();
  return success();
}
