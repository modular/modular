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

    if (iface.getConstraintsAttr() != incIface.getConstraintsAttr()) {
      return (symbol->emitError("interface @")
              << name << " was redeclared with different constraints")
                 .attachNote(included->getLoc())
             << "previous declaration here";
    }

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

  // Structs can be redeclared. Ensure parameters and fields match.
  if (auto type = dyn_cast<StructDeclOp>(symbol)) {
    auto incType = cast<StructDeclOp>(included);

    // Emit nice diagnostics for for the obvious possible differences: input
    // parameters.
    if (failed(verifyParamDeclsMatch(
            "struct redeclaration", type.getParamDecls(), type.getLoc(),
            "previous struct declaration", incType.getParamDeclsAttr(),
            incType.getLoc())))
      return failure();

    // Check other attributes.
    if (symbol->getAttrDictionary() != included->getAttrDictionary()) {
      InFlightDiagnostic diag =
          symbol->emitError("redeclaration of interface @")
          << name << " has different attributes";
      return diag.attachNote(included->getLoc())
             << "see previous declaration here";
    }

    // Check that the fields match.
    unsigned numFields = std::distance(type.field_begin(), type.field_end());
    unsigned incNumFields =
        std::distance(incType.field_begin(), incType.field_end());
    if (numFields != incNumFields) {
      InFlightDiagnostic diag = symbol->emitError("type @")
                                << name << " redeclared with " << numFields
                                << " fields";
      diag.attachNote(included->getLoc())
          << "previously declared with " << incNumFields << " fields here";
      return failure();
    }

    unsigned i = 0;
    auto checkStructField = [&](StructFieldOp lhs, StructFieldOp rhs, auto lhsE,
                                auto rhsE, StringRef kind) {
      if (lhsE == rhsE)
        return false;
      InFlightDiagnostic diag = lhs->emitError("struct @")
                                << name << " field #" << i
                                << " redeclared with different " << kind << " "
                                << lhsE;
      diag.attachNote(rhs.getLoc())
          << "previously declared as " << rhsE << " here";
      return true;
    };
    for (auto [lhs, rhs] :
         llvm::zip(type.getFieldDecls(), incType.getFieldDecls())) {
      if (checkStructField(lhs, rhs, lhs.getNameAttr(), rhs.getNameAttr(),
                           "name") ||
          checkStructField(lhs, rhs, lhs.getType(), rhs.getType(), "type") ||
          checkStructField(lhs, rhs, lhs->getAttrDictionary(),
                           rhs->getAttrDictionary(), "attributes"))
        return failure();
      ++i;
    }

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
