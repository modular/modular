//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ASTType class.
//
//===----------------------------------------------------------------------===//

#include "ASTType.h"
#include "ASTDecl.h"
#include "IRValues.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "LitSharedState.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

ASTDecl *ASTType::getDecl(LitSharedState &shared) const {
  if (auto declRef = dyn_cast<DeclRefType>(mlirType))
    return &shared.declResolver->getDeclForTypeSymbol(declRef.getSymbol());
  return nullptr;
}

/// If this is a parametric user defined type, return all parameter bindings
/// on this reference to the type.  Note that this is potentially a partial
/// binding set - incomplete bindings (missing bindings) are valid.
ParamBindArrayAttr ASTType::getParamBindings() const {
  if (auto declRef = dyn_cast<DeclRefType>(mlirType))
    return declRef.getParamValues();
  return ParamBindArrayAttr::get(mlirType.getContext(), {});
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  return mlirType == other.mlirType;
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType astType) {
  if (!astType)
    return os << "<<NULL ASTTYPE>>";

  auto type = astType.mlirType;
  if (auto declRef = dyn_cast<DeclRefType>(type)) {
    SymbolRefAttr symbol = declRef.getSymbol();
    os << symbol.getRootReference().strref();
    for (FlatSymbolRefAttr nestedRef : symbol.getNestedReferences())
      os << "::" << nestedRef.getValue();

    ParamBindArrayAttr params = declRef.getParamValues();
    if (!params.empty()) {
      os << '[';
      llvm::interleaveComma(params, os, [&](ParamBindAttr bind) {
        printParamValue(bind.getValue(), os);
      });
      os << ']';
    }
  } else if (isa<LIT::NoneType>(type)) {
    os << "None";
  } else {
    os << "__mlir_type." << type;
  }

  return os;
}

std::string ASTType::getAsString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  return os.str();
}

mlir::Diagnostic &M::KGEN::LIT::operator<<(mlir::Diagnostic &diag,
                                           ASTType type) {
  return diag << '\'' << type.getAsString() << '\'';
}

  /// Print to standard error with newline after it, for use in a debugger.
  void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }
