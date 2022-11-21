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
  if (auto declRef = dyn_cast<LITDeclRefType>(type))
    return &shared.getDeclForSymbol(declRef.getSymbol());
  return nullptr;
}

std::vector<ASTType::ParamBinding> ASTType::getParamValues() const {
  assert(type && "Cannot dereference null ASTType");
  auto declRef = dyn_cast<LITDeclRefType>(type);
  if (!declRef)
    return {};

  std::vector<ASTType::ParamBinding> result;
  for (auto bind : declRef.getParamValues()) {
    TypedAttr x = bind.getValue();

    if (auto type = dyn_cast<ConcreteTypeConstantAttr>(x)) {
      result.push_back({bind.getDecl(), MValue(ASTType(type.getValue()))});
    } else if (auto type = dyn_cast<ParameterizedTypeConstantAttr>(x)) {
      result.push_back({bind.getDecl(), MValue(ASTType(type.getValue()))});
    } else {
      result.push_back({bind.getDecl(), MValue(x)});
    }
  }

  return result;
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  return type == other.type;
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType astType) {
  if (!astType)
    return os << "<<NULL ASTTYPE>>";

  auto type = astType.getMLIRType();
  if (auto declRef = dyn_cast<LITDeclRefType>(type)) {
    // TODO: Could include name scope information.
    os << declRef.getSymbol().getRootReference().str();

    ParamBindArrayAttr params = declRef.getParamValues();
    if (!params.empty()) {
      os << '[';
      llvm::interleaveComma(params, os, [&](ParamBindAttr bind) {
        printParamValue(bind.getValue(), os);
      });
      os << ']';
    }
  } else {
    if (isa<KGEN::NoneType>(type))
      os << "None";
    else if (isa<MLIRTypeType>(type))
      os << "type";
    else
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
