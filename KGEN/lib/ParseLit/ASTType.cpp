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

ASTTypeStorage::ASTTypeStorage(
    ASTDecl &decl, ArrayRef<LitSharedState::ParamBinding> paramValues)
    : decl(decl), paramValues(paramValues) {}

ArrayRef<ASTType::ParamBinding> ASTType::getParamValues() const {
  assert(pointer && "Cannot dereference null ASTType");
  return pointer->paramValues;
}

/// If this is a builtin lit Pointer type, return the element type, otherwise
/// return null.
MValue ASTType::getPointerElementType() const {
  if (isNull())
    return {};

  // Ensure that this is a Pointer type and that its parameters have been bound.
  ASTDecl &decl = getDecl();
  auto params = getParamValues();
  if (decl.magicKind != MagicDeclKind::kPointerType || params.size() != 1)
    return {};
  return params[0].second;
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType type) {
  if (!type)
    return os << "<<NULL ASTTYPE>>";

  os << "'";

  ASTDecl &decl = type.getDecl();
  if (auto typeDecl = dyn_cast<LITStructDeclOp>(decl)) {
    // TODO: Could include name scope information.
    os << typeDecl.getName();
  } else if (decl.isMagic()) {
    switch (decl.magicKind) {
    case MagicDeclKind::kNormal:
      llvm_unreachable("not a magic declaration?");
    case MagicDeclKind::kPointerType:
    case MagicDeclKind::kFunctionType:
      llvm_unreachable("Implemented as a struct, so should be handled");
    case MagicDeclKind::kTypeType:
      os << "type";
      break;
    case MagicDeclKind::kFloatLiteralType:
      os << "FloatLiteralType";
      break;
    case MagicDeclKind::kStringLiteralType:
      os << "StringLiteralType";
      break;
    case MagicDeclKind::kIndexType:
      os << "!builtin.index";
      break;
    case MagicDeclKind::kNoneType:
      os << "!lit.none";
      break;
    case MagicDeclKind::kTypeCheckErrorType:
      os << "<<TypeCheckError>>";
      break;
    }
  } else {
    // TODO: Add "aka" information when we have "type defs".
    os << "<<unknown ASTType>>";
  }

  ArrayRef<LitSharedState::ParamBinding> params = type.getParamValues();
  if (!params.empty()) {
    os << '[';
    llvm::interleaveComma(params, os, [&](LitSharedState::ParamBinding bind) {
      // TODO: This isn't really right, but will work enough for now.
      if (auto attrVal = bind.second.getIfMAValue())
        printParamValue(attrVal.get(), os);
      else
        os << bind.second.getIfMTValue();
    });
    os << ']';
  }

  os << '\'';
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
  return diag << type.getAsString();
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }
