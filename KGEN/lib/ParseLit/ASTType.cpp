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
#include "LitExprNode.h"
#include "SharedState.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

// Initialize an ASTType from a parameter expression of metatype type.
ASTType::ASTType(TypedAttr typeParamExpr) {
  if (!typeParamExpr) // Null attribute.
    return;

  // Avoid MLIRContext round trip in common case.
  if (auto type = dyn_cast<TypeConstantAttr>(typeParamExpr)) {
    mlirType = type.getValue();
    return;
  }

  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  assert(isa<MLIRTypeType>(typeParamExpr.getType()) &&
         "parameter expr must have metatype type");
  mlirType = ParamRefType::get(typeParamExpr);
}

ASTDecl *ASTType::getDecl(SharedState &shared) const {
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

/// Return the StructDeclOp::RegisterPassable enum for this type.
uint8_t ASTType::getRegisterPassability(llvm::SMLoc loc,
                                        SharedState &shared) const {
  ASTDecl *decl = getDecl(shared);
  if (!decl) // MLIR types are assumed to be register-passable + Trivial.
    return StructDeclOp::RP_RegisterPassableTrivial;

  // Make sure we know about the signature of the type.
  if (failed(shared.declResolver->resolveSignature(*decl, loc)))
    return StructDeclOp::RP_MemoryOnly;

  auto structOp = dyn_cast<StructDeclOp>(*decl);
  assert(structOp && "only one user-defined type so far");
  return structOp.getRegisterPassable();
}

/// Return true if this type is a register-passable type that can be passed
/// around and copied in SSA values instead of having to live in memory.
///
/// The location specifies the location of the reference in case the use is
/// invalid in this location.
bool ASTType::isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) != StructDeclOp::RP_MemoryOnly;
}

/// Return true if this type needs to be destroyed.  This is false for trivial
/// types like Int.  Note: this resolves the body of a struct type.
bool ASTType::hasDestructor(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *decl = getDecl(shared);
  if (!decl) // MLIR types are assumed to be register-passable + Trivial.
    return false;

  // Make sure we know about the signature of the type.
  if (failed(shared.declResolver->resolveFully(*decl, loc)))
    return false;

  auto structOp = dyn_cast<StructDeclOp>(*decl);
  assert(structOp && "only one user-defined type so far");
  return structOp.getDestructorAttr() != TypedAttr();
}

/// Return true if this type is copyable, either because it is trivial or has
/// a copy constructor. Note: this resolves the body of a struct type.
bool ASTType::isCopyable(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are copyable.

  if (failed(shared.declResolver->resolveFully(*typeDecl, loc)))
    return true;

  // If the type is trivial, then it is copyable.
  if (cast<StructDeclOp>(*typeDecl).getRegisterPassable() ==
      StructDeclOp::RP_RegisterPassableTrivial)
    return true;

  auto copyName = StringAttr::get(shared.getContext(), "__copyinit__");
  return typeDecl->lookupInCurrentScope(copyName) != nullptr;
}

/// Return true if this type is movable, either because it is trivial, a
/// register passable type, or has a move constructor. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovableFrom(ASTExprAnd<CValue> value,
                            SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl) // MLIR Types are movable.
    return true;

  SMLoc loc = value.expr->getLoc();
  if (failed(shared.declResolver->resolveFully(*typeDecl, loc)))
    return true;

  // If the type is register passable at all, then it is movable.
  if (isRegisterPassable(loc, shared))
    return true;

  auto moveName = StringAttr::get(shared.getContext(), "__moveinit__");
  const TinyPtrVector<ASTDecl *> *moveDecls =
      typeDecl->lookupInCurrentScope(moveName);
  if (!moveDecls)
    return false;

  // Check all the available candidate to see if we have one that cooperates
  // with this value kind.
  for (ASTDecl *decl : *moveDecls) {
    auto func = dyn_cast<LIT::FuncOp>(*decl);
    if (!func || failed(shared.declResolver->resolveFully(*decl, loc)))
      continue;

    auto signature = func.getSignature();
    if (signature.getValueInputConventions().size() != 2 ||
        signature.getValueInputConventions()[0] !=
            ValueInputConvention::InitSelf)
      continue;
    if (signature.getValueInputConventions()[1] ==
            ValueInputConvention::ByRef &&
        value.ir.getIfLValue())
      return true;
    if (signature.getValueInputConventions()[1] ==
            ValueInputConvention::OwnedInMem &&
        value.ir.getIfRValue())
      return true;
  }
  return false;
}

/// Given a POP::PointerType, return the element as an ASTType.  This aborts
/// if the current type isn't a pointer.
ASTType ASTType::getPointerElementType() const {
  return ASTType(llvm::cast<POP::PointerType>(mlirType).getElementType());
}

/// Given a VariadicType, return the element as an ASTType.  This aborts if
/// the current type isn't a VariadicType.
ASTType ASTType::getVariadicElementType() const {
  return ASTType(cast<VariadicType>(mlirType).getElementType());
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
        // If the parameter is a type, print it nicely.
        auto val = PValue(bind.getValue());

        if (ASTType type = val.getIfTypeValue())
          if (!isa<ParamRefType>(type.mlirType)) {
            os << type;
            return;
          }

        // Otherwise, ask KGEN to do it.  This is gross and needs to be
        // improved.
        os << getParamAsString(val.get());
      });
      os << ']';
    }
  } else if (isa<LIT::NoneType>(type)) {
    os << "None";

  } else if (auto sig = dyn_cast<SignatureType>(type)) {
    if (sig.isAsync())
      os << "async ";
    os << "fn";
    if (!sig.getInputParamTypes().empty() ||
        !sig.getResultParamTypes().empty()) {
      os << '[';
      if (!sig.getInputParamTypes().empty()) {
        auto printFn = [&](auto p) {
          auto [i, type] = p;
          if (bitEnumContainsAny(sig.getFnEffects(), FnEffects::ParamVararg) &&
              i == sig.getInputParamTypes().size() - 1) {
            os << '*';
            os << ASTType(cast<VariadicType>(type).getElementType());
          } else {
            os << ASTType(type);
          }
        };
        llvm::interleaveComma(llvm::enumerate(sig.getInputParamTypes()), os,
                              printFn);
      } else {
        os << "()";
      }
      if (!sig.getResultParamTypes().empty()) {
        os << " -> ";
        llvm::interleaveComma(sig.getResultParamTypes(), os,
                              [&](Type type) { os << ASTType(type); });
      }
      os << ']';
    }
    os << '(';
    Type inMemResult;
    for (auto [i, type, convention] : llvm::enumerate(
             sig.getValueInputs(), sig.getValueInputConventions())) {
      if (i > (inMemResult ? 1 : 0))
        os << ", ";
      if (convention == ValueInputConvention::ByRefResult) {
        // Print this later.
        inMemResult = type;
        continue;
      }
      bool needSpace = false;
      if (convention == ValueInputConvention::OwnedInMem ||
          convention == ValueInputConvention::OwnedInReg) {
        os << "owned";
        needSpace = true;
      }
      if (sig.isVararg(i) || sig.isPackVararg(i)) {
        os << '*';
        needSpace = sig.isPackVararg(i);
      }
      if (convention == ValueInputConvention::ByRef)
        os << '&';
      if (needSpace)
        os << ' ';
      if (sig.isPackVararg(i)) {
        os << '*';
        // This should always be a parameter reference. If not, print the value
        // directly.
        TypedAttr types = cast<POP::PackType>(type).getVariadic();
        if (auto ref = dyn_cast<ParamIndexRefAttr>(types))
          os << '$' << ref.getIndex();
        else
          os << cast<POP::PackType>(type).getVariadic();
        continue;
      }
      ASTType actualType = type;
      if (sig.isVararg(i))
        actualType = cast<VariadicType>(actualType.mlirType).getElementType();
      if (convention != ValueInputConvention::OwnedInReg &&
          convention != ValueInputConvention::BorrowedInReg) {
        actualType =
            cast<POP::PointerType>(actualType.mlirType).getElementType();
      }
      os << actualType;
    }
    os << ')';
    for (auto [enabled, effect] :
         {std::make_pair(sig.isThrows(), "raises"),
          std::make_pair(sig.isCapturing(), "capturing")})
      if (enabled)
        os << ' ' << effect;
    os << " -> ";
    if (inMemResult)
      os << ASTType(cast<POP::PointerType>(inMemResult).getElementType());
    else if (isa<NoneType>(sig.getValueResults().front()))
      os << "None";
    else
      os << ASTType(sig.getValueResults().front());

  } else if (auto paramRef = dyn_cast<ParamRefType>(type)) {
    if (auto indexRef = dyn_cast<ParamIndexRefAttr>(paramRef.getParam()))
      os << '$' << indexRef.getIndex();
    else
      os << getParamAsString(paramRef.getParam());
  } else if (isa<MLIRTypeType>(type)) {
    os << "AnyType";
  } else {
    os << "__mlir_type." << type;
  }

  return os;
}

std::string ASTType::getAsString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(result.begin(), result.end(), '@', '_');
  return os.str();
}

void LIT::addToDiagnostic(ASTType type, LitDiagnostic &diag) {
  diag << '\'' << type.getAsString() << '\'';
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }
