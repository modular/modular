//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ASTType class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"

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
  if (auto metaType = dyn_cast<MetaTypeType>(mlirType))
    return &shared.declResolver->getDeclForTypeSymbol(metaType.getSymbol());
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

/// Return true if this is a None type.
bool ASTType::isNoneType() const { return mlirType.isa<LIT::NoneType>(); }

/// Return true if this is a TypeCheckError type.
bool ASTType::isTypeCheckErrorType() const {
  return mlirType.isa<TypeCheckErrorType>();
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

/// Return true if this type is a 'trivial' type, that is one that can be
/// passed around by copying the bits, and whose destructor is a noop.
bool ASTType::isTrivial(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) ==
         StructDeclOp::RP_RegisterPassableTrivial;
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

  return !typeDecl->lookupInCurrentScope("__copyinit__").empty();
}

/// Return true if this type is movable from its own type, either because it
/// is trivial or has a move constructor from self. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovable(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are copyable.

  if (failed(shared.declResolver->resolveFully(*typeDecl, loc)))
    return true;

  // If the type is a register type, it is trivially movable.
  if (cast<StructDeclOp>(*typeDecl).getRegisterPassable() !=
      StructDeclOp::RP_MemoryOnly)
    return true;

  return !typeDecl->lookupInCurrentScope("__moveinit__").empty() ||
         !typeDecl->lookupInCurrentScope("__takeinit__").empty();
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

  // Check all the available candidate to see if we have one that cooperates
  // with this value kind.
  StringRef initName;
  if (value.ir.getIfLValue())
    initName = "__takeinit__";
  else if (value.ir.getIfRValue())
    initName = "__moveinit__";
  else
    return false;

  return shared
      .lookupAndResolveDecl(initName, value.expr->getLoc(), *typeDecl,
                            /*searchParentScopes=*/false)
      .isSuccess();
}

/// Given a PointerType, return the element as an ASTType.  This aborts
/// if the current type isn't a pointer.
ASTType ASTType::getPointerElementType() const {
  return ASTType(cast<PointerType>(mlirType).getElementType());
}

/// Given a reference, return the element as an ASTType.  This aborts
/// if the current type isn't a reference.
///
ASTType ASTType::getReferenceElementType() const {
  /// TODO: This accepts pointer types while we're phasing in first class
  /// references.
  if (auto refType = dyn_cast<RefType>(mlirType))
    return ASTType(refType.getElementType());
  return getPointerElementType();
}

/// Given a VariadicType, return the element as an ASTType.  This aborts if
/// the current type isn't a VariadicType.
ASTType ASTType::getVariadicElementType() const {
  return ASTType(cast<VariadicType>(mlirType).getElementType());
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
ASTType ASTType::getSignatureUserResultType() const {
  auto sigType = cast<SignatureType>(mlirType);
  return LIT::getSignatureUserResultType(sigType, sigType.getValueInputs(),
                                         sigType.getValueResults().front());
}

/// Pretty print a symbol reference.
static void printSymbol(raw_ostream &os, SymbolRefAttr symbol, bool forDiag) {
  if (forDiag) {
    StringAttr leaf;
    if (symbol.getNestedReferences().empty())
      leaf = symbol.getRootReference();
    else
      leaf = symbol.getNestedReferences().back().getAttr();
    // Demangle the function name.
    StringRef name = leaf.getValue();
    if (size_t mangleStart = name.find('('); mangleStart != std::string::npos)
      name = name.take_front(mangleStart);
    // For constructors, print the type name instead.
    // TODO: Handle other dunder methods.
    if (name == "__init__" && symbol.getNestedReferences().size() == 2)
      name = symbol.getNestedReferences().front().getValue();
    os << name;
  } else {
    os << symbol.getRootReference().strref();
    for (FlatSymbolRefAttr nestedRef : symbol.getNestedReferences())
      os << "::" << nestedRef.getValue();
  }
}

/// Pretty print a parameter value.
static void printParam(raw_ostream &os, TypedAttr param, bool forDiag) {
  if (auto structAttr = dyn_cast<StructAttr>(param)) {
    // If the struct has a single element, elide the braces.
    if (forDiag && structAttr.getValues().size() == 1) {
      printParam(os, std::get<1>(structAttr.getValues().front()), forDiag);
    } else {
      os << '{';
      llvm::interleaveComma(structAttr.getValues(), os, [&](auto value) {
        printParam(os, std::get<1>(value), forDiag);
      });
      os << '}';
    }
    return;
  }
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param)) {
    printSymbol(os, symbolCst.getSymbol(), forDiag);
    if (!symbolCst.getParamValues().empty()) {
      os << '[';
      llvm::interleaveComma(
          symbolCst.getParamValues(), os,
          [&](TypedAttr value) { printParam(os, value, forDiag); });
      os << ']';
    }
    return;
  }
  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    // Sugar the parameter operators the parser can generate.
    switch (op.getOpcode()) {
    case POC::Apply:
      printParam(os, op.getOperands().front(), forDiag);
      os << '(';
      llvm::interleaveComma(
          op.getOperands().drop_front(), os,
          [&](TypedAttr value) { printParam(os, value, forDiag); });
      os << ')';
      return;
    case POC::BindSignature:
      printParam(os, op.getOperands().front(), forDiag);
      os << '[';
      llvm::interleaveComma(
          op.getOperands().drop_front(), os,
          [&](TypedAttr value) { printParam(os, value, forDiag); });
      os << ']';
      return;
    case POC::Rebind:
      // Just omit the types.
      printParam(os, op.getOperands().front(), forDiag);
      return;
    case POC::VariadicGet:
      printParam(os, op.getOperands().front(), forDiag);
      os << '[';
      printParam(os, op.getOperands().back(), forDiag);
      os << ']';
      return;
    default:
      break;
    }
  }
  if (auto typeAttr = dyn_cast<TypeConstantAttr>(param)) {
    ASTType(typeAttr.getValue()).print(os, forDiag);
    return;
  }
  if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(param)) {
    printParam(os, extractAttr.getStructValue(), forDiag);
    os << '.' << extractAttr.getField().getValue();
    return;
  }
  if (auto variadicCst = dyn_cast<VariadicAttr>(param)) {
    // VariadicAttr appears in a pack list, so it doesn't need extra []'s around
    // it.
    llvm::interleaveComma(variadicCst.getValues(), os, [&](TypedAttr value) {
      printParam(os, value, forDiag);
    });
    return;
  }

  os << getParamAsString(param);
}

/// Pretty print a parameter value and optionally demangle it.
/// TODO(16040): Remove this overload when symbol names are name-erased.
static void printParam(raw_ostream &os, TypedAttr param, bool forDiag,
                       bool demangleParams) {
  if (forDiag || demangleParams)
    param = demangleIfNeeded(param);
  printParam(os, param, forDiag);
}

void ASTType::print(raw_ostream &os, bool forDiag, bool demangleParams) const {
  // We demangle parameters when printing for diagnostics.
  demangleParams |= forDiag;

  if (!mlirType) {
    os << "<<NULL ASTTYPE>>";
    return;
  }

  Type type = mlirType;
  if (auto declRef = dyn_cast<DeclRefType>(type)) {
    SymbolRefAttr symbol = declRef.getSymbol();
    // Only print the leaf reference when pretty printing types.
    printSymbol(os, symbol, forDiag);

    ParamBindArrayAttr params = declRef.getParamValues();
    if (!params.empty()) {
      os << '[';
      llvm::interleaveComma(params, os, [&](ParamBindAttr bind) {
        // If the parameter is a type, print it nicely.
        auto val = PValue(bind.getValue());

        if (ASTType type = val.getIfTypeValue())
          if (!isa<ParamRefType>(type.mlirType))
            return type.print(os, forDiag);

        printParam(os, val, forDiag, demangleParams);
      });
      os << ']';
    }
  } else if (isNoneType()) {
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
          if (sig.hasParamVarArgs() && i == sig.getNumInputParams() - 1) {
            os << '*';
            ASTType(cast<VariadicType>(type).getElementType())
                .print(os, forDiag);
          } else {
            ASTType(type).print(os, forDiag);
          }
        };
        llvm::interleaveComma(llvm::enumerate(sig.getInputParamTypes()), os,
                              printFn);
      } else {
        os << "()";
      }
      if (!sig.getResultParamTypes().empty()) {
        os << " -> ";
        llvm::interleaveComma(sig.getResultParamTypes(), os, [&](Type type) {
          ASTType(type).print(os, forDiag);
        });
      }
      os << ']';
    }
    os << '(';
    Type inMemResult;
    for (auto [i, type, convention, name] :
         llvm::enumerate(sig.getValueInputs(), sig.getValueInputConventions(),
                         sig.getArgNames())) {
      if (i > (inMemResult ? 1 : 0))
        os << ", ";
      if (convention == ValueInputConvention::ByRefResult) {
        // Print this later.
        inMemResult = type;
        continue;
      }
      if (name.size())
        os << name.getValue() << " = ";
      bool needSpace = false;
      if (convention == ValueInputConvention::OwnedInMem ||
          convention == ValueInputConvention::OwnedInReg) {
        os << "owned";
        needSpace = true;
      } else if (convention == ValueInputConvention::ByRef) {
        os << "inout";
        needSpace = true;
      }

      if (sig.isVarArg(i) || sig.isPackVarArg(i)) {
        if (needSpace) {
          os << ' ';
          needSpace = false;
        }
        os << '*';
        needSpace = sig.isPackVarArg(i);
      }
      if (needSpace)
        os << ' ';
      if (sig.isPackVarArg(i)) {
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
      if (sig.isVarArg(i))
        actualType = cast<VariadicType>(actualType.mlirType).getElementType();
      if (convention != ValueInputConvention::OwnedInReg &&
          convention != ValueInputConvention::BorrowedInReg) {
        actualType = cast<PointerType>(actualType.mlirType).getElementType();
      }
      actualType.print(os, forDiag);
    }
    os << ')';
    for (auto [enabled, effect] :
         {std::make_pair(sig.isThrows(), "raises"),
          std::make_pair(sig.isCapturing(), "capturing")})
      if (enabled)
        os << ' ' << effect;
    os << " -> ";
    Type resultType = sig.getValueResults().front();
    if (inMemResult) {
      ASTType(cast<PointerType>(inMemResult).getElementType())
          .print(os, forDiag);
    } else if (isa<NoneType>(resultType)) {
      os << "None";
    } else if (sig.isThrows()) {
      ASTType(cast<POP::VariantType>(resultType).getTypes().back())
          .print(os, forDiag);
    } else {
      ASTType(resultType).print(os, forDiag);
    }
  } else if (auto paramRef = dyn_cast<ParamRefType>(type)) {
    if (auto indexRef = dyn_cast<ParamIndexRefAttr>(paramRef.getParam()))
      os << '$' << indexRef.getIndex();
    else
      printParam(os, paramRef.getParam(), forDiag, demangleParams);
  } else if (isa<MLIRTypeType>(type)) {
    os << "AnyType";
  } else if (auto fnType = dyn_cast<FunctionType>(type)) {
    os << "fn (";
    llvm::interleaveComma(fnType.getInputs(), os,
                          [&](Type type) { ASTType(type).print(os, forDiag); });
    os << ") -> (";
    llvm::interleaveComma(fnType.getResults(), os,
                          [&](Type type) { ASTType(type).print(os, forDiag); });
    os << ")";
  } else if (auto variantType = dyn_cast<POP::VariantType>(type)) {
    os << "Variant[";
    llvm::interleaveComma(variantType.getParameterizedElementTypes(), os,
                          [&](Type type) { ASTType(type).print(os, forDiag); });
    os << "]";
  } else {
    // Use KGEN pretty printing when printing bare MLIR types for diagnostics.
    if (forDiag)
      printKGENType(os, demangleIfNeeded(type));
    else
      os << "__mlir_type." << (demangleParams ? demangleIfNeeded(type) : type);
  }
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType astType) {
  if (!astType)
    return os << "<<NULL ASTTYPE>>";
  astType.print(os);
  return os;
}

std::string ASTType::getAsString(bool forDiag, bool demangleParams) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, forDiag, demangleParams);
  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(result.begin(), result.end(), '@', '_');
  return os.str();
}

void PValue::printForDiag(raw_ostream &os) const {
  printParam(os, *this, /*forDiag=*/true);
}

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  diag << '\'' << type.getAsString(/*forDiag=*/true) << '\'';
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }
