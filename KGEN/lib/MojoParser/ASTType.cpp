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

#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"

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
    mlirType = type.getMlirType();
    return;
  }

  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  assert(LIT::isTypeExpr(typeParamExpr) &&
         "parameter expr must be a type expression");
  mlirType = ParamRefType::get(typeParamExpr);
}

Type ASTType::getMetaType() const {
  if (!mlirType)
    return {};
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return declRef.getMetaType();
  if (auto paramRef = dyn_cast<ParamRefType>(mlirType))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(mlirType))
    return traitRef.getMetaType();
  // This is some generic MLIR type.
  return {};
}

ASTDecl *ASTType::getDecl(SharedState &shared) const {
  Type type = getMetaType();
  if (!type) {
    // FIXME: we currently support references directly to the metatype as a way
    // to look up the decl.  This is pretty weird.
    if (isa_and_nonnull<AnyStructType>(mlirType))
      type = mlirType;
    else
      return nullptr;
  }

  if (auto anyStruct = dyn_cast<AnyStructType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(anyStruct.getSymbol());
  if (auto traitType = dyn_cast<TraitType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(traitType.getSymbol());
  if (auto anyTrait = dyn_cast<AnyTraitType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(
        anyTrait.getTraitType().getSymbol());
  return nullptr;
}

ArrayRef<TypedAttr> ASTType::getParamBindings() const {
  if (AnyStructType metaType = dyn_cast_or_null<AnyStructType>(getMetaType()))
    return metaType.getParamValues();
  return {};
}

/// Return this type with any parameter bindings removed.
ASTType ASTType::getWithoutParameters(SharedState &shared) const {
  if (!mlirType)
    return {};
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return cast<StructDeclOp>(getDecl(shared)).bindReference();
  if (AnyStructType metaType = dyn_cast_or_null<AnyStructType>(mlirType))
    return AnyStructType::get(metaType.getSymbol(), metaType.getSignature());
  // Not parameterized.
  return *this;
}

ArrayRef<TypedAttr> ASTType::getDefaultPosParams() const {
  // Query the metatype for the parameter signature.
  if (AnyStructType metaType = dyn_cast_or_null<AnyStructType>(getMetaType()))
    return metaType.getSignature().getDefaultPosParams();
  return {};
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  if (mlirType == other.mlirType)
    return true;
  // Types with the same metatype are always equal. This is used to detect when
  // two type aliases refer to the same underlying type.
  if (auto meta = dyn_cast_or_null<AnyStructType>(getMetaType()))
    if (meta == other.getMetaType())
      return true;
  return false;
}

/// Return true if this is the same as another ASTType are the same, or if they
/// match when UnknownAttr parameters in the 'this' type are treated as
/// the same as the corresponding parameter in the second type.
///    Foo[1] != Foo[2]   but  Bar[?, 1] == Bar[7, 1]
bool ASTType::isEqualAllowingUnknownAttr(ASTType other,
                                         SharedState &shared) const {
  if (isEqualCanon(other))
    return true;

  // Must have the same struct declarations.
  if (getDecl(shared) != other.getDecl(shared))
    return false;

  ArrayRef<TypedAttr> lhsParams = getParamBindings();
  ArrayRef<TypedAttr> rhsParams = other.getParamBindings();
  assert(lhsParams.size() == rhsParams.size() &&
         "Type with the same decl should have consistent number of params");
  for (auto [lhsParam, rhsParam] : llvm::zip(lhsParams, rhsParams)) {
    if (lhsParam != rhsParam && !isa<UnboundAttr>(lhsParam))
      return false;
  }
  return true;
}

/// Return true if this is a None type.
bool ASTType::isNoneType() const { return isa<KGEN::NoneType>(mlirType); }

/// Return true if this is a TypeCheckError type.
bool ASTType::isTypeCheckErrorType() const {
  return isa<TypeCheckErrorType>(mlirType);
}

/// Return the nonmaterializable decorator target for the type, or null if there
/// is none.
ASTType ASTType::getNonmaterializableTarget(SharedState &shared) const {
  if (auto structOp = dyn_cast_or_null<StructDeclOp>(getDecl(shared)))
    if (TypeAttr targetMlirType = structOp.getNonmaterializableTargetAttr())
      return ASTType(targetMlirType.getValue());
  return {};
}

/// Return whether the specified type is known to be @register_passable; if
/// generic, this returns the 'genericsDefault' value.
static TypeConvention getRegisterPassability(ASTType type, llvm::SMLoc loc,
                                             SharedState &shared,
                                             TypeConvention genericDefault) {
  ASTDecl *decl = type.getDecl(shared);

  if (!decl) {
    // If this is a generic type, use the default specification.
    if (auto paramRefTy = dyn_cast<ParamRefType>(type.mlirType))
      if (isa<ParamRefType, AnyTraitType>(paramRefTy.getParam().getType()))
        return genericDefault;

    // MLIR types are assumed to be register-passable + Trivial.
    return TypeConvention::RegisterPassableTrivial;
  }

  // Make sure we know about the signature of the type.
  if (failed(shared.declResolver->resolveSignature(*decl, loc)))
    return TypeConvention::MemoryOnly;

  // We don't yet have a runtime representation for packages or modules, but
  // when we do, it will not be register-passable.
  if (isa<FileModuleOp, PackageOp>(decl))
    return TypeConvention::MemoryOnly;

  // Trait values are generic and therefore use the default specification.
  if (isa<TraitDeclOp>(decl))
    return genericDefault;

  auto structOp = dyn_cast<StructDeclOp>(decl);
  assert(structOp && "only one user-defined type so far");
  return structOp.getConvention();
}

/// Return the StructDeclOp::RegisterPassable enum for this type.
TypeConvention ASTType::getRegisterPassability(llvm::SMLoc loc,
                                               SharedState &shared) const {
  // If this is a generic type, we treat it as memory only. If the metatype
  // is a parameter reference, then pessimistically assume it is memory-only.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::MemoryOnly);
}

/// Return true if this type is a 'trivial' type, that is one that can be
/// passed around by copying the bits, and whose destructor is a noop.
bool ASTType::isTrivial(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) ==
         TypeConvention::RegisterPassableTrivial;
}

/// Return true if this type is a register-passable type that can be passed
/// around and copied in SSA values instead of having to live in memory.
///
/// The location specifies the location of the reference in case the use is
/// invalid in this location.
bool ASTType::isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) != TypeConvention::MemoryOnly;
}

/// Return true if this type is @register_passable or if it is a generic type
/// that could bind to a concrete @register_passable type.
bool ASTType::mightBeRegisterPassable(llvm::SMLoc loc,
                                      SharedState &shared) const {
  // If this is a generic type, we treat it as register passable conservatively.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::RegisterPassable) !=
         TypeConvention::MemoryOnly;
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

  // Generic types are always destructable.
  if (isa<TraitDeclOp>(decl))
    return true;

  auto structOp = dyn_cast<StructDeclOp>(decl);
  assert(structOp && "only one user-defined type so far");
  return structOp.getDestructorAttr() != TypedAttr();
}

/// Return true if this type is copyable, either because it is trivial or has
/// a copy constructor. Note: this resolves the body of a struct type.
bool ASTType::isCopyable(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are copyable.

  // If the type is trivial, then it is copyable.
  if (auto structDecl = dyn_cast<StructDeclOp>(*typeDecl);
      structDecl && structDecl.isRegisterPassableTrivial())
    return true;

  // Look for a copy constructor.
  if (failed(shared.declResolver->resolveFully(*typeDecl, loc)))
    return true;
  return !typeDecl->lookupInCurrentScope("__copyinit__").empty();
}

/// Return true if this type is movable from its own type, either because it
/// is trivial or has a move constructor from self. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovable(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR types are movable.

  // If the type is register-passable, it is trivially movable.
  if (auto structDecl = dyn_cast<StructDeclOp>(*typeDecl);
      structDecl && structDecl.isRegisterPassable())
    return true;

  // Look for a move constructor.
  if (failed(shared.declResolver->resolveFully(*typeDecl, loc)))
    return true;
  return !typeDecl->lookupInCurrentScope("__moveinit__").empty();
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
  if (!value.ir.getIfRValue())
    return false;

  return shared.typeHasMember(*typeDecl, "__moveinit__", value.expr->getLoc());
}

/// Given a reference, return the element as an ASTType.  This aborts
/// if the current type isn't a reference.
///
ASTType ASTType::getReferenceElementType() const {
  return ASTType(cast<RefType>(mlirType).getElementType());
}

/// Given a VariadicType, return the element as an ASTType.  This aborts if
/// the current type isn't a VariadicType.
ASTType ASTType::getVariadicElementType() const {
  return ASTType(cast<VariadicType>(mlirType).getElementType());
}

/// Return the RefPackType that corresponds to the VariadicPack instance.
RefPackType ASTType::getVariadicPackInfo() const {
  auto bindings = getParamBindings();
  assert(bindings.size() == 4 && bindings[0].getType().isInteger(1) &&
         isa<LifetimeType>(bindings[1].getType()) &&
         isa<AnyTraitType>(bindings[2].getType()) &&
         isa<VariadicType>(bindings[3].getType()) &&
         "Not a VariadicPack struct?");

  return RefPackType::get(
      /*variadicList*/ bindings[3], /*lifetime*/ bindings[1],
      IntegerAttr::get(IndexType::get(bindings[1].getContext()), 0));
}

ASTType ASTType::getKwargsDictValueType() const {
  return cast<TypeConstantAttr>(getParamBindings()[0]).getMlirType();
}

ASTType ASTType::getKwargsDictRefValueType() const {
  return getReferenceElementType().getKwargsDictValueType();
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
ASTType ASTType::getSignatureUserResultType() const {
  auto sigType = cast<SignatureType>(mlirType);
  return LIT::getSignatureUserResultType(sigType, sigType.getArguments(),
                                         sigType.getResults().front());
}

/// Given a SymbolRefAttr, return the underlying symbol name.
static StringRef getNameFromSymbolRef(SymbolRefAttr symbol, bool isFunc) {
  StringAttr leaf;
  if (symbol.getNestedReferences().empty())
    leaf = symbol.getRootReference();
  else
    leaf = symbol.getNestedReferences().back().getAttr();

  // Demangle the function name.
  StringRef name = leaf.getValue();
  if (isFunc)
    if (size_t mangleStart = name.find('('); mangleStart != std::string::npos)
      name = name.take_front(mangleStart);
  return name;
}

// If we are a builtin symbol, then just strip everything but the name of the
// type. E.g. Print ::Int instead of stdlib::builtin::int::Int.
static StringRef trimBuiltinNamespace(StringRef nestedSymbolName) {
  StringRef prettyName(nestedSymbolName);

  if (prettyName.starts_with("stdlib::builtin::")) {
    size_t lastSeperatorLoc = prettyName.rfind("::");
    if (lastSeperatorLoc != StringRef::npos)
      return prettyName.drop_front(lastSeperatorLoc);
  }

  return prettyName;
}

/// Pretty print a symbol reference.
static void printSymbol(raw_ostream &os, StringRef name, SymbolRefAttr symbol) {
  // For constructors, print the type name instead.
  // TODO: Handle other dunder methods.
  if (name == "__init__" && symbol.getNestedReferences().size() >= 2)
    name = symbol.getNestedReferences().drop_back().back().getAttr();
  os << trimBuiltinNamespace(name);
}

static void printSymbol(raw_ostream &os, SymbolRefAttr symbol, bool forDiag,
                        bool isFunc) {
  if (forDiag) {
    printSymbol(os, getNameFromSymbolRef(symbol, isFunc), symbol);
    return;
  }

  std::string nestedSymbolName;
  llvm::raw_string_ostream buff(nestedSymbolName);
  printNestedSymbolReference(buff, symbol);
  os << trimBuiltinNamespace(nestedSymbolName);
}

/// Try to extract a symbol reference from the given parameter. Returns nullptr
/// otherwise.
static SymbolRefAttr tryGetSymbolName(TypedAttr param) {
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param))
    return symbolCst.getSymbol();
  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    switch (op.getOpcode()) {
    case POC::Rebind:
      return tryGetSymbolName(op.getOperands().front());
    default:
      break;
    }
  }
  return {};
}

/// Return a string to use when pretty printing the given kgen dtype.
static std::string getDTypeAsString(KGENDType dtype) {
  // Follow the library spelling for exposed dtypes where they differ.
  switch (dtype.getValue()) {
  case KGENDType::si8:
    return "int8";
  case KGENDType::ui8:
    return "uint8";
  case KGENDType::si16:
    return "int16";
  case KGENDType::ui16:
    return "uint16";
  case KGENDType::si32:
    return "int32";
  case KGENDType::ui32:
    return "uint32";
  case KGENDType::si64:
    return "int64";
  case KGENDType::ui64:
    return "uint64";
  case KGENDType::bf16:
    return "bfloat16";
  case KGENDType::f16:
    return "float16";
  case KGENDType::f32:
    return "float32";
  case KGENDType::tf32:
    return "tensor_float32";
  case KGENDType::f64:
    return "float64";
  default:
    return dtype.getAsString();
  }
}

/// Pretty print a parameter value.
static void printDemangledParam(raw_ostream &os, TypedAttr param,
                                bool forDiag) {
  if (auto structAttr = dyn_cast<LITStructAttr>(param)) {
    // If the struct has a single element, elide the braces.
    if (forDiag && structAttr.getValues().size() == 1) {
      printDemangledParam(os, std::get<1>(structAttr.getValues().front()),
                          forDiag);
    } else {
      os << '{';
      llvm::interleaveComma(structAttr.getValues(), os, [&](auto value) {
        printDemangledParam(os, std::get<1>(value), forDiag);
      });
      os << '}';
    }
    return;
  }
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param)) {
    printSymbol(os, symbolCst.getSymbol(), forDiag, /*isFunc=*/true);
    if (!symbolCst.getParamValues().empty()) {
      os << '[';
      llvm::interleaveComma(
          symbolCst.getParamValues(), os,
          [&](TypedAttr value) { printDemangledParam(os, value, forDiag); });
      os << ']';
    }
    return;
  }
  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    ArrayRef<TypedAttr> operands = op.getOperands();

    // Sugar the parameter operators the parser can generate.
    switch (op.getOpcode()) {
    case POC::Apply:
    case POC::ApplyResultSlot: {
      // Check if we're applying a known symbol, in which case we can do some
      // more specialized printing.
      if (SymbolRefAttr nameAttr = tryGetSymbolName(operands.front())) {
        StringRef name = getNameFromSymbolRef(nameAttr, /*isFunc=*/true);

        // If this is an init and we have a single argument, elide the init.
        if (name == "__init__" && nameAttr.getNestedReferences().size() >= 2) {
          if (operands.size() == 2)
            return printDemangledParam(os, operands.back(), forDiag);
        }
        if (name == "__mlir_i1__" && operands.size() == 2)
          return printDemangledParam(os, operands.back(), forDiag);

        // Otherwise, print the symbol and go through the normal argument list.
        printSymbol(os, name, nameAttr);
      } else {
        printDemangledParam(os, operands.front(), forDiag);
      }

      os << '(';
      llvm::interleaveComma(operands.drop_front(), os, [&](TypedAttr value) {
        printDemangledParam(os, value, forDiag);
      });
      os << ')';
      return;
    }
    case POC::BindSignature:
      printDemangledParam(os, operands.front(), forDiag);
      os << '[';
      llvm::interleaveComma(operands.drop_front(), os, [&](TypedAttr value) {
        printDemangledParam(os, value, forDiag);
      });
      os << ']';
      return;
    case POC::Cond:
      printDemangledParam(os, operands[1], forDiag);
      os << " if ";
      printDemangledParam(os, operands[0], forDiag);
      os << " else ";
      printDemangledParam(os, operands[2], forDiag);
      return;
    case POC::Rebind:
      // Just omit the types.
      printDemangledParam(os, operands.front(), forDiag);
      return;
    case POC::VariadicGet:
      printDemangledParam(os, operands.front(), forDiag);
      os << '[';
      printDemangledParam(os, operands.back(), forDiag);
      os << ']';
      return;
    default:
      break;
    }
  }
  if (auto typeAttr = dyn_cast<TypeConstantAttr>(param)) {
    ASTType(typeAttr.getMlirType()).print(os, forDiag);
    return;
  }
  if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(param)) {
    printDemangledParam(os, extractAttr.getStructValue(), forDiag);
    os << '.' << extractAttr.getField().getValue();
    return;
  }
  if (auto variadicCst = dyn_cast<VariadicAttr>(param)) {
    // VariadicAttr appears in a pack list, so it doesn't need extra []'s around
    // it.
    llvm::interleaveComma(variadicCst.getValues(), os, [&](TypedAttr value) {
      printDemangledParam(os, value, forDiag);
    });
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param)) {
    os << '$' << indexRef.getIndex();
    return;
  }
  if (auto memAttr = dyn_cast<StoreToMemAttr>(param))
    return printDemangledParam(os, memAttr.getValue(), forDiag);
  if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(param)) {
    os << getDTypeAsString(dtypeAttr.getDType());
    return;
  }

  os << getParamAsString(param, forDiag);
}

/// Pretty print a parameter value and optionally demangle it.
/// TODO(16040): Remove this overload when symbol names are name-erased.
void ASTType::printParam(raw_ostream &os, TypedAttr param, bool forDiag,
                         bool demangleParams) {
  if (forDiag || demangleParams)
    param = demangleIfNeeded(param);
  printDemangledParam(os, param, forDiag);
}

void ASTType::print(raw_ostream &os, bool forDiag, bool demangleParams) const {
  // We demangle parameters when printing for diagnostics.
  demangleParams |= forDiag;

  if (!mlirType) {
    os << "<<NULL ASTTYPE>>";
    return;
  }

  Type type = mlirType;
  auto printUserType = [&](SymbolRefAttr symbol, ArrayRef<TypedAttr> params) {
    // Only print the leaf reference when pretty printing types.
    printSymbol(os, symbol, forDiag, /*isFunc=*/false);

    if (params.empty())
      return;

    os << '[';
    llvm::interleaveComma(params, os, [&](TypedAttr value) {
      // If the parameter is a type, print it nicely.
      auto val = PValue(value);

      if (ASTType type = val.getIfTypeValue())
        if (!isa<ParamRefType>(type.mlirType))
          return type.print(os, forDiag);

      printParam(os, val, forDiag, demangleParams);
    });
    os << ']';
  };
  if (auto declRef = dyn_cast<StructType>(type)) {
    printUserType(declRef.getSymbol(), declRef.getParamValues());
  } else if (auto anyStruct = dyn_cast<AnyStructType>(type)) {
    os << "AnyStruct[";
    printUserType(anyStruct.getSymbol(), anyStruct.getParamValues());
    os << ']';
  } else if (auto traitType = dyn_cast<TraitType>(type)) {
    printSymbol(os, traitType.getSymbol(), forDiag, /*isFunc=*/false);
  } else if (auto anyTrait = dyn_cast<AnyTraitType>(type)) {
    os << "AnyTrait[";
    ASTType(anyTrait.getTraitType()).print(os, forDiag, demangleParams);
    os << ']';
  } else if (isNoneType()) {
    os << "None";
  } else if (auto sig = dyn_cast<LITSignatureType>(type)) {
    if (sig.isAsync())
      os << "async ";
    os << "fn";
    if (!sig.getParamTypes().empty() || !sig.getResultParamTypes().empty()) {
      os << '[';
      if (!sig.getParamTypes().empty()) {
        auto printFn = [&](auto p) {
          auto [i, type] = p;
          if (sig.hasParamVarArgs() && i == sig.getNumParams() - 1) {
            os << '*';
            ASTType(cast<VariadicType>(type).getElementType())
                .print(os, forDiag);
          } else {
            ASTType(type).print(os, forDiag);
          }
        };
        llvm::interleaveComma(llvm::enumerate(sig.getParamTypes()), os,
                              printFn);
      } else {
        os << "()";
      }
      assert(sig.getResultParamTypes().empty() &&
             "Mojo doesn't support result parameters");
      os << ']';
    }
    os << '(';
    PassingKindPrinter passingKindPrinter(os, sig.getArgListAttrs());
    bool hadAnyNames = false;
    for (auto [idx, typeX, conventionX] :
         llvm::enumerate(sig.getArguments(), sig.getArgConventions())) {
      ASTType type = typeX;
      ArgConvention convention = conventionX;
      if (SignatureType::isResultSlot(convention))
        continue; // Don't print result in argument list.

      if (idx)
        os << ", ";
      passingKindPrinter.printOptionalStarSlash(idx);

      bool printStar = false;
      if (sig.isPosVarArg(idx)) { // Print with the element of the variadic.
        auto variadic = cast<VariadicType>(type);
        type = variadic.getElementType();
        convention = variadic.getConvention();
        printStar = true;
      }

      auto printConvention = [&]() {
        if (convention == ArgConvention::OwnedInMem ||
            convention == ArgConvention::OwnedInReg)
          os << "owned ";
        else if (convention == ArgConvention::InOut ||
                 convention == ArgConvention::InitSelf)
          os << "inout ";
      };

      // The formal type is VariadicPack[] and the thing to print is a pack
      // attribute, not a type.
      StringAttr name = sig.getArgName(idx);
      hadAnyNames |= !name.empty();
      if (sig.isPackVarArg(idx)) {
        convention = sig.getPackVarArgConvention(idx);
        printConvention();
        os << '*';
        if (!name.empty())
          os << name.getValue() << ": ";
        else
          os << ' ';
        os << '*';

        TypedAttr variadic = ASTType(sig.getIfVariadicPack(idx))
                                 .getVariadicPackInfo()
                                 .getVariadic();
        printParam(os, variadic, forDiag, demangleParams);
      } else {
        printConvention();

        if (printStar)
          os << '*';

        if (!name.empty())
          os << name.getValue() << ": ";

        if (convention == ArgConvention::Ref) {
          os << "ref [";
          auto refType = cast<RefType>(type);
          printParam(os, refType.getLifetime(), forDiag, demangleParams);
          if (!refType.isDefaultAddrSpace()) {
            os << ", ";
            printParam(os, refType.getLifetime(), forDiag, demangleParams);
          }
          os << ']';
        }

        if (SignatureType::hasAddress(convention))
          type = type.getReferenceElementType();
        type.print(os, forDiag);
      }

      // Check if we are at the end; if so, we might still have to print a '/'.
      // If we're pretty printing for a diagnostic, and don't have any names,
      // then we don't print the trailing slash. This makes the extremely
      // common case of a source signature `fn(...) -> ...` look nicer.
      if (!forDiag || hadAnyNames)
        passingKindPrinter.printOptionalTrailingSlash(idx);
    }
    os << ')';
    for (auto [enabled, effect] :
         {std::make_pair(sig.isThrows(), "raises"),
          std::make_pair(sig.isCapturing(), "capturing"),
          std::make_pair(sig.isEscaping(), "escaping")})
      if (enabled)
        os << ' ' << effect;
    os << " -> ";
    Type resultType = ASTType(sig).getSignatureUserResultType();

    if (sig.isRefResult()) {
      auto refType = cast<RefType>(resultType);
      os << "ref [";
      printParam(os, refType.getLifetime(), forDiag, demangleParams);
      os << "] ";
      resultType = refType.getElementType();
    }

    if (isa<KGEN::NoneType>(resultType))
      os << "None";
    else
      ASTType(resultType).print(os, forDiag);
  } else if (auto paramRef = dyn_cast<ParamRefType>(type)) {
    printParam(os, paramRef.getParam(), forDiag, demangleParams);
  } else if (isa<TypeType>(type)) {
    os << "AnyTrivialRegType";
  } else if (auto fnType = dyn_cast<FunctionType>(type)) {
    os << "fn (";
    llvm::interleaveComma(fnType.getInputs(), os,
                          [&](Type type) { ASTType(type).print(os, forDiag); });
    os << ") -> (";
    llvm::interleaveComma(fnType.getResults(), os,
                          [&](Type type) { ASTType(type).print(os, forDiag); });
    os << ")";
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

/// Get the specified parameter as a string.
std::string ASTType::getParamAsString(TypedAttr param, bool forDiag,
                                      bool demangleParams) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printParam(os, param, forDiag, demangleParams);
  return os.str();
}

void PValue::printForDiag(raw_ostream &os) const {
  ASTType::printParam(os, *this, /*forDiag=*/true, /*demangleParams=*/false);
}

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  diag << '\'' << type.getAsString(/*forDiag=*/true) << '\'';
}

void M::addToDiagnostic(TypedAttr paramValue, InflightDiag &diag) {
  diag << '\'';
  diag << ASTType::getParamAsString(paramValue, /*forDiag=*/true,
                                    /*demangleParams=*/true);
  diag << '\'';
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }

RefType ASTType::getRefForArgument(const Twine &argName, bool isMut) {
  auto ctx = mlirType.getContext();
  auto selfLifetime = ParamDeclRefAttr::get(StringAttr::get(ctx, argName + "`"),
                                            LifetimeType::get(ctx, isMut));
  return RefType::get(mlirType, selfLifetime, /*addressSpace=*/0);
}

namespace {
// Class to determine if there are any parameter references in the attribute
// value.
class ParamIndexRefAttrFinder {
public:
  bool hasReferences(TypedAttr value) { return hasReferencesImpl(value, 0); }
  bool hasReferences(Type type) { return hasReferencesImpl(type, 0); }

private:
  template <typename T>
  bool hasReferencesImpl(T value, size_t depth) {
    if (!value)
      return false;

    // If we've already processed this value, just reuse the memoized result.
    auto it = cached.find(value.getAsOpaquePointer());
    if (it != cached.end())
      return it->second;

    // Signatures push a parameter scope.
    if constexpr (std::is_base_of_v<Type, T>)
      if (isa<ParameterScopeTypeInterface>(value))
        ++depth;

    bool hasReference = false;
    // Check to see if this is locally an index with the right depth.
    if constexpr (std::is_base_of_v<Attribute, T>)
      if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value))
        if (indexRef.getDepth() == depth)
          hasReference = true;

    value.walkImmediateSubElements(
        [&](Attribute attr) { hasReference |= hasReferencesImpl(attr, depth); },
        [&](Type type) { hasReference |= hasReferencesImpl(type, depth); });

    cached[value.getAsOpaquePointer()] = hasReference;
    return hasReference;
  }

private:
  // Don't revisit types and attributes multiple times.
  DenseMap<const void *, bool> cached;
};
} // namespace

/// If this type is parameterized, and if any of the parameters refer to a
/// ParamIndexRefAttr, replace it with an UnboundAttr so parameter inference
/// will infer it.
///
/// This makes parameter inference sensitive to what to propagate vs infer. For
/// example, if expectedType is known to be 'SIMD[uint8, 1]', then we can infer
/// which constructor to use when the input is an IntLiteral.
///
/// On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
/// argument is an Int8, then we need the implicit conversion to infer the
/// base element.  Our solution to this is to rip and replace parameters that
/// contain unbound parameters, replacing them with UnboundAttr so inference
/// can find them.
ASTType ASTType::getWithUnknownParametersReplaced(SharedState &shared) const {
  // If this is a struct type, try unbinding just the parameters that have
  // parameter references in it.
  if (auto drt = dyn_cast<StructType>(*this)) {
    ParamIndexRefAttrFinder finder;

    // Otherwise, check each bound parameter to see if it is unknown.  If so,
    // replace it.
    SmallVector<TypedAttr> newParms;
    bool anyBound = false;
    for (auto curValue : getParamBindings()) {
      if (!finder.hasReferences(curValue)) {
        // Keep this value if it has no references.
        anyBound = true;
      } else {
        // Keep the argument type if it has no references.
        Type paramType = curValue.getType();
        if (finder.hasReferences(paramType))
          paramType =
              ASTType(UnboundAttr::get(TypeType::get(paramType.getContext())));
        curValue = UnboundAttr::get(paramType);
      }
      newParms.push_back(curValue);
    }

    if (anyBound)
      return cast<StructDeclOp>(getDecl(shared)).bindReference(newParms);
  }

  // Otherwise return it with all parameters replaced.
  if (Type nonParam = getWithoutParameters(shared))
    return nonParam;
  return *this;
}
