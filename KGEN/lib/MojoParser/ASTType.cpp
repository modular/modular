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
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/IRValues.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"

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

  // ParamType is the canonical way to turn a parameter expression into a type.
  // It handles stripping of metatype information, looks through upcasts etc.
  assert(LIT::isTypeExpr(typeParamExpr) &&
         "parameter expr must be a type expression");
  mlirType = ParamType::get(typeParamExpr);
}

Type ASTType::getMetaType() const {
  if (!mlirType)
    return {};
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return StructMetaType::get(LIT::StructType::get(
        declRef.getSymbol(), declRef.getParamValues(), declRef.getSignature()));
  if (auto paramRef = dyn_cast<ParamType>(mlirType))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(mlirType))
    return traitRef.getMetaType();
  // This is some generic MLIR type.
  return {};
}

/// If this is a user declared type, return the declaration that this came
/// from.  If this is a raw MLIR type or a metatype, return null.
ASTDecl *ASTType::getDecl(SharedState &shared) const {
  // We get the declaration from the metatype of the type.  For example, if we
  // have a parametric type like "T" where "T: AnyType", we can know that T has
  // AnyType bound.
  Type type = getMetaType();
  if (!type)
    return nullptr;

  // If our metatype is itself parametric, for example, we have something like:
  //     !kgen.param<:!lit.anytrait<<@Movable>> elt_trait>
  // Then this type conforms to some parametric trait that is bound by at least
  // Movable.  Use Movable as the declaration we're working with.
  if (auto paramRef = dyn_cast<ParamType>(type)) {
    // AnyTrait is the only metatype of a metatype.
    type = cast<AnyTraitType>(paramRef.getParam().getType());
  }

  if (auto anyStruct = dyn_cast<StructMetaType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(anyStruct.getSymbol());

  if (auto anyTrait = dyn_cast<AnyTraitType>(type))
    type = anyTrait.getTraitType();

  if (auto traitType = dyn_cast<TraitType>(type))
    return shared.declResolver->getTraitDecl(traitType);

  return nullptr;
}

ArrayRef<TypedAttr> ASTType::getParamBindings() const {
  if (StructMetaType metaType = dyn_cast_or_null<StructMetaType>(getMetaType()))
    return metaType.getParamValues();
  return {};
}

/// Return this type with any parameter bindings removed.
ASTType ASTType::getWithoutParameters(SharedState &shared) const {
  if (!mlirType)
    return {};
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return cast<StructDeclOp>(getDecl(shared)).bindReference();
  if (StructMetaType metaType = dyn_cast_or_null<StructMetaType>(mlirType))
    return StructMetaType::get(
        LIT::StructType::get(metaType.getSymbol(), metaType.getSignature()));
  // Not parameterized.
  return *this;
}

ArrayRef<TypedAttr> ASTType::getDefaultPosParams() const {
  // Query the metatype for the parameter signature.
  if (StructMetaType metaType = dyn_cast_or_null<StructMetaType>(getMetaType()))
    return metaType.getSignature().getDefaultPosParams();
  return {};
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  if (mlirType == other.mlirType)
    return true;
  // Types with the same metatype are always equal. This is used to detect when
  // two type aliases refer to the same underlying type.
  if (auto meta = dyn_cast_or_null<StructMetaType>(getMetaType()))
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
    if (auto paramRefTy = dyn_cast<ParamType>(type.mlirType))
      if (isa<ParamType, AnyTraitType>(paramRefTy.getParam().getType()))
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
  if (auto trait = dyn_cast<TraitDeclOp>(decl)) {
    TypeConvention convention = trait.getConvention();
    if (convention == TypeConvention::Unspecified)
      return genericDefault;
    return convention;
  }

  if (TraitType traitType =
          dyn_cast_or_null<TraitType>(decl->getIfTypeValue())) {
    // The register passability of a trait composition is the strictest of its
    // members.
    TypeConvention convention = TypeConvention::Unspecified;
    for (SymbolRefAttr symbol : traitType.getSymbols()) {
      TypeConvention singleTraitRP =
          ASTType(TraitType::get(symbol)).getRegisterPassability(loc, shared);
      if (singleTraitRP == TypeConvention::Unspecified)
        continue;
      if (convention == TypeConvention::Unspecified)
        convention = singleTraitRP;
      else
        convention = std::max(convention, singleTraitRP);
    }
    if (convention == TypeConvention::Unspecified)
      return genericDefault;
    return convention;
  }

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
  TypeConvention convention = getRegisterPassability(loc, shared);
  return convention == TypeConvention::RegisterPassable ||
         convention == TypeConvention::RegisterPassableTrivial;
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
  if (failed(shared.declResolver->resolveBody(*decl, loc)))
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
  if (failed(shared.declResolver->resolveBody(*typeDecl, loc)))
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
  if (failed(shared.declResolver->resolveBody(*typeDecl, loc)))
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
  if (failed(shared.declResolver->resolveBody(*typeDecl, loc)))
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

/// Given a VariadicType, return the argument convention.  This aborts if
/// the current type isn't a VariadicType.
ArgConvention ASTType::getVariadicConvention() const {
  return cast<VariadicType>(mlirType).getConvention();
}

/// Return the RefPackType that corresponds to the VariadicPack instance.
RefPackType ASTType::getVariadicPackInfo(SharedState &shared) const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  // NOTE: `bindings[0]` and `bindings[1]` are expected to be the Mojo `Bool`
  // type, and `bindings[2]` is an Origin.
  assert(bindings.size() == 5 && isa<LIT::StructType>(bindings[0].getType()) &&
         isa<LIT::StructType>(bindings[1].getType()) &&
         isa<LIT::StructType>(bindings[2].getType()) &&
         isa<AnyTraitType>(bindings[3].getType()) &&
         isa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");

  TypedAttr origin = ASTType::extractOriginOf(SMLoc(), bindings[2], shared);
  return RefPackType::get(
      /*variadicList*/ bindings[4], origin,
      /*addrSpace*/
      IntegerAttr::get(IndexType::get(shared.getContext()), 0));
}

/// Return the type list for the variadic argument in a VariadicPack.  This
/// will be a VariadicAttr when concrete (e.g. on the caller side) or a
/// parameter on the callee side.
TypedAttr ASTType::getVariadicPackTypeList() const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  // NOTE: `bindings[0]` and `bindings[1]` are expected to be the Mojo `Bool`
  // type, and `bindings[2]` is an Origin.
  assert(bindings.size() == 5 && isa<LIT::StructType>(bindings[0].getType()) &&
         isa<LIT::StructType>(bindings[1].getType()) &&
         isa<LIT::StructType>(bindings[2].getType()) &&
         isa<AnyTraitType>(bindings[3].getType()) &&
         isa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");
  return bindings[4];
}

ASTType ASTType::getKwargsDictValueType() const {
  return ASTType(getParamBindings()[0]);
}

ASTType ASTType::getKwargsDictRefValueType() const {
  return getReferenceElementType().getKwargsDictValueType();
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
ASTType ASTType::getSignatureUserResultType() const {
  auto sigGenType = cast<FnTypeGeneratorType>(mlirType);
  return LIT::getSignatureUserResultType(sigGenType, sigGenType.getArguments(),
                                         sigGenType.getResults().front());
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

// Get the typename of the symbol
static StringRef tryGetTypeNameFromSymbolRef(SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().size() >= 2)
    return symbol.getNestedReferences().drop_back().back().getAttr();
  return {};
}

// If we are a builtin symbol, then just strip everything but the name of the
// type. E.g. Print ::Int instead of stdlib::builtin::int::Int.
static StringRef trimBuiltinNamespace(StringRef nestedSymbolName) {
  // List of common namespace prefixes to trim
  static const StringRef commonPrefixes[] = {
      "stdlib::", "layout::",
      // Add other common prefixes here
  };

  StringRef prettyName(nestedSymbolName);
  for (StringRef prefix : commonPrefixes) {
    if (prettyName.starts_with(prefix)) {
      const size_t lastSeparatorLoc = prettyName.rfind("::");
      if (lastSeparatorLoc != StringRef::npos)
        return prettyName.substr(lastSeparatorLoc);
    }
  }

  return prettyName;
}

static void printSymbol(raw_ostream &os, SymbolRefAttr symbol,
                        SharedState *diagShared, bool isFunc) {
  const bool forDiag = diagShared != nullptr;
  if (forDiag) {
    StringRef name = getNameFromSymbolRef(symbol, isFunc);
    // For constructors, print the type name instead.
    // TODO: Handle other dunder methods.
    if (name == "__init__" && symbol.getNestedReferences().size() >= 2)
      name = symbol.getNestedReferences().drop_back().back().getAttr();

    // Disable printing all parameters for the user defined `Index` functions.
    // For example, 'Index[Intable, Intable](16,16)' -> 'Index(16,16)'
    if (getNameFromSymbolRef(symbol, true).starts_with("Index")) {
      os << "Index";
      return;
    }

    os << trimBuiltinNamespace(name);
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
  param = ParamOperatorAttr::stripRebind(param);
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param))
    return symbolCst.getSymbol();
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
  case KGENDType::tf32:
    return "tensor_float32";
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, ...)                              \
  case KGENDType::SHORT_NAME:                                                  \
    return #LONG_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return dtype.getAsString();
  }
}
static bool isKnownNonStaticMethod(SharedState *diagShared,
                                   SymbolRefAttr callee) {
  if (!diagShared) // Need SharedState to figure this out.
    return false;
  // Must be able to figure out the decl in question, and must be a method.
  ASTDecl *decl = diagShared->getDeclResolver().getDeclForFuncSymbol(callee);
  if (!decl || !decl->tryGetMethodParentDecl())
    return false;

  // Return false for static methods.
  return !cast<FnOp>(*decl).getIsStatic();
}

/// Pretty print a parameter value.
static void printDemangledParam(raw_ostream &os, TypedAttr param,
                                SharedState *diagShared) {
  auto printOperands =
      [&](ArrayRef<TypedAttr> operands, StringRef separator = ", ",
          StringRef lSeparator = "(", StringRef rSeparator = ")") -> void {
    os << lSeparator;
    llvm::interleave(
        operands, os,
        [&](TypedAttr value) {
          // Don't print extracts out of Int.value.
          if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
            value = extract.getStructValue();
          printDemangledParam(os, value, diagShared);
        },
        separator);
    os << rSeparator;
  };

  if (auto bindParams = dyn_cast<BindParamsAttr>(param)) {
    printDemangledParam(os, bindParams.getGenerator(), diagShared);
    printOperands(bindParams.getParamValues(), ", ", "[", "]");
    return;
  }
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param)) {
    printSymbol(os, symbolCst.getSymbol(), diagShared, /*isFunc=*/true);
    if (!symbolCst.getParamValues().empty())
      printOperands(symbolCst.getParamValues(), ", ", "[", "]");
    return;
  }
  if (auto refPack = dyn_cast<RefPackAttr>(param)) {
    llvm::interleaveComma(refPack.getValues(), os, [&](TypedAttr value) {
      printDemangledParam(os, value, diagShared);
    });
    return;
  }

  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    ArrayRef<TypedAttr> operands = op.getOperands();

    // Sugar the parameter operators the parser can generate.
    switch (op.getOpcode()) {
    case POC::Apply:
    case POC::ApplyResultSlot: {
      ArrayRef<TypedAttr> operandsToPrint = operands.drop_front();

      // Check if we're applying a known symbol, in which case we can do some
      // more specialized printing.
      if (SymbolRefAttr nameAttr = tryGetSymbolName(operands.front())) {
        StringRef name = getNameFromSymbolRef(nameAttr, /*isFunc=*/true);
        // Don't print conversions of boolean's to i1.
        if (name == "__mlir_i1__" && operands.size() == 2)
          return printDemangledParam(os, operands.back(), diagShared);

        // Print arithmetic functions using their mathematical form rather than
        // as dunder method calls.
        static SmallDenseMap<StringRef, StringRef> binaryOpNames{
            {"__add__", " + "},     {"__sub__", " - "},
            {"__mul__", " * "},     {"__mod__", " % "},
            {"__truediv__", " / "}, {"__floordiv__", " // "},
            {"__xor__", " ^ "},     {"__and__", " & "},
            {"__or__", " | "},      {"__lshift__", " << "},
            {"__rshift__", " >> "}, {"__eq__", " == "},
            {"__lt__", " < "},      {"__le__", " <= "},
            {"__in__", " in "},     {"__ne__", " != "},
            {"__gt__", " > "},      {"__ge__", " >= "},
            {"__matmul__", " @ "},  {"__pow__", " ** "},
            {"__is__", " is "},     {"__isnot__", " isnot "},
        };
        if (auto it = binaryOpNames.find(name); it != binaryOpNames.end())
          return printOperands(operandsToPrint, /*separator=*/it->second);

        // If we can tell that this is a method call, print the receiver first.
        if (!operandsToPrint.empty() &&
            isKnownNonStaticMethod(diagShared, nameAttr)) {
          printDemangledParam(os, operandsToPrint.front(), diagShared);
          os << '.';
          operandsToPrint = operandsToPrint.drop_front();
        }

        // Otherwise, print the symbol and go through the normal argument list.
        printSymbol(os, nameAttr, diagShared, /*isFunc=*/true);

        // Omit the last 'VariadicPack' operand as its the 'is_owned' bit.
        if (diagShared &&
            tryGetTypeNameFromSymbolRef(nameAttr) == "VariadicPack" &&
            operandsToPrint.size() > 1) {
          operandsToPrint = operandsToPrint.drop_back();
        }

      } else {
        printDemangledParam(os, operands.front(), diagShared);
      }

      return printOperands(operandsToPrint);
    }
    case POC::Cond: {
      printDemangledParam(os, operands[1], diagShared);
      os << " if ";
      auto cond = operands[0];
      // Don't print extracts of Bool.value.
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(cond))
        cond = extract.getStructValue();
      printDemangledParam(os, cond, diagShared);
      os << " else ";
      printDemangledParam(os, operands[2], diagShared);
      return;
    }
    case POC::Rebind:
      // Just omit the types.
      printDemangledParam(os, operands.front(), diagShared);
      return;
    case POC::VariadicGet:
      printDemangledParam(os, operands.front(), diagShared);
      os << '[';
      printDemangledParam(os, operands.back(), diagShared);
      os << ']';
      return;
    default:
      const char *binOp = nullptr;
      switch (op.getOpcode()) {
      case POC::Add:
        binOp = " + ";
        break;
      case POC::Mul:
      case POC::MulNuw:
        binOp = " * ";
        break;
      case POC::Div:
        binOp = " / ";
        break;
      case POC::Mod:
        binOp = " % ";
        break;
      case POC::And:
        binOp = " & ";
        break;
      case POC::Or:
        binOp = " | ";
        break;
      case POC::Xor:
        binOp = " ^ ";
        break;
      case POC::Shl:
        binOp = " << ";
        break;
      case POC::Shr:
        binOp = " >> ";
        break;
      case POC::EQ:
        binOp = " == ";
        break;
      case POC::LT:
        binOp = " < ";
        break;
      case POC::LE:
        binOp = " <= ";
        break;
      case POC::In:
        binOp = " in ";
        break;
      default:
        break;
      }
      // Simple things that show up in integer param expressions.
      if (binOp)
        return printOperands(operands, /*separator=*/binOp);

      break;
    }
  }
  if (auto typeAttr = dyn_cast<TypeParamAttr>(param)) {
    ASTType(typeAttr.getMlirType()).print(os, diagShared);
    return;
  }
  if (auto upcast = dyn_cast<UpcastAttr>(param))
    return printDemangledParam(os, upcast.getInputTypeValue(), diagShared);

  if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(param)) {
    printDemangledParam(os, extractAttr.getStructValue(), diagShared);
    os << '.' << extractAttr.getField().getValue();
    return;
  }
  if (auto variadicCst = dyn_cast<VariadicAttr>(param)) {
    // VariadicAttr appears in a pack list, so it doesn't need extra []'s around
    // it.
    llvm::interleaveComma(variadicCst.getValues(), os, [&](TypedAttr value) {
      printDemangledParam(os, value, diagShared);
    });
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param)) {
    os << '$';
    if (size_t depth = indexRef.getDepth())
      os << depth << '|';
    os << indexRef.getIndex();
    return;
  }
  if (auto memAttr = dyn_cast<StoreToMemAttr>(param))
    return printDemangledParam(os, memAttr.getValue(), diagShared);

  if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(param)) {
    os << getDTypeAsString(dtypeAttr.getDType());
    return;
  }

  if (auto originField = dyn_cast<OriginFieldAttr>(param)) {
    if (isa<StaticOriginAttr>(originField.getBase())) {
      if (originField.getField().str() == "__constants__" &&
          originField.getType().isMutableKnown(false)) {
        os << "StaticConstantOrigin";
        return;
      }
    }

    printDemangledParam(os, originField.getBase(), diagShared);
    os << '.' << originField.getField().str();
    return;
  }
  if (auto originUnion = dyn_cast<OriginUnionAttr>(param)) {
    os << '{';
    llvm::interleaveComma(originUnion.getOperands(), os, [&](TypedAttr param) {
      printDemangledParam(os, param, diagShared);
    });
    os << '}';
    return;
  }

  if (auto indirect = dyn_cast<IndirectOriginAttr>(param)) {
    printDemangledParam(os, indirect.getBase(), diagShared);
    os << "[]";
    return;
  }

  if (auto mutcast = dyn_cast<OriginMutCastAttr>(param)) {
    if (mutcast.getType().isMutableKnown(false))
      os << "(muttoimm ";
    else
      os << "(mutcast ";
    printDemangledParam(os, mutcast.getOperand(), diagShared);
    os << ")";
    return;
  }

  if (auto anyLife = dyn_cast<AnyOriginAttr>(param)) {
    if (anyLife.getType().isMutableKnown(true))
      os << "MutableAnyOrigin";
    else if (anyLife.getType().isMutableKnown(false))
      os << "ImmutableAnyOrigin";
    else
      os << "SomeAnyOrigin";
    return;
  }

  // Special case bool constants instead of printing as 0/1.
  if (auto boolAttr = dyn_cast<BoolAttr>(param)) {
    os << (boolAttr.getValue() ? "True" : "False");
    return;
  }

  if (auto noneAttr = dyn_cast<NoneAttr>(param)) {
    os << "None";
    return;
  }

  if (auto strAttr = dyn_cast<StringAttr>(param)) {
    os << '"';
    printAsMojoStringLiteral(strAttr, os);
    os << '"';
    return;
  }
  /// A StructAttr is due to an inline @always_inline("builtin") initializer.
  /// Elide it if we have the default type with a literal so we don't print
  /// Int(42), but print it if it is something weird like IntLiteral(42)
  if (auto structAttr = dyn_cast<LITStructAttr>(param)) {
    // If the struct has a single element, elide the braces.
    if (diagShared && structAttr.getValues().size() == 1) {
      ASTDecl *decl = ASTType(structAttr.getType()).getDecl(*diagShared);
      StringRef typeName;
      if (decl && isa<LIT::StructDeclOp>(*decl))
        typeName = cast<LIT::StructDeclOp>(*decl).getDeclName().strref();
      TypedAttr elt = std::get<1>(structAttr.getValues().front());
      if (typeName == "Int" || typeName == "Bool" || typeName == "Origin" ||
          typeName == "DType") {
        if (auto extract = dyn_cast<LIT::StructExtractAttr>(elt))
          elt = extract.getStructValue();
        printDemangledParam(os, elt, diagShared);
        return;
      }
    }

    ASTType(structAttr.getType()).print(os, diagShared);
    os << '(';
    // TODO: Could print keywords for the labels if there is a reason someday.
    llvm::interleaveComma(structAttr.getValues(), os, [&](auto elt) {
      TypedAttr value = std::get<1>(elt);
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
        value = extract.getStructValue();
      printDemangledParam(os, value, diagShared);
    });
    os << ')';
    return;
  }

  if (auto convert = dyn_cast<POP::IntLiteralConvertAttr>(param)) {
    printDemangledParam(os, convert.getInput(), diagShared);
    return;
  }

  if (auto intLitBin = dyn_cast<POP::IntLiteralBinAttr>(param)) {
    const char *binOp = nullptr;
    switch (intLitBin.getOper().getValue()) {
    case POP::IntLiteralBinKind::Add:
      binOp = " + ";
      break;
    case POP::IntLiteralBinKind::Sub:
      binOp = " - ";
      break;
    case POP::IntLiteralBinKind::Mul:
      binOp = " * ";
      break;
    case POP::IntLiteralBinKind::FloorDiv:
      binOp = " // ";
      break;
    case POP::IntLiteralBinKind::Mod:
      binOp = " % ";
      break;
    case POP::IntLiteralBinKind::Lshift:
      binOp = " << ";
      break;
    case POP::IntLiteralBinKind::Rshift:
      binOp = " >> ";
      break;
    case POP::IntLiteralBinKind::And:
      binOp = " & ";
      break;
    case POP::IntLiteralBinKind::Or:
      binOp = " | ";
      break;
    case POP::IntLiteralBinKind::Xor:
      binOp = " ^ ";
      break;
    }

    return printOperands({intLitBin.getLhs(), intLitBin.getRhs()},
                         /*separator=*/binOp);
  }

  if (auto fpLit = dyn_cast<POP::FloatLiteralAttr>(param)) {
    switch (fpLit.getSpecial().getValue()) {
    case POP::FloatLiteralSpecialValues::NegZero:
      os << "-0.0";
      return;
    case POP::FloatLiteralSpecialValues::Inf:
      os << "inf";
      return;
    case POP::FloatLiteralSpecialValues::NegInf:
      os << "-inf";
      return;
    case POP::FloatLiteralSpecialValues::Nan:
      os << "nan";
      return;
    case POP::FloatLiteralSpecialValues::Normal:
      // Convert to f64 to print out the value.
      auto ctx = fpLit.getContext();
      auto f64Type = POP::SIMDType::get(ctx, 1, DType::f64);
      auto simdVal = cast<POP::SIMDAttr>(
          POP::FloatLiteralConvertAttr::get(ctx, f64Type, fpLit));
      os << simdVal.getValues()[0].getFloatVal();
      return;
    }
  }

  // IntLiteral/FloatLiteral/StringLiteral are stateless values that end up as
  // UnknownAttr.
  if (isa<UnknownAttr>(param) && diagShared) {
    ASTDecl *decl = ASTType(param.getType()).getDecl(*diagShared);
    StringRef typeName;
    if (decl && isa<LIT::StructDeclOp>(*decl))
      typeName = cast<LIT::StructDeclOp>(*decl).getDeclName().strref();
    if (typeName == "IntLiteral" || typeName == "FloatLiteral" ||
        typeName == "StringLiteral") {
      auto structType = cast<LIT::StructType>(param.getType());
      if (structType.getParamValues().size() == 1) {
        printDemangledParam(os, structType.getParamValues()[0], diagShared);
        return;
      }
    }
  }

  os << getParamAsString(param, diagShared);
}

/// Pretty print a parameter value and optionally demangle it.
/// TODO(16040): Remove this overload when symbol names are name-erased.
void ASTType::printParam(raw_ostream &os, TypedAttr param,
                         SharedState *diagShared, bool demangleParams) {
  if (diagShared || demangleParams)
    param = demangleIfNeeded(param);
  printDemangledParam(os, param, diagShared);
}

void ASTType::print(raw_ostream &os, SharedState *diagShared,
                    bool demangleParams) const {
  // We demangle parameters when printing for diagnostics.
  demangleParams |= (diagShared != nullptr);

  if (!mlirType) {
    os << "<<NULL ASTTYPE>>";
    return;
  }

  Type type = mlirType;
  auto printUserType = [&](SymbolRefAttr symbol, ArrayRef<TypedAttr> params,
                           ASTDecl *typeDecl) {
    // Handle special cases that should be aliased.
    // FIXME(MOCO-367): maintain "typedef" sugar in the type system.
    if (typeDecl && isa<LIT::StructDeclOp>(*typeDecl) && params.size() == 1 &&
        cast<LIT::StructDeclOp>(*typeDecl).getDeclName().strref() == "Origin") {
      // Check to see if we have a Bool with a known constant parameter.
      //   #lit.struct<{value: i1 = 1}>
      if (auto strParam = dyn_cast<LITStructAttr>(params[0])) {
        if (strParam.getValues().size() == 1) {
          if (auto value =
                  dyn_cast<BoolAttr>(std::get<1>(strParam.getValues()[0]))) {
            os << (value.getValue() ? "MutableOrigin" : "ImmutableOrigin");
            return;
          }
        }
      }
    }

    // Only print the leaf reference when pretty printing types.
    printSymbol(os, symbol, diagShared, /*isFunc=*/false);
    if (params.empty())
      return;

    SmallVector<std::pair<StringAttr, TypedAttr>> paramsToPrint;

    // If we're printing for diagnostics, we'll have a 'typeDecl' corresponding
    // to this.  In that case we want to avoid printing defaulted parameter
    // values that are the same as their default value.
    if (typeDecl) {
      TypeSignatureType origSig = cast<StructDeclOp>(*typeDecl).getSignature();
      PogListAttr paramInfo = origSig.getParamListAttrs();
      assert(paramInfo.size() == params.size() &&
             "Unexpected number of bound params");

      ParameterEvaluator evaluator(params);

      // Find out about default parameter values.
      DefaultValueHandler defaultValueHandler(paramInfo);
      bool skippedPositional = false;
      for (auto [idx, pog, paramValue] :
           llvm::enumerate(paramInfo.getPogs(), params)) {

        auto passingKind = pog.getPassingKind();

        // See if this parameter has a default value.  If so, and if the
        // provided value matches it, then don't print the parameter in the
        // list.
        if (auto def = defaultValueHandler.getDefault(idx)) {
          // Make sure to substitute other parameter values in, e.g. so we can
          // handle things like:
          //   struct UnsafePointer[type: AnyType,
          //                        align: Int = _default_alignment[type]()]:
          def = evaluator.getReboundAttribute(def);
          if (paramValue == def && passingKind != PassingKind::PosOnly) {
            // If we skip a posOrKw then include keyword names for any other
            // posOrKw's that come after it.
            skippedPositional |= (passingKind == PassingKind::PosOrKw);
            continue;
          }
        }

        StringAttr name;
        switch (passingKind) {
        case PassingKind::Implicit:
        case PassingKind::Inferred:
          continue; // Don't print implicit parameters at all.
        case PassingKind::PosOnly:
          break; // Never include a name.
        case PassingKind::PosOrKw:
          if (!skippedPositional)
            break; // Don't include a name unless we skipped another one.
          [[fallthrough]];
        case PassingKind::KwOnly:
          name = paramInfo.getName(idx);
          break;
        }
        paramsToPrint.push_back({name, paramValue});
      }

    } else {
      // When generating mangled names, don't include names for parameters since
      // positional information is enough.
      for (TypedAttr paramValue : params)
        paramsToPrint.push_back({StringAttr(), paramValue});
    }

    if (!paramsToPrint.empty()) {
      os << '[';
      llvm::interleaveComma(
          paramsToPrint, os, [&](std::pair<StringAttr, TypedAttr> param) {
            if (param.first)
              os << param.first.strref() << '=';
            printParam(os, param.second, diagShared, demangleParams);
          });
      os << ']';
    }
  };

  auto printConvention = [&os](ArgConvention conv) {
    if (conv == ArgConvention::OwnedMem)
      os << "owned ";
    else if (conv == ArgConvention::Mut)
      os << "mut ";
    else if (conv == ArgConvention::ByRefResult)
      os << "out ";
  };

  auto printRef = [&](RefType refType) {
    os << "ref [";
    printParam(os, refType.getOrigin(), diagShared, demangleParams);
    if (!refType.isDefaultAddrSpace()) {
      os << ", ";
      printParam(os, refType.getOrigin(), diagShared, demangleParams);
    }
    os << "] ";
  };

  if (auto structTy = dyn_cast<StructType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(type).getDecl(*diagShared);
    printUserType(structTy.getSymbol(), structTy.getParamValues(), decl);
  } else if (auto anyStruct = dyn_cast<StructMetaType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(anyStruct.getType()).getDecl(*diagShared);
    os << "AnyStruct[";
    printUserType(anyStruct.getSymbol(), anyStruct.getParamValues(), decl);
    os << ']';
  } else if (auto traitType = dyn_cast<TraitType>(type)) {
    llvm::interleave(
        traitType.getSymbols(), os,
        [&](SymbolRefAttr symbol) {
          printSymbol(os, symbol, diagShared, /*isFunc=*/false);
        },
        " & ");
  } else if (auto anyTrait = dyn_cast<AnyTraitType>(type)) {
    os << "AnyTrait[";
    ASTType(anyTrait.getTraitType()).print(os, diagShared, demangleParams);
    os << ']';
  } else if (isNoneType()) {
    os << "None";
  } else if (auto ref = dyn_cast<RefType>(type)) {
    printRef(ref);
    ASTType(ref.getElementType()).print(os, diagShared, demangleParams);
  } else if (auto variadic = dyn_cast<VariadicType>(type)) {
    os << "Variadic[";
    printConvention(variadic.getConvention());
    ASTType(variadic.getElementType()).print(os, diagShared, demangleParams);
    os << "]";
  } else if (auto sig = dyn_cast<FnTypeGeneratorType>(type)) {
    if (sig.isAsync())
      os << "async ";
    os << "fn";
    if (!sig.getInputParamTypes().empty()) {
      os << '[';
      if (!sig.getInputParamTypes().empty()) {
        auto printFn = [&](auto p) {
          auto [i, type] = p;
          if (sig.getParamListAttrs().isVariadic(i)) {
            os << '*';
            ASTType(cast<VariadicType>(type).getElementType())
                .print(os, diagShared, demangleParams);
          } else {
            ASTType(type).print(os, diagShared, demangleParams);
          }
        };
        llvm::interleaveComma(llvm::enumerate(sig.getInputParamTypes()), os,
                              printFn);
      } else {
        os << "()";
      }
      os << ']';
    }
    os << '(';
    PassingKindPrinter passingKindPrinter(os, sig.getArgListAttrs());
    bool hadAnyNames = false;
    for (auto [idx, typeX, conventionX] :
         llvm::enumerate(sig.getArguments(), sig.getArgConventions())) {
      ASTType type = typeX;
      ArgConvention convention = conventionX;
      if (isResultSlot(convention))
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

      // The formal type is VariadicPack[] and the thing to print is a pack
      // attribute, not a type.
      StringAttr name = sig.getArgName(idx);
      hadAnyNames |= !name.empty();
      if (sig.isPackVarArg(idx)) {
        convention = sig.getPackVarArgConvention(idx);
        printConvention(convention);
        os << '*';
        if (!name.empty())
          os << name.getValue() << ": ";
        else
          os << ' ';
        os << '*';

        TypedAttr variadic =
            ASTType(sig.getIfVariadicPack(idx)).getVariadicPackTypeList();
        printParam(os, variadic, diagShared, demangleParams);
      } else {
        printConvention(convention);

        if (printStar)
          os << '*';

        if (convention == ArgConvention::Ref ||
            convention == ArgConvention::MutRef)
          printRef(cast<RefType>(type));

        if (!name.empty())
          os << name.getValue() << ": ";

        if (hasAddress(convention))
          type = type.getReferenceElementType();
        type.print(os, diagShared, demangleParams);
      }

      // Check if we are at the end; if so, we might still have to print a '/'.
      // If we're pretty printing for a diagnostic, and don't have any names,
      // then we don't print the trailing slash. This makes the extremely
      // common case of a source signature `fn(...) -> ...` look nicer.
      if (!diagShared || hadAnyNames)
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
    Type resultType = sig.getUserResultType();

    if (sig.isRefResult()) {
      auto refType = cast<RefType>(resultType);
      printRef(refType);
      resultType = refType.getElementType();
    }

    if (isa<KGEN::NoneType>(resultType))
      os << "None";
    else
      ASTType(resultType).print(os, diagShared, demangleParams);
  } else if (auto paramRef = dyn_cast<ParamType>(type)) {
    printParam(os, paramRef.getParam(), diagShared, demangleParams);
  } else if (isa<TypeType>(type)) {
    os << "AnyTrivialRegType";
  } else if (auto fnType = dyn_cast<FunctionType>(type)) {
    os << "fn (";
    llvm::interleaveComma(fnType.getInputs(), os, [&](Type type) {
      ASTType(type).print(os, diagShared, demangleParams);
    });
    os << ") -> (";
    llvm::interleaveComma(fnType.getResults(), os, [&](Type type) {
      ASTType(type).print(os, diagShared, demangleParams);
    });
    os << ')';
  } else if (auto originType = dyn_cast<OriginType>(type)) {
    if (originType.isMutableKnown(true))
      os << "MutableOrigin";
    else if (originType.isMutableKnown(false))
      os << "ImmutableOrigin";
    else {
      os << "Origin[";
      printDemangledParam(os, originType.isMutable(), diagShared);
      os << ']';
    }
  } else {
    // Use KGEN pretty printing when printing bare MLIR types for diagnostics.
    if (diagShared)
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

std::string ASTType::getAsString(SharedState *forDiags,
                                 bool demangleParams) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, forDiags, demangleParams);

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(result.begin(), result.end(), '@', '_');
  return os.str();
}

/// Get the specified parameter as a string.
std::string ASTType::getParamAsString(TypedAttr param, SharedState *diagShared,
                                      bool demangleParams) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printParam(os, param, diagShared, demangleParams);
  return os.str();
}

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.

  auto *shared = static_cast<SharedState *>(diag.getDiags()->extraContext);
  diag << '\'' << type.getAsString(/*forDiag=*/shared) << '\'';
}

void M::addToDiagnostic(TypedAttr paramValue, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.

  diag << '\'';
  auto *shared = static_cast<SharedState *>(diag.getDiags()->extraContext);
  diag << ASTType::getParamAsString(paramValue, /*forDiag=*/shared,
                                    /*demangleParams=*/true);
  diag << '\'';
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }

RefType ASTType::getRefForArgument(const Twine &argName, bool isMut) {
  auto ctx = mlirType.getContext();
  auto selfOrigin = ParamDeclRefAttr::get(StringAttr::get(ctx, argName + "`"),
                                          OriginType::get(ctx, isMut));
  return RefType::get(mlirType, selfOrigin, /*addressSpace=*/0);
}

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
