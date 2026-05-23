//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions primarily for parsing, printing and
// verifying LIT related operations and types.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENPogUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

bool LIT::isMetaType(Type type) {
  type = SugarAttr::strip(type);
  if (auto genType = dyn_cast<GeneratorType>(type))
    return isMetaType(genType.getBody());
  if (auto param = dyn_cast<ParamType>(type))
    return sugarIsa<StructMetaType, AnyTraitType>(param.getParam().getType());
  if (isa<NonStructTypeType, StructMetaType, StructMetaMetaType, TraitType,
          AnyTraitType, TypeType, FnLiteralTypeGeneratorMetaType>(type))
    return true;
  return false;
}
bool LIT::isVariadicOfMetaType(Type type) {
  auto va = sugarDynCast<ParamListType>(type);
  return va && LIT::isMetaType(va.getElementType());
}

bool LIT::isFirstLevelTypeExpr(TypedAttr typeExpr) {
  auto type = SugarAttr::strip(typeExpr.getType());
  if (auto param = dyn_cast<ParamType>(type))
    return sugarIsa<StructMetaMetaType, AnyTraitType>(
        param.getParam().getType());
  if (isa<StructMetaType, TraitType, NonStructTypeType,
          FnLiteralTypeGeneratorMetaType>(type))
    return true;

  // TypeType is not always a L1 type expression.
  return false;
}

bool LIT::isTypeExpr(TypedAttr attr) { return isMetaType(attr.getType()); }
bool LIT::isVariadicOfTypeExpr(TypedAttr attr) {
  auto va = sugarDynCast<ParamListAttr>(attr);
  return va && llvm::all_of(va.getValues(), LIT::isTypeExpr);
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print a (potentially) parametric mutability specifier and then a value.  The
/// forms are: "imm expr", "mut expr", "mut=<expr>, expr" and "muttoimm expr"
/// without quotes.
void LIT::printOriginParamValue(AsmPrinter &p, TypedAttr value) {
  // If the type is sugared, then we don't want to sugar this operation because
  // round tripping would lose the sugar.
  if (auto castVal = dyn_cast<OriginMutCastAttr>(value)) {
    if (auto type = dyn_cast<OriginType>(value.getType())) {
      if (auto srcType = dyn_cast<OriginType>(castVal.getOperand().getType())) {
        // It is extremely common to have a OriginMutCastAttr cast from known
        // mutable origin to known immutable origin (this happens when borrowed
        // arguments are formed).  So much so that we sugar it.
        if (type.isMutableKnown(false) && srcType.isMutableKnown(true)) {
          p << "muttoimm ";
          // Now that the type is specified, print the origin value itself.
          printParamValue(p, castVal.getOperand());
          return;
        }
      }
    }
  }

  TypedAttr mutability = sugarCast<OriginType>(value.getType()).isMutable();
  if (auto boolAttr = dyn_cast<BoolAttr>(mutability)) {
    p << (boolAttr.getValue() ? "mut " : "imm ");
  } else {
    p << "mut=";
    printParamValue(p, mutability);
    p << ", ";
  }

  // Now that the type is specified, print the origin value itself.
  printParamValue(p, value);
}

ParseResult LIT::parseOriginParamValue(AsmParser &p, TypedAttr &result) {
  OriginType type;
  // Parse the pretty type specifier if present.
  if (succeeded(p.parseOptionalKeyword("imm"))) {
    type = OriginType::get(p.getContext(), false);
  } else if (succeeded(p.parseOptionalKeyword("mut"))) {
    // !lit.ref<T, mut origin>    ==> mutable
    TypedAttr mutability;
    if (failed(p.parseOptionalEqual())) {
      mutability = BoolAttr::get(p.getContext(), true);
    } else {
      // !lit.ref<T, mut=expr, origin  ==> parametric
      if (parseI1ParamValue(p, mutability) || p.parseComma())
        return failure();
    }
    type = OriginType::get(mutability);
  } else if (succeeded(p.parseOptionalKeyword("muttoimm"))) {
    // Operand is mutable, casted to immutable.
    if (KGEN::parseParamValue(p, result, OriginType::get(p.getContext(), true)))
      return failure();
    result = OriginMutCastAttr::get(result, false);
    return success();
  } else {
    // If none of "mut/imm/muttoimm" are specified, it may be an "ugly" style.
    // This is useful to support for Mojo composability.
    return p.parseAttribute(result);
  }

  // Ok, we found the type of the origin, parse the value next.
  return KGEN::parseParamValue(p, result, type);
}

void LIT::printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol) {
  os << symbol.getRootReference().strref();
  for (FlatSymbolRefAttr nestedRef : symbol.getNestedReferences())
    os << "::" << nestedRef.getValue();
}

ParseResult LIT::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                           Type type, bool hasAddress) {
  if (hasAddress)
    if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementType();
  return KGEN::parseOptionalDefaultValue(p, defaultVal, type);
}

void LIT::printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal,
                                    Type type, bool hasAddress) {
  if (hasAddress)
    if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementType();
  KGEN::printOptionalDefaultValue(p, defaultVal, type);
}

ParseResult LIT::parseOriginSet(AsmParser &p,
                                SmallVectorImpl<TypedAttr> &lifetimes) {
  OptionalParseResult result = parseOptionalOriginSet(p, lifetimes);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(), "expected a '{'");
  return *result;
}

OptionalParseResult
LIT::parseOptionalOriginSet(AsmParser &p,
                            SmallVectorImpl<TypedAttr> &lifetimes) {
  if (failed(p.parseOptionalLBrace()))
    return std::nullopt;
  if (succeeded(p.parseOptionalRBrace()))
    return mlir::success();

  auto parseLifetime = [&]() -> ParseResult {
    TypedAttr mut;
    if (succeeded(p.parseOptionalKeyword("mut")))
      mut = BoolAttr::get(p.getContext(), true);
    else if (succeeded(p.parseOptionalKeyword("imm")))
      mut = BoolAttr::get(p.getContext(), false);
    else if (p.parseLParen() || parseI1ParamValue(p, mut) || p.parseRParen())
      return failure();
    return parseParamValue(p, lifetimes.emplace_back(), OriginType::get(mut));
  };
  if (p.parseCommaSeparatedList(parseLifetime))
    return failure();
  return p.parseRBrace();
}

void LIT::printOriginSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes) {
  p << '{';
  auto printLifetime = [&](TypedAttr origin) {
    auto type = cast<OriginType>(origin.getType());
    TypedAttr mut = type.isMutable();
    // If the mutability is known, pretty print it. Otherwise, print the
    // parametric mutability expression within parens.
    if (auto known = dyn_cast<BoolAttr>(mut)) {
      p << (known.getValue() ? "mut" : "imm");
    } else {
      p << '(';
      printParamValue(p, mut);
      p << ')';
    }
    p << ' ';
    printParamValue(p, origin);
  };
  llvm::interleaveComma(lifetimes, p, printLifetime);
  p << '}';
}

bool LIT::isEmptyOriginSet(TypedAttr attr) {
  if (!attr)
    return true;
  if (auto set = dyn_cast<OriginSetAttr>(attr))
    return set.getOperands().empty();
  return false;
}

void LIT::printFnType(AsmPrinter &p, FnType signature) {
  FnMetadataAttr metadata = signature.getMetadata();
  if (unsigned numOriginDecls = metadata.getNumImplicitOriginDecls())
    p << '[' << numOriginDecls << ']';
  if (!isEmptyOriginSet(metadata.getCaptureOrigins())) {
    p << ':';
    printParamValue(p, metadata.getCaptureOrigins());
    p << ':';
  }
  if (signature.getIsNestedOriginExclusivityCheckingDisabled())
    p << "no_nested_origin_exclusivity";

  PogListAttr argListAttr = signature.getArgListAttrs();
  PassingKindPrinter passingKindPrinter(p, argListAttr, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(i);

    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getArgument(i);
    ArgConvention argConv = signature.getArgConvention(i);
    VariadicKind variadicness = argListAttr.getVariadicKind(i);
    if (variadicness == VariadicKind::PosVarArg ||
        variadicness == VariadicKind::PackVarArg) {
      assert(argConv == ArgConvention::ReadMem ||
             argConv == ArgConvention::Mut ||
             argConv == ArgConvention::OwnedMem ||
             argConv == ArgConvention::OwnedReg);
      argConv = signature.getVariadicConvention(i);
    }
    printConventionAndVariadicness(p, argConv, variadicness);
    printOptionalDefaultValue(p, argListAttr.getDefault(i),
                              signature.getArgument(i), hasAddress(argConv));

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(i);
  };

  printSignatureValues(p, printElt, signature.getValues(),
                       signature.getArgConventions(), signature.getFnEffects(),
                       /*optionalResultList=*/false);
  assert(argListAttr.getBodyConstraints().empty());
  assert(llvm::all_of(argListAttr.getPogs(), [](PogMetadataAttr pog) {
    return pog.getConstraints().empty();
  }));
}

//===----------------------------------------------------------------------===//
// MangledSymbol
//===----------------------------------------------------------------------===//

MangledSymbol MangledSymbol::mangle(mlir::SymbolOpInterface op) {
  MangledSymbol out;
  // The parser mangles the argument types into the symbol name.
  size_t firstParen = op.getName().find('(');
  if (firstParen == std::string::npos)
    firstParen = op.getName().size();
  // Get the name of the func.
  out.symName =
      StringAttr::get(op.getContext(), op.getName().take_front(firstParen));
  out.identifier = StringAttr::get(
      op->getContext(),
      op.getName().take_front(op.getName().find_first_of("[(")));

  auto signatureStr =
      StringAttr::get(op.getContext(), op.getName().drop_front(firstParen));
  // If the operation is function-like, we can get its signature. However, using
  // it for name mangling breaks a lot of things right now.
  // TODO(10920): We have to re-evaluate if we want to have the parser doing
  //   some of this, or if we want to mangle it here.
  if (auto funcLike = dyn_cast<FuncInterface>(op.getOperation()))
    out.signature = funcLike.getFunctionType();
  else
    out.signature = nullptr;

  // Grab parent structs/modules/etc., add them in order from in -> out (they'll
  // be added to the name from out->in).
  Operation *parentOp = op;
  while ((parentOp = parentOp->getParentOp())) {
    TypeSwitch<Operation *>(parentOp)
        .Case([&](StructDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case([&](TraitDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case([&](ExtensionDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case<FileModuleOp, PackageOp>(
            [&](auto op) { out.moduleNames.push_back(op.getNameAttr()); });
  }
  std::reverse(out.structNames.begin(), out.structNames.end());
  std::reverse(out.moduleNames.begin(), out.moduleNames.end());

  std::string mangledName;
  llvm::raw_string_ostream nameStream(mangledName);
  // Emit the parent module and struct names. Module names are prefixed with `$`
  // - which provides a signal for what's a module vs struct when demangling.
  for (auto name : llvm::concat<StringAttr>(out.moduleNames, out.structNames))
    nameStream << name.getValue() << "::";
  // Finally, function name and argument types. Use the string coming out of the
  // parser rather than the actual function type.
  nameStream << out.symName.getValue() << signatureStr.getValue();

  out.mangled = StringAttr::get(op.getContext(), mangledName);
  return out;
}

/// Parse a mangled signature from `typeStr`. Expects a signature that looks
/// like `(type1,type2)rtype1,rtype2`.
static FailureOr<FunctionType> parseMangledSignature(MLIRContext *ctx,
                                                     StringRef typeStr) {
  SmallVector<Type> inputTypes, resultTypes;
  SmallVector<Type> *typeVec = &inputTypes;
  if (typeStr.empty())
    return FunctionType{};

  // Drop the first '(' if there is one.
  if (typeStr.starts_with("("))
    typeStr = typeStr.drop_front();

  // If the first thing in the string is the closing paren, move straight to
  // result types.
  if (typeStr.starts_with(")")) {
    typeStr = typeStr.drop_front();
    typeVec = &resultTypes;
  }

  // Now, parse the type string.
  while (!typeStr.empty()) {
    size_t numBytes = 0;
    Type t = mlir::parseType(typeStr, ctx, &numBytes);
    if (!t)
      return failure();

    typeVec->push_back(t);
    typeStr = typeStr.drop_front(numBytes);
    // Drop the comma.
    if (typeStr.starts_with(","))
      typeStr = typeStr.drop_front();

    // If we have reached the closing paren, then skip it and parse any
    // leftovers into the result types.
    if (typeStr.starts_with(")")) {
      typeStr = typeStr.drop_front();
      typeVec = &resultTypes;
    }
  }

  return FunctionType::get(ctx, inputTypes, resultTypes);
}

FailureOr<MangledSymbol> MangledSymbol::demangle(StringAttr mangled,
                                                 bool parseSignature) {
  MangledSymbol out;
  out.mangled = mangled;
  StringRef m = mangled.getValue();
  // We'll first tokenize the owning module and structs.
  size_t separator = m.find("::");
  size_t firstOpen = m.find_first_of("([");
  for (; separator != std::string::npos && separator < firstOpen;
       separator = m.find("::"), firstOpen = m.find_first_of("([")) {
    StringRef current = m.take_front(separator);
    // Drop until the separator.
    m = m.drop_front(separator);
    // Skip past the separator as well (if it exists).
    m.consume_front("::");
    // FIXME: Can't distinguish between struct and modules, but does it matter?
    out.moduleNames.push_back(StringAttr::get(mangled.getContext(), current));
  }
  // Get the name of the func and the types of its arguments.
  StringRef nameWithParameters = m.take_front(m.find('('));
  StringRef nameWithoutParameters = m.take_front(firstOpen);

  out.symName = StringAttr::get(mangled.getContext(), nameWithParameters);
  out.identifier = StringAttr::get(mangled.getContext(), nameWithoutParameters);

  size_t firstParen = m.find('(');
  if (firstParen == std::string::npos)
    firstParen = m.size();

  // If there's no parenthesis here, don't even parse out the signature.
  if (firstParen == m.size()) {
    out.signature = nullptr;
    return out;
  }

  // If there are more mangled symbols, then there are Mojo types we cannot
  // parse in general.
  if (separator != std::string::npos)
    return out;

  if (!parseSignature)
    return out;

  // If we *have* a signature, parse it out.
  FailureOr<FunctionType> sigOr =
      parseMangledSignature(mangled.getContext(), m.drop_front(firstParen));
  if (failed(sigOr))
    return failure();

  out.signature = *sigOr;
  return out;
}

llvm::raw_ostream &LIT::operator<<(raw_ostream &os, const MangledSymbol &ms) {
  os << "Mangled: \"";
  // Need to escape the mangled string, it might have some characters that
  // terminals don't like.
  llvm::printEscapedString(ms.mangled.getValue(), os);
  os << "\" - ";
  os << "Modules: [";
  llvm::interleaveComma(ms.moduleNames, os);
  os << "], Structs: [";
  llvm::interleaveComma(ms.structNames, os);
  os << "], Symbol: " << ms.symName;
  os << ", Identifier: " << ms.identifier;
  os << ", Signature: ";
  if (ms.signature)
    os << ms.signature;
  else
    os << "(none)";
  return os;
}

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

void LIT::sortAndDeduplicateSymbols(SmallVectorImpl<SymbolRefAttr> &symbols) {
  llvm::sort(symbols, [&](SymbolRefAttr a, SymbolRefAttr b) {
    if (a.getRootReference() != b.getRootReference())
      return a.getRootReference().getValue() < b.getRootReference().getValue();
    // Compare each segment of the symbols in dictionary order.
    ArrayRef<FlatSymbolRefAttr> aSegments = a.getNestedReferences();
    ArrayRef<FlatSymbolRefAttr> bSegments = b.getNestedReferences();
    for (auto [aSeg, bSeg] : llvm::zip(aSegments, bSegments)) {
      if (aSeg != bSeg)
        return aSeg.getValue() < bSeg.getValue();
    }
    return aSegments.size() < bSegments.size();
  });
  symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
}

void LIT::canonicalizeTraitCompositionSymbols(
    SmallVectorImpl<SymbolRefAttr> &symbols,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {

  // Pull in the entire ancestor chain.
  DenseSet<SymbolRefAttr> seen;
  for (SymbolRefAttr symbol : symbols) {
    if (!seen.insert(symbol).second)
      continue;

    TraitDeclOp traitOp = traitDeclResolver(symbol);
    // Only one level of parent lookup is needed because parentTypes always
    // include their entire ancestor chain.
    ArrayRef<SymbolRefAttr> parentSymbols =
        traitOp.getCanonicalTrait().getSymbols();
    seen.insert(parentSymbols.begin(), parentSymbols.end());
  }
  symbols.assign(seen.begin(), seen.end());

  sortAndDeduplicateSymbols(symbols);
}

FailureOr<TypedAttr> LIT::simplifyConformsToAgainstTypeValue(
    TypeConformsToTraitAttr conformsTo,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  TypedAttr typeValue = UpcastAttr::strip(conformsTo.getTypeValue());

  TraitType traitType;
  if (auto typeParam = sugarDynCast<TypeParamAttr>(typeValue)) {
    if (auto paramType = dyn_cast<ParamType>(typeParam.getMlirType()))
      traitType =
          dyn_cast<TraitType>(getCanonicalType(paramType.getParam().getType()));
  }
  if (!traitType)
    traitType = dyn_cast<TraitType>(getCanonicalType(typeValue.getType()));
  if (!traitType)
    return failure();

  // This is unfortunate that we have non-canonical trait types produced during
  // parsing. Canonicalize the trait type to get the full list of symbols.
  SmallVector<mlir::SymbolRefAttr> symbols(traitType.getSymbols());
  canonicalizeTraitCompositionSymbols(symbols, traitDeclResolver);

  // We can not prove falseness at parsing time for a looser-bound
  // type value, but we can prove correctness if the type variable has
  // a tighter trait bound.
  DenseSet<SymbolRefAttr> symbolSet(symbols.begin(), symbols.end());
  for (SymbolRefAttr toCheck : conformsTo.getTraitSymbols()) {
    if (!symbolSet.contains(toCheck))
      return failure();
  }

  return {getScalarBoolConstant(conformsTo.getContext(), true)};
}

static LIT::StructType getStructTypeForTypeValue(TypedAttr typeValue) {
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return nullptr;
  return sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
}

TypedAttr LIT::foldDowncastToStructType(DowncastAttr downcast) {
  if (auto structTp = getStructTypeForTypeValue(downcast.getInputTypeValue()))
    // FIXME: We should raise an error when the resolved struct type does not
    // conform to the downcast traits. The folding below is unsafe.
    return TypeParamAttr::get(structTp, downcast.getType());
  return {};
}

FailureOr<ResolvedStructHandle>
LITSymTabEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                            bool acceptAsync) {
  // LITSymTabEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.

  // First try to resolve a LIT struct decl.
  if (auto structType = getStructTypeForTypeValue(typeValue)) {
    if (auto decl = symtab.lookupSymbolIn<StructDeclOp>(
            module, structType.getSymbol())) {
      return ResolvedStructHandle{
          cast<StructDeclInterface>(decl.getOperation()),
          structType.getParamValues(), nullptr,
          /*instance=*/nullptr};
    }
  }
  // Otherwise, fall back to KGEN struct resolution.
  return SymTabEvaluationContext::resolveStructOp(typeValue, acceptAsync);
}

Operation *LITSymTabEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, StringAttr traitName) {
  return SymTabEvaluationContext::resolveConformanceForStruct(resolved,
                                                              traitName);
}

FuncInterface
LITSymTabEvaluationContext::resolveFunctionDecl(SymbolRefAttr symbol) {
  // Functions in the LIT phase are `lit.fn` ops; if not yet lowered to a
  // `kgen.generator`, fall back to the base lookup.
  if (auto fn = symtab.lookupSymbolIn<FnOp>(module, symbol))
    return fn;
  return SymTabEvaluationContext::resolveFunctionDecl(symbol);
}

FailureOr<TypedAttr> LITSymTabEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle TypeConformsToTraitAttr.
  if (auto conformsTo =
          sugarDynCastIfPresent<TypeConformsToTraitAttr>(typedAttr)) {
    // Try LIT-specific trait type folding first, then fall back to
    // constraint-aware struct resolution.
    FailureOr<TypedAttr> result = simplifyConformsToAgainstTypeValue(
        conformsTo, [&](SymbolRefAttr symbol) -> TraitDeclOp {
          return symtab.lookupSymbolIn<TraitDeclOp>(module, symbol);
        });
    if (succeeded(result))
      return result;
    return conformsTo.evaluateWithContext(*this);
  }

  // Handle DowncastAttr.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (auto structTp =
            getStructTypeForTypeValue(downcast.getInputTypeValue())) {
      // FIXME: We should raise an error when the resolved struct type does not
      // conforms to the downcast traits. The folding below is unsafe.
      return TypeParamAttr::get(structTp, downcast.getType());
    }

    auto toTrait = sugarDynCast<TraitType>(downcast.getType());

    // Extract source trait from the input type value.
    Type fromType = downcast.getInputTypeValue().getType();
    TraitType fromTrait = sugarDynCast<TraitType>(fromType);
    if (auto paramTrait = sugarDynCast<ParamType>(fromType);
        !fromTrait && paramTrait) {
      // if this is a !param<:anytrait<trait>, trait_val>, we can still get a
      // loosest bound from the trait meta type.
      if (auto anyTrait =
              sugarDynCast<AnyTraitType>(paramTrait.getParam().getType())) {
        fromTrait = anyTrait.getTraitType();
      }
      return failure();
    }

    if (!toTrait || !fromTrait)
      return failure();

    // FIXME: TraitType should be canonicalized by default. (Otherwise, you can
    // not test trait type equality by `traitType1 ==  traitType2` in MLIR).
    SmallVector<mlir::SymbolRefAttr> fromTraitSymbols(fromTrait.getSymbols());
    canonicalizeTraitCompositionSymbols(
        fromTraitSymbols, [&](SymbolRefAttr symbol) -> TraitDeclOp {
          return symtab.lookupSymbolIn<TraitDeclOp>(module, symbol);
        });

    SmallVector<mlir::SymbolRefAttr> toTraitsSymbols(toTrait.getSymbols());
    canonicalizeTraitCompositionSymbols(
        toTraitsSymbols, [&](SymbolRefAttr symbol) -> TraitDeclOp {
          return symtab.lookupSymbolIn<TraitDeclOp>(module, symbol);
        });

    llvm::SmallPtrSet<SymbolRefAttr, 16> fromSymbols(fromTraitSymbols.begin(),
                                                     fromTraitSymbols.end());
    bool fromImpliesTo =
        llvm::all_of(toTraitsSymbols, [&](SymbolRefAttr symbol) {
          return fromSymbols.contains(symbol);
        });
    if (fromImpliesTo) {
      // If we are downcasting a more-refined trait to a less-refined trait,
      // this is actually an upcast.
      return UpcastAttr::get(downcast.getType(), downcast.getInputTypeValue());
    } else {
      // We can not reuse fromTraitSymbols above as those are canonicalized :(
      // FIXME: ensure all trait type contains a canonicalized list of symbols.
      SmallVector<SymbolRefAttr> allTraitSymbols(fromTrait.getSymbols());
      llvm::append_range(allTraitSymbols, toTrait.getSymbols());
      sortAndDeduplicateSymbols(allTraitSymbols);

      auto allTraits = TraitType::get(attr.getContext(), allTraitSymbols, {});

      auto ret = UpcastAttr::get(
          downcast.getType(),
          DowncastAttr::get(allTraits, downcast.getInputTypeValue()));

      return ret;
    }
  }

  // Delegate to parent class for other context-specific handling.
  return SymTabEvaluationContext::evaluateContextSpecific(attr);
}

//===----------------------------------------------------------------------===//
// IndexToDeclRefRemapper
//===----------------------------------------------------------------------===//

Attribute IndexToDeclRefRemapper::tryReplace(Attribute attr, size_t depth) {
  if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
    if (ref.getDepth() == depth) {
      return ParamDeclRefAttr::get(paramListAttr.getName(ref.getIndex()),
                                   ref.getType());
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Constraint Implication
//===----------------------------------------------------------------------===//

/// If \p prop is a multi-trait TypeConformsToTraitAttr, decompose it into an
/// AND of individual single-trait conforms_to attrs. Returns a null attr if
/// \p prop is not a multi-trait conforms_to.
static TypedAttr decomposeConformsTo(TypedAttr prop) {
  auto conformsTo = dyn_cast<TypeConformsToTraitAttr>(prop);
  if (!conformsTo)
    return {};

  ArrayRef<SymbolRefAttr> traitSymbols = conformsTo.getTraitSymbols();
  if (traitSymbols.size() <= 1)
    return {};

  SmallVector<TypedAttr> operands;
  operands.reserve(traitSymbols.size());
  for (SymbolRefAttr sym : traitSymbols) {
    operands.push_back(
        TypeConformsToTraitAttr::get(conformsTo.getTypeValue(), {sym}));
  }
  return ParamOperatorAttr::get(POC::And, operands);
}

/// Check if prop is NOT(inner), i.e., XOR(inner, true). Returns inner if so.
static TypedAttr getNotOperand(TypedAttr prop) {
  auto xorOp = dyn_cast<ParamOperatorAttr>(prop);
  if (!xorOp || xorOp.getOpcode() != POC::Xor ||
      xorOp.getOperands().size() != 2)
    return {};

  // NOT is represented as XOR(x, true). Check both operand orderings.
  for (auto [maybeInner, maybeTrue] :
       {std::pair{xorOp.getOperand(0), xorOp.getOperand(1)},
        std::pair{xorOp.getOperand(1), xorOp.getOperand(0)}}) {
    if (isTriviallyTrueProposition(maybeTrue))
      return maybeInner;
  }
  return {};
}

ConstraintRelation LIT::inferConstraintRelation(TypedAttr propA,
                                                TypedAttr propB) {
  using CR = ConstraintRelation;

  // Canonicalize and decompose multi-trait conforms_to into AND of single-trait
  // ones so the general conjunction rules handle subsumption uniformly.
  propA = getCanonicalAttr(propA);
  propB = getCanonicalAttr(propB);
  if (TypedAttr d = decomposeConformsTo(propA))
    propA = d;
  if (TypedAttr d = decomposeConformsTo(propB))
    propB = d;

  // Direct equality: A implies A.
  if (propA == propB)
    return CR::Implies;

  // Trivially true is implied by anything.
  if (isTriviallyTrueProposition(propB))
    return CR::Implies;
  // Trivially false constraints are violated under any assumption. This is
  // sound because we know propA is not also trivially false at this point.
  if (isTriviallyFalseProposition(propB))
    return CR::Contradicts;

  // Negation rule: A contradicts NOT(A).
  // If B = NOT(inner) and A implies inner, then A contradicts B.
  if (TypedAttr innerB = getNotOperand(propB))
    if (constraintImplies(propA, innerB))
      return CR::Contradicts;
  // Symmetric: if A = NOT(inner) and B implies inner, then A contradicts B.
  if (TypedAttr innerA = getNotOperand(propA))
    if (constraintImplies(propB, innerA))
      return CR::Contradicts;

  if (auto paramOpB = dyn_cast<ParamOperatorAttr>(propB)) {
    // Weakening: A implies (A OR B) for any B.
    if (paramOpB.getOpcode() == POC::Or) {
      for (Attribute operand : paramOpB.getOperands())
        if (constraintImplies(propA, cast<TypedAttr>(operand)))
          return CR::Implies;
    }
    // Conjunction introduction: A implies (B AND C) iff A implies every
    // conjunct. A contradicts (B AND C) if A contradicts any conjunct.
    if (paramOpB.getOpcode() == POC::And) {
      CR result = CR::Implies;
      for (Attribute operand : paramOpB.getOperands()) {
        CR rel = inferConstraintRelation(propA, cast<TypedAttr>(operand));
        if (rel == CR::Contradicts)
          return CR::Contradicts;
        if (rel == CR::Unprovable)
          result = CR::Unprovable;
      }
      return result;
    }
  }

  // Conjunction elimination: (A AND B) implies B if any conjunct implies B.
  // AND decomposition: (A AND B) contradicts Z if any conjunct contradicts Z.
  if (auto paramOpA = dyn_cast<ParamOperatorAttr>(propA)) {
    if (paramOpA.getOpcode() == POC::And) {
      bool anySatisfied = false;
      for (Attribute operand : paramOpA.getOperands()) {
        CR rel = inferConstraintRelation(cast<TypedAttr>(operand), propB);
        if (rel == CR::Contradicts)
          return CR::Contradicts;
        if (rel == CR::Implies)
          anySatisfied = true;
      }
      if (anySatisfied)
        return CR::Implies;
    }
  }

  // Fallback: A implies B iff AND(A, B) == A.
  TypedAttr combined = ParamOperatorAttr::get(POC::And, {propA, propB});
  if (combined == propA)
    return CR::Implies;

  return CR::Unprovable;
}

LIT::ConformanceResult
LIT::evaluateConstraint(ParameterEvaluator &evaluator,
                        ConstraintAttr constraint,
                        ArrayRef<ConstraintAttr> callerAssumptions) {
  TypedAttr prop = getCanonicalAttr(constraint.getProposition());
  TypedAttr rebound = getCanonicalAttr(evaluator.getReboundAttribute(prop));

  if (isTriviallyTrueProposition(rebound))
    return ConformanceResult::Yes;
  if (isTriviallyFalseProposition(rebound))
    return ConformanceResult::No;

  bool anyImplies = false;
  for (ConstraintAttr assumption : callerAssumptions) {
    switch (inferConstraintRelation(
        getCanonicalAttr(assumption.getProposition()), rebound)) {
    case ConstraintRelation::Contradicts:
      return ConformanceResult::No;
    case ConstraintRelation::Implies:
      anyImplies = true;
      break;
    case ConstraintRelation::Unprovable:
      break;
    }
  }
  if (anyImplies)
    return ConformanceResult::Yes;

  return ConformanceResult::NeedsEvidence;
}

/// Visit each TypeConformsToTraitAttr found in a constraint proposition.
/// Canonical AND is already flattened to a single n-ary node, so a single
/// top-level loop over its operands is sufficient. OR / NOT are not visited
/// since they are not definite knowledge.
static void forEachConformsToInProposition(
    TypedAttr proposition,
    llvm::function_ref<void(TypeConformsToTraitAttr)> callback) {
  proposition = getCanonicalAttr(proposition);

  auto visit = [&](TypedAttr attr) {
    if (auto ct = dyn_cast<TypeConformsToTraitAttr>(getCanonicalAttr(attr)))
      callback(ct);
  };

  // Canonical AND is flattened to a single n-ary node, so iterate its
  // operands directly. Otherwise treat the proposition itself as a single
  // candidate.
  if (auto op = dyn_cast<ParamOperatorAttr>(proposition);
      op && op.getOpcode() == POC::And) {
    for (TypedAttr operand : op.getOperands())
      visit(operand);
    return;
  }
  visit(proposition);
}

/// Peel off transparent wrappers that do not change identity for the purpose
/// of matching a type parameter against a conforms_to constraint: rebind,
/// upcast, and downcast. Stripping upcasts/downcasts lets us match the same
/// underlying parameter even when one side has been statically widened
/// (e.g. `T: Movable` upcast to `AnyType`) or narrowed.
static TypedAttr stripIdentityWrappers(TypedAttr attr) {
  while (true) {
    TypedAttr stripped = ParamOperatorAttr::stripRebind(attr);
    stripped = UpcastAttr::strip(stripped);
    stripped = DowncastAttr::strip(stripped);
    if (stripped == attr)
      return attr;
    attr = stripped;
  }
}

TraitType LIT::getTraitBoundFromAssumptions(
    TypedAttr typeAttr, ArrayRef<ConstraintAttr> assumptions,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  typeAttr = getCanonicalAttr(typeAttr);

  if (assumptions.empty())
    return {};

  TypedAttr targetStripped = stripIdentityWrappers(typeAttr);

  // Collect trait symbols from all relevant conforms_to constraints.
  SmallVector<SymbolRefAttr> allTraits;
  for (ConstraintAttr assumption : assumptions) {
    forEachConformsToInProposition(
        assumption.getProposition(), [&](TypeConformsToTraitAttr ct) {
          TypedAttr ctStripped =
              stripIdentityWrappers(getCanonicalAttr(ct.getTypeValue()));
          if (!isEqualCanon(ctStripped, targetStripped))
            return;
          for (SymbolRefAttr symbol : ct.getTraitSymbols()) {
            if (!llvm::is_contained(allTraits, symbol))
              allTraits.push_back(symbol);
          }
        });
  }

  if (allTraits.empty())
    return {};

  // Canonicalize to include ancestor traits.
  canonicalizeTraitCompositionSymbols(allTraits, traitDeclResolver);

  return TraitType::get(typeAttr.getContext(), allTraits);
}

ParamDeclRefAttr LIT::extractParamDeclRef(TypedAttr attr) {
  if (auto upcast = dyn_cast<UpcastAttr>(attr))
    return extractParamDeclRef(upcast.getInputTypeValue());

  if (auto typeParam = dyn_cast<TypeParamAttr>(attr)) {
    Type innerType = typeParam.getTypeValue();
    if (auto innerParamType = dyn_cast<ParamType>(innerType))
      return extractParamDeclRef(innerParamType.getParam());
  }

  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr))
    return paramRef;

  return {};
}
