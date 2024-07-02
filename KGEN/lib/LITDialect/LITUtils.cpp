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
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

bool LIT::isTypeExpr(TypedAttr attr) {
  if (auto param = dyn_cast<ParamRefType>(attr.getType()))
    return isa<AnyStructType, AnyTraitType>(param.getParam().getType());
  return isa<TypeType, AnyStructType, TraitType, AnyTraitType>(attr.getType());
}

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

StringRef LIT::demangleParameterName(StringRef name) {
  // Strip the "`" postfix and and trailing depth and unique ID.
  return name.empty() ? name : name.take_front(name.find('`'));
}

/// Hide the implementation of `demangleIfNeeded` from the header file by
/// putting the combined type and attribute implementation in the source file.
template <typename AttrOrType>
static AttrOrType demangleIfNeededImpl(AttrOrType arg) {
  auto demangle = [](auto declOrRef) {
    return decltype(declOrRef)::get(demangleParameterName(declOrRef.getName()),
                                    demangleIfNeeded(declOrRef.getType()));
  };

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](ParamDeclRefAttr declRef) { return demangle(declRef); });
  replacer.addReplacement([&](ParamDeclAttr decl) { return demangle(decl); });
  return replacer.replace(arg);
}

Attribute LIT::impl::demangleIfNeeded(Attribute arg) {
  return demangleIfNeededImpl(arg);
}

Type LIT::impl::demangleIfNeeded(Type arg) { return demangleIfNeededImpl(arg); }

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print a (potentially) parametric mutability specifier and then a value.  The
/// forms are: "imm expr", "mut expr", "mut=<expr>, expr" and "muttoimm expr"
/// without quotes.
void LIT::printLifetimeParamValue(AsmPrinter &p, TypedAttr value) {
  LifetimeType type = cast<LifetimeType>(value.getType());

  // It is extremely common to have a LifetimeMutCastAttr cast from knwon
  // mutable lifetime to known immutable lifetime (this happens when borrowed
  // arguments are formed).  So much so that we sugar it.
  if (auto castVal = dyn_cast<LifetimeMutCastAttr>(value);
      castVal && type.isMutableKnown(false) &&
      cast<LifetimeType>(castVal.getOperand().getType()).isMutableKnown(true)) {
    p << "muttoimm ";
    value = castVal.getOperand();
  } else {
    TypedAttr mutability = type.isMutable();
    if (auto boolAttr = dyn_cast<BoolAttr>(mutability)) {
      p << (boolAttr.getValue() ? "mut " : "imm ");
    } else {
      p << "mut=";
      printParamValue(p, mutability);
      p << ", ";
    }
  }

  // Now that the type is specified, print the lifetime value itself.
  printParamValue(p, value);
}

ParseResult LIT::parseLifetimeParamValue(AsmParser &p, TypedAttr &result) {
  LifetimeType type;
  // Parse the pretty type specifier if present.
  if (succeeded(p.parseOptionalKeyword("imm"))) {
    type = LifetimeType::get(p.getContext(), false);
  } else if (succeeded(p.parseOptionalKeyword("mut"))) {
    // !lit.ref<T, mut lifetime>    ==> mutable
    TypedAttr mutability;
    if (failed(p.parseOptionalEqual())) {
      mutability = BoolAttr::get(p.getContext(), true);
    } else {
      // !lit.ref<T, mut=expr, lifetime  ==> parametric
      if (parseI1ParamValue(p, mutability) || p.parseComma())
        return failure();
    }
    type = LifetimeType::get(mutability);
  } else if (succeeded(p.parseOptionalKeyword("muttoimm"))) {
    // Operand is mutable, casted to immutable.
    if (KGEN::parseParamValue(p, result,
                              LifetimeType::get(p.getContext(), true)))
      return failure();
    result = LifetimeMutCastAttr::get(result, false);
    return success();
  } else {
    // If none of "mut/imm/muttoimm" are specified, it may be an "ugly" style.
    // This is useful to support for Mojo composability.
    return p.parseAttribute(result);
  }

  // Ok, we found the type of the lifetime, parse the value next.
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
  if (succeeded(p.parseOptionalEqual()))
    return parseParamValue(p, defaultVal, type);
  return success();
}

/// Helper to parse sigils that indicate that an argument/parameter is variadic
/// or a pack. The given index is emplaced in the appropriate list of indices,
/// if a `var` or `pack` sigil is parsed.
static ParseResult parseVariadicness(AsmParser &p,
                                     SmallVectorImpl<size_t> &variadicIndices,
                                     ssize_t &packIndex, size_t idx) {
  mlir::SMLoc loc = p.getCurrentLocation();
  StringRef sigil;
  if (succeeded(p.parseOptionalKeyword(&sigil))) {
    if (sigil == "var")
      variadicIndices.emplace_back(idx);
    else if (sigil == "pack") {
      if (packIndex != -1)
        return p.emitError(loc, "multiple packs not supported");
      packIndex = idx;
    } else
      return p.emitError(loc, "expected 'var' or 'pack', got: ") << sigil;
  }
  return success();
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations, and default values.
/// parameter-decl   ::= identifier (`[` identifier `]`)?
///                        (`:` type (`=` expression)? )?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult LIT::parseOptionalParameterSpec(AsmParser &p,
                                            ParamDeclArrayAttr &inputParamDecls,
                                            PogListAttr &paramListAttr) {
  MLIRContext *ctx = p.getContext();
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> defaultPosParams;
  SmallVector<TypedAttr> defaultKwOnlyParams;
  SmallVector<size_t> variadicIndices;
  ssize_t packIndex = -1;
  std::optional<ArgConvention> origPackConvention;

  bool foundPosDefault = false;
  bool foundKwOnlyDefault = false;

  llvm::SMLoc startLoc = p.getCurrentLocation();
  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseWithDefault =
      [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    llvm::SMLoc loc = p.getCurrentLocation();
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    StringAttr paramName;
    if (succeeded(p.parseOptionalLSquare())) {
      std::string str;
      if (p.parseString(&str) || p.parseRSquare())
        return failure();
      paramName = StringAttr::get(ctx, str);
    }

    ParamDeclAttr decl;
    if (failed(parseParamDecl(p, decl)))
      return failure();
    decls.emplace_back(decl);

    // We store an empty string for the name of implicit parameters.
    bool isImplicit = passingKindParser.isCurrentImplicit();
    if (!paramName) {
      paramName = StringAttr::get(
          ctx, isImplicit ? "" : demangleParameterName(decl.getName()));
    }
    paramNames.emplace_back(paramName);

    if (failed(parseVariadicness(p, variadicIndices, packIndex, idx++)))
      return failure();

    // Parameters don't really have ArgConvention's.
    if (packIndex == ssize_t(idx - 1))
      origPackConvention = ArgConvention::BorrowedInReg;

    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, decl.getType())))
      return failure();
    if (defaultVal) {
      if (passingKindParser.isCurrentKwOnly()) {
        defaultKwOnlyParams.emplace_back(defaultVal);
        foundKwOnlyDefault = true;
      } else {
        defaultPosParams.emplace_back(defaultVal);
        foundPosDefault = true;
      }
    } else if (!isImplicit) {
      if (passingKindParser.isCurrentKwOnly() && foundKwOnlyDefault) {
        return p.emitError(
            loc, "expected keyword-only parameter with default value");
      }
      if (!passingKindParser.isCurrentKwOnly() && foundPosDefault) {
        return p.emitError(loc,
                           "expected positional parameter with default value");
      }
    }
    return success();
  };

  ParamDeclArrayAttr resultParamDecls;
  if (failed(KGEN::parseOptionalParameterSpec(
          p, inputParamDecls, resultParamDecls, parseWithDefault)))
    return failure();
  if (!resultParamDecls.empty())
    return p.emitError(startLoc, "expected no result parameters");

  passingKindParser.populatePassingKinds(paramPassingKinds);

  paramListAttr = PogListAttr::get(
      ctx, paramNames, paramPassingKinds, defaultPosParams, defaultKwOnlyParams,
      variadicIndices, packIndex, std::move(origPackConvention));
  return success();
}

ParseResult LIT::parseConventionAndVariadicness(
    AsmParser &p, ArgConvention &convention,
    SmallVectorImpl<size_t> &variadicIndices, ssize_t &argPackIndex,
    std::optional<ArgConvention> &origArgPackConvention, size_t idx) {
  mlir::SMLoc loc = p.getCurrentLocation();
  StringRef str;
  convention = ArgConvention::OwnedInReg;
  if (succeeded(p.parseOptionalKeyword(&str))) {
    if (std::optional<ArgConvention> conv = symbolizeArgConvention(str)) {
      convention = *conv;
      // If we just had a convention and no vertical bar, we're done.
      if (failed(p.parseOptionalVerticalBar()))
        return success();
      // Otherwise we also parse a variadicness
      if (parseVariadicness(p, variadicIndices, argPackIndex, idx))
        return failure();

      if (argPackIndex == ssize_t(idx)) {
        argPackIndex = idx;
        origArgPackConvention = convention;
        if (convention == ArgConvention::OwnedInMem)
          convention = ArgConvention::OwnedInReg;
        else
          convention = ArgConvention::BorrowedInReg;
      }
      return success();
    }
    if (str == "var")
      variadicIndices.push_back(idx);
    else if (str == "pack") {
      if (argPackIndex != -1)
        return p.emitError(loc, "multiple packs not supported");
      argPackIndex = idx;
      origArgPackConvention = convention;
      if (convention == ArgConvention::OwnedInMem)
        convention = ArgConvention::OwnedInReg;
      else
        convention = ArgConvention::BorrowedInReg;
    } else
      return p.emitError(loc, "expected convention|variadicnes, got: ") << str;
  }
  return success();
}

/// Print the variadicness as a strings.
static void printVariadicness(AsmPrinter &p, Variadicness variadicness,
                              char separator = ' ') {
  if (variadicness == Variadicness::kNone)
    return;

  p << separator;
  if (variadicness == Variadicness::kVariadic)
    p << "var";
  else if (variadicness == Variadicness::kPack) {
    p << "pack";
  } else
    llvm_unreachable("unknown Variadicness");
}

void LIT::printConventionAndVariadicness(AsmPrinter &p,
                                         ArgConvention convention,
                                         Variadicness variadicness) {
  if (convention == ArgConvention::OwnedInReg)
    return printVariadicness(p, variadicness);

  p << ' ' << stringifyArgConvention(convention);
  printVariadicness(p, variadicness, '|');
}

/// Return an array of enums representing the variadicness of each
/// argument/parameter in the given list.
SmallVector<Variadicness> LIT::getVariadicness(PogListAttr pogListAttr) {
  size_t numPogs = pogListAttr.getPogs().size();
  SmallVector<Variadicness> res;
  res.reserve(numPogs);
  for (size_t idx = 0; idx < numPogs; ++idx) {
    res.push_back(pogListAttr.isVariadic(idx) ? Variadicness::kVariadic
                                              : Variadicness::kNone);
  }
  if (pogListAttr.hasPack())
    res[pogListAttr.getPackIndex()] = Variadicness::kPack;
  return res;
}

void LIT::printOptionalParameterSpec(AsmPrinter &p,
                                     ArrayRef<ParamDeclAttr> paramDecls,
                                     PogListAttr paramListAttr,
                                     ParameterEvaluator &evaluator) {
  // Substitute input parameters when printing default parameters.
  for (ParamDeclAttr param : paramDecls)
    evaluator.addInputValue(ParamDeclRefAttr::get(param));

  DefaultValueHandler defaultHandler(paramListAttr);
  SmallVector<Variadicness> variadicness = getVariadicness(paramListAttr);
  size_t idx = 0;
  PassingKindPrinter passingKindPrinter(p, paramListAttr, '|');
  auto printWithDefault = [&](ParamDeclAttr decl) {
    passingKindPrinter.printOptionalStarSlash(idx);

    StringAttr name = paramListAttr.getName(idx);
    // If we can't encode the parameter name inside the mangled decl name, then
    // print it explicitly.
    if (paramListAttr.getPassingKind(idx) != PassingKind::Implicit &&
        name != demangleParameterName(decl.getName()))
      p << '[' << name << ']';
    printParamDecl(p, decl);
    printVariadicness(p, variadicness[idx]);

    if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
      p << " = ";
      printParamValue(
          p, cast<TypedAttr>(evaluator.getReboundAttribute(defaultOr)));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
  };
  printOptionalParameterSpec(p, paramDecls, /*resultParams=*/{},
                             printWithDefault);
}

ParseResult
LIT::parseOptionalParamSignature(AsmParser &p,
                                 SmallVectorImpl<Type> &inputParamTypes,
                                 PogListAttr &paramListAttr) {
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> defaultPosParams;
  SmallVector<TypedAttr> defaultKwOnlyParams;
  SmallVector<size_t> variadicIndices;
  ssize_t packIndex = -1;

  // Parse the input parameter types and optional default values.
  llvm::SMLoc startLoc = p.getCurrentLocation();
  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseInputParam = [&](SmallVectorImpl<Type> &inputs) -> ParseResult {
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    // Parse an optional parameter name.
    if (parseOptionalName(p, paramNames.emplace_back()))
      return {};

    Type &type = inputs.emplace_back();
    if (failed(parseKGENType(p, type)))
      return failure();

    if (failed(parseVariadicness(p, variadicIndices, packIndex, idx++)))
      return failure();

    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type)))
      return failure();
    if (defaultVal) {
      if (passingKindParser.isCurrentKwOnly())
        defaultKwOnlyParams.emplace_back(defaultVal);
      else
        defaultPosParams.emplace_back(defaultVal);
    }
    return success();
  };

  SmallVector<Type> resultParamTypes;
  if (failed(KGEN::parseOptionalParamSignature(
          p, inputParamTypes, resultParamTypes, parseInputParam)))
    return failure();
  if (!resultParamTypes.empty())
    return p.emitError(startLoc, "expected no result parameters");

  passingKindParser.populatePassingKinds(paramPassingKinds);

  if (packIndex != -1)
    return p.emitError(startLoc, "pack not supported in parameter list");

  paramListAttr = PogListAttr::get(
      p.getContext(), paramNames, paramPassingKinds, defaultPosParams,
      defaultKwOnlyParams, variadicIndices, packIndex, std::nullopt);
  return success();
}

void LIT::printOptionalParamSignature(AsmPrinter &p,
                                      ArrayRef<Type> inputParamTypes,
                                      PogListAttr paramListAttr) {
  DefaultValueHandler defaultHandler(paramListAttr);
  SmallVector<Variadicness> variadicness = getVariadicness(paramListAttr);
  size_t idx = 0;
  PassingKindPrinter passingKindPrinter(p, paramListAttr, '|');
  auto printWithDefault = [&](Type type) {
    passingKindPrinter.printOptionalStarSlash(idx);

    if (StringAttr name = paramListAttr.getName(idx); !name.empty()) {
      p.printString(name);
      p << ": ";
    }
    printKGENType(p, type);
    printVariadicness(p, variadicness[idx]);

    if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
      p << " = ";
      printParamValue(p, defaultOr);
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
  };

  KGEN::printOptionalParamSignature(p, inputParamTypes, /*resultParamTypes=*/{},
                                    printWithDefault);
}

ParseResult LIT::parseOptionalName(AsmParser &p, StringAttr &name) {
  std::string argName;
  if (succeeded(p.parseOptionalString(&argName)))
    if (failed(p.parseColon()))
      return failure();
  name = StringAttr::get(p.getContext(), argName);
  return success();
}

ParseResult LIT::parseLifetimeSet(AsmParser &p,
                                  SmallVectorImpl<TypedAttr> &lifetimes) {
  OptionalParseResult result = parseOptionalLifetimeSet(p, lifetimes);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(), "expected a '{'");
  return *result;
}

OptionalParseResult
LIT::parseOptionalLifetimeSet(AsmParser &p,
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
    return parseParamValue(p, lifetimes.emplace_back(), LifetimeType::get(mut));
  };
  if (p.parseCommaSeparatedList(parseLifetime))
    return failure();
  return p.parseRBrace();
}

void LIT::printLifetimeSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes) {
  p << '{';
  auto printLifetime = [&](TypedAttr lifetime) {
    auto type = cast<LifetimeType>(lifetime.getType());
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
    printParamValue(p, lifetime);
  };
  llvm::interleaveComma(lifetimes, p, printLifetime);
  p << '}';
}

//===----------------------------------------------------------------------===//
// Pog Utils
//===----------------------------------------------------------------------===//

size_t LIT::countNumInferredKinds(ArrayRef<PogMetadataAttr> pogs) {
  size_t num = 0;
  for (PogMetadataAttr pogAttr : pogs) {
    if (pogAttr.getPassingKind() != PassingKind::Inferred)
      break;
    ++num;
  }
  return num;
}

size_t LIT::countNumInferredKinds(PogListAttr pogListAttr) {
  return countNumInferredKinds(pogListAttr.getPogs());
}

size_t LIT::countNumPosOnly(ArrayRef<PogMetadataAttr> pogs) {
  size_t idx = 0;
  for (PogMetadataAttr pog : pogs) {
    PassingKind kind = pog.getPassingKind();
    if (kind == PassingKind::Inferred)
      continue;
    if (kind != PassingKind::PosOnly)
      break;
    ++idx;
  }
  return idx;
}

size_t LIT::countNumPosOnly(PogListAttr pogListAttr) {
  return countNumPosOnly(pogListAttr.getPogs());
}

size_t LIT::countNumPositional(ArrayRef<PogMetadataAttr> pogs) {
  size_t idx = 0;
  for (PogMetadataAttr pog : pogs) {
    PassingKind kind = pog.getPassingKind();
    if (kind == PassingKind::Inferred)
      continue;
    if (kind != PassingKind::PosOnly && kind != PassingKind::PosOrKw)
      break;
    ++idx;
  }
  return idx;
}

size_t LIT::countNumPositional(PogListAttr pogListAttr) {
  return countNumPositional(pogListAttr.getPogs());
}

size_t LIT::countNumImplicitKinds(ArrayRef<PogMetadataAttr> pogs) {
  size_t num = 0;
  for (PogMetadataAttr pogAttr : llvm::reverse(pogs)) {
    if (pogAttr.getPassingKind() != PassingKind::Implicit)
      break;
    ++num;
  }
  return num;
}

size_t LIT::countNumImplicitKinds(PogListAttr pogListAttr) {
  return countNumImplicitKinds(pogListAttr.getPogs());
}

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

static std::optional<PassingKindParser::Marker>
parseOptionalMarker(AsmParser &p) {
  // We want to allow a standalone * in the signature to represent information
  // about a signature list, but we don't want to interfere with *"fo o" escaped
  // name parsing.  Do a bit of grotty lookahead to make sure we're ok to
  // consume a star.  While grotty, this cannot overrun the end of the file
  // because the MLIR asmparser guarantees the buffer is always NUL terminated.
  // This doesn't support whitespace/comments etc between the star and quote
  // though.
  llvm::SMLoc loc = p.getCurrentLocation();
  if (loc.getPointer()[0] == '*' && loc.getPointer()[1] == '"')
    return {};

  if (succeeded(p.parseOptionalPlus()))
    return PassingKindParser::PLUS;
  if (succeeded(p.parseOptionalVerticalBar()))
    return PassingKindParser::BAR;
  if (succeeded(p.parseOptionalStar()))
    return PassingKindParser::STAR;
  if (succeeded(p.parseOptionalQuestion()))
    return PassingKindParser::QUESTION;
  return {};
}

OptionalParseResult PassingKindParser::parseOptionalStarSlash() {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::optional<Marker> marker = parseOptionalMarker(parser);
  if (!marker) {
    ++idx;
    return std::nullopt;
  }

  // Error if the same marker was already found.
  if (foundMarkers[*marker]) {
    return parser.emitError(loc, "only one '")
           << markers[*marker] << "' allowed in signature";
  }
  // Error if any markers that are supposed to come after were already parsed.
  for (int i = *marker + 1; i < NUM_MARKERS; ++i) {
    if (foundMarkers[i]) {
      return parser.emitError(loc, "'") << markers[i] << "' cannot precede '"
                                        << markers[*marker] << "' in signature";
    }
  }

  foundMarkers[*marker] = true;
  idxOfEach[*marker] = idx;
  return mlir::success();
}

void PassingKindParser::populatePassingKinds(
    SmallVectorImpl<PassingKind> &kinds) const {
  size_t lastIdx = 0;
  // Compute the number of elements from the previous marker to this marker.
  std::array<size_t, NUM_MARKERS + 1> fwdSegments{}, revSegments{};
  for (int i = 0; i < NUM_MARKERS; ++i) {
    if (foundMarkers[i]) {
      fwdSegments[i] = idxOfEach[i] - lastIdx;
      lastIdx = idxOfEach[i];
    } else {
      fwdSegments[i] = 0;
    }
  }
  fwdSegments[NUM_MARKERS] = idx - lastIdx;

  // Compute the number of elements from the next marker to this marker.
  lastIdx = idx;
  for (int i = NUM_MARKERS - 1; i >= 0; --i) {
    if (foundMarkers[i]) {
      revSegments[i] = lastIdx - idxOfEach[i];
      lastIdx = idxOfEach[i];
    } else {
      revSegments[i] = 0;
    }
  }
  revSegments[0] = lastIdx;

  // Number of inferred and positional only are the number of elements that come
  // before the marker, until the previous marker or beginning. Number of
  // implicit or keyword-only are the number of elements that come after the
  // marker, until the next marker or the end. The number of keyword or position
  // is everything else.
  kinds.append(fwdSegments[PLUS], PassingKind::Inferred);
  kinds.append(fwdSegments[BAR], PassingKind::PosOnly);
  kinds.append(idx - fwdSegments[PLUS] - fwdSegments[BAR] - revSegments[STAR] -
                   revSegments[QUESTION],
               PassingKind::PosOrKw);
  kinds.append(revSegments[STAR], PassingKind::KwOnly);
  kinds.append(revSegments[QUESTION], PassingKind::Implicit);
}

PassingKindPrinter::PassingKindPrinter(
    raw_ostream &os, size_t numPogs,
    std::function<PassingKind(size_t)> getPassingKind,
    bool suppressSlashAfterSelf, char slash)
    : os(os), numPogs(numPogs), getPassingKind(std::move(getPassingKind)),
      prevPassingKind(PassingKind::Inferred),
      suppressSlashAfterSelf(suppressSlashAfterSelf), slash(slash) {}

PassingKindPrinter::PassingKindPrinter(raw_ostream &os, PogListAttr pogListAttr,
                                       bool suppressSlashAfterSelf, char slash)
    : PassingKindPrinter(
          os, pogListAttr.getPogs().size(),
          [pogListAttr](size_t idx) { return pogListAttr.getPassingKind(idx); },
          suppressSlashAfterSelf, slash) {}

PassingKindPrinter::PassingKindPrinter(AsmPrinter &printer,
                                       PogListAttr pogListAttr, char slash)
    : PassingKindPrinter(printer.getStream(), pogListAttr,
                         /*suppressSlashAfterSelf=*/false, slash) {}

void PassingKindPrinter::printOptionalStarSlash(size_t idx) {
  PassingKind passingKind = getPassingKind(idx);
  if (prevPassingKind == passingKind)
    return;

  switch (prevPassingKind) {
  case PassingKind::Inferred:
    if (idx != 0)
      os << "+, ";
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::PosOnly:
    // Check if we are in the starting state; if no, this was the last
    // positional-only argument. Optionally, we may want to suppress '/' before
    // the second argument.
    assert(idx != 0);
    if (!suppressSlashAfterSelf || idx != 1)
      os << slash << ", ";
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::PosOrKw:
    assert(passingKind != PassingKind::PosOnly &&
           "positional-only argument cannot follow positional-or-keyword");
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::KwOnly:
    assert(passingKind == PassingKind::Implicit);
    os << "?, ";
    break;
  case PassingKind::Implicit:
    llvm_unreachable("implicit must be the last passing kind");
  }
  prevPassingKind = passingKind;
}

void PassingKindPrinter::printOptionalTrailingSlash(size_t idx) const {
  if (suppressSlashAfterSelf && idx == 0)
    return;
  if (idx == numPogs - 1) {
    if (prevPassingKind == PassingKind::PosOnly)
      os << ", " << slash;
    else if (prevPassingKind == PassingKind::Inferred)
      os << ", +";
  }
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
// DefaultValueHandler
//===----------------------------------------------------------------------===//

DefaultValueHandler::DefaultValueHandler(PogListAttr pogListAttr)
    : DefaultValueHandler(pogListAttr.getPogs(), pogListAttr.getDefaultPos(),
                          pogListAttr.getDefaultKwOnly()) {}

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

LogicalResult
LIT::verifyDefaultTypes(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<TypedAttr> defaultsPos,
                        ArrayRef<TypedAttr> defaultsKwOnly,
                        PogListAttr pogListAttr, ArrayRef<Type> types,
                        StringRef argOrParam, ArrayRef<ArgConvention> convs) {
  ArrayRef<PogMetadataAttr> pogs = pogListAttr.getPogs();
  DefaultValueHandler defaultHandler(pogs, defaultsPos, defaultsKwOnly);
  for (size_t idx = 0; idx < pogs.size(); ++idx) {
    TypedAttr defaultOr = defaultHandler.getDefault(idx);
    if (!defaultOr || pogListAttr.isPack(idx) || pogListAttr.isVariadic(idx))
      continue;

    Type expectedType = types[idx];
    Type defaultType = defaultOr.getType();

    // Memory-only arguments store their default values as pure values.
    if (!convs.empty())
      if (SignatureType::hasAddress(convs[idx]))
        expectedType = ::cast<RefType>(expectedType).getElementType();

    if (defaultType != expectedType &&
        !::isa<TypeCheckErrorType>(expectedType)) {
      return emitError() << argOrParam << " #" << idx << " has type "
                         << expectedType << " but the default " << argOrParam
                         << " value has type " << defaultType;
    }
  }

  return success();
}

LogicalResult
LIT::verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<PogMetadataAttr> pogs, size_t numPosDefaults,
                        size_t numKwOnlyDefaults, StringRef argOrParam) {
  // First, verify the order of passing kinds.
  auto latestKind = PassingKind::PosOnly;
  auto emitDiag = [&](PassingKind kind) {
    return emitError() << stringifyPassingKind(kind)
                       << " passing kind cannot follow "
                       << stringifyPassingKind(latestKind);
  };

  for (PogMetadataAttr pogAttr : pogs) {
    PassingKind kind = pogAttr.getPassingKind();
    if (kind == PassingKind::Implicit) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::Implicit)
      return emitDiag(kind);
    if (kind == PassingKind::KwOnly) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::KwOnly)
      return emitDiag(kind);
    if (kind == PassingKind::PosOrKw) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::PosOrKw)
      return emitDiag(kind);
    assert(latestKind == PassingKind::PosOnly);
  }

  auto emitTooManyDefaults = [&](size_t numDefaults, size_t numPassingKinds,
                                 StringRef kindStr) {
    return emitError() << "there are more default " << kindStr << " "
                       << argOrParam << "s than " << kindStr << " "
                       << argOrParam << "s: " << numDefaults << " vs. "
                       << numPassingKinds;
  };

  size_t numPos = countNumPositional(pogs);
  if (numPosDefaults > numPos)
    return emitTooManyDefaults(numPosDefaults, numPos, "positional");

  size_t numEl = pogs.size();
  size_t numKwOnly = numEl - numPos - countNumImplicitKinds(pogs) -
                     countNumInferredKinds(pogs);
  if (numKwOnlyDefaults > numKwOnly)
    return emitTooManyDefaults(numKwOnlyDefaults, numKwOnly, "keyword-only");

  return success();
}
