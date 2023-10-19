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
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

std::string LIT::mangleParameter(const Twine &baseName, unsigned line,
                                 unsigned col) {
  return ("_" + Twine(line) + "x" + Twine(col) + "_" + baseName).str();
}

StringRef LIT::demangleParameterName(StringRef name) {
  llvm::Regex re("^_[0-9]+x[0-9]+_");
  if (!re.match(name))
    return name;
  // Strip the prefix. Drop the leading underscore and the drop until the second
  // underscore. This way, the function can avoid returning a `std::string`.
  name = name.drop_front();
  return name.drop_front(name.find('_') + 1);
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

ParseResult LIT::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                           Type type, bool hasAddress) {
  if (hasAddress) {
    if (auto ptr = dyn_cast<PointerType>(type))
      type = ptr.getElementAsType();
    else if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementAsType();
  }
  if (succeeded(p.parseOptionalEqual()))
    return parseParamValue(p, defaultVal, type);
  return success();
}

ParseResult LIT::parseParamDecl(AsmParser &p, ParamDeclAttr &result,
                                StringAttr &name) {
  StringAttr declName;
  if (parseParamName(p, declName))
    return failure();

  // Parse the unmangled name or set it to empty if not present.
  if (succeeded(p.parseOptionalLSquare())) {
    if (parseParamName(p, name) || p.parseRSquare())
      return failure();
  } else {
    name = StringAttr::get(p.getContext());
  }

  Type type;
  if (parseColonTypeOrIndex(p, type))
    return failure();
  result = ParamDeclAttr::get(declName, type);
  return success();
}

void LIT::printParamDecl(AsmPrinter &p, ParamDeclAttr decl, StringAttr name) {
  printParamName(p, decl.getName());
  if (!name.empty()) {
    p << '[';
    printParamName(p, name);
    p << ']';
  }
  printColonTypeOrIndex(p, decl.getType());
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations, and default values.
/// parameter-decl   ::= identifier (`[` identifier `]`)?
///                        (`:` type (`=` expression)? )?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult
LIT::parseOptionalParameterSpec(AsmParser &p,
                                ParamDeclArrayAttr &inputParamDecls,
                                ParamDeclArrayAttr &resultParamDecls,
                                SmallVectorImpl<StringAttr> &paramNames,
                                SmallVectorImpl<PassingKind> &paramPassingKinds,
                                SmallVectorImpl<TypedAttr> &defaultParams) {
  bool foundDefault = false;

  StarSlashParser ssParser(p);
  auto parseWithDefault =
      [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    llvm::SMLoc loc = p.getCurrentLocation();

    if (OptionalParseResult res = ssParser.parseOptionalStarSlash(loc);
        res.has_value())
      return res.value();

    ParamDeclAttr decl;
    StringAttr name;
    if (failed(parseParamDecl(p, decl, name)))
      return failure();
    decls.emplace_back(decl);
    paramNames.emplace_back(name);

    TypedAttr defaultValue;
    if (failed(parseOptionalDefaultValue(p, defaultValue, decl.getType())))
      return failure();
    if (defaultValue) {
      foundDefault = true;
      defaultParams.emplace_back(defaultValue);
    } else if (foundDefault) {
      return p.emitError(loc, "expected parameter with default value");
    }
    return success();
  };

  if (failed(KGEN::parseOptionalParameterSpec(
          p, inputParamDecls, resultParamDecls, parseWithDefault)))
    return failure();

  auto [numPosOnly, numPosOrKw, numKwOnly] = ssParser.getNumPassingKinds();
  paramPassingKinds.append(numPosOnly, PassingKind::PosOnly);
  paramPassingKinds.append(numPosOrKw, PassingKind::PosOrKw);
  paramPassingKinds.append(numKwOnly, PassingKind::KwOnly);

  return success();
}

void LIT::printOptionalParameterSpec(AsmPrinter &p,
                                     ArrayRef<ParamDeclAttr> inputParamDecls,
                                     ArrayRef<ParamDeclAttr> resultParamDecls,
                                     ArrayRef<StringAttr> paramNames,
                                     ArrayRef<PassingKind> paramPassingKinds,
                                     ArrayRef<TypedAttr> defaultParams,
                                     ParameterEvaluator &evaluator) {
  // Substitute input and result parameters when printing default parameters.
  for (ParamDeclAttr param : inputParamDecls)
    evaluator.addInputValue(ParamDeclRefAttr::get(param));
  for (ParamDeclAttr param : resultParamDecls)
    evaluator.addResultValue(ParamDeclRefAttr::get(param));

  size_t numParams = inputParamDecls.size();
  size_t defaultIdxStart = numParams - defaultParams.size();
  size_t idx = 0;

  StarSlashPrinter ssPrinter(p, numParams, '|');
  auto printWithDefault = [&](ParamDeclAttr decl) {
    ssPrinter.printOptionalStarSlash(paramPassingKinds[idx], idx);

    printParamDecl(p, decl, paramNames[idx]);
    if (idx >= defaultIdxStart) {
      p << " = ";
      printParamValue(p, cast<TypedAttr>(evaluator.getReboundAttribute(
                             defaultParams[idx - defaultIdxStart])));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    ssPrinter.printOptionalTrailingSlash(idx++);
  };
  printOptionalParameterSpec(p, inputParamDecls, resultParamDecls,
                             printWithDefault);
}

ParseResult LIT::parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    SmallVectorImpl<Type> &resultParamTypes,
    SmallVectorImpl<StringAttr> &paramNames,
    SmallVectorImpl<PassingKind> &paramPassingKinds,
    SmallVectorImpl<TypedAttr> &defaultParams) {
  // Parse the input parameter types and optional default values.
  StarSlashParser ssParser(p);
  auto parseInputParam = [&](SmallVectorImpl<Type> &inputs) -> ParseResult {
    if (OptionalParseResult res =
            ssParser.parseOptionalStarSlash(p.getCurrentLocation());
        res.has_value())
      return res.value();

    // Parse an optional parameter name.
    if (parseOptionalName(p, paramNames.emplace_back()))
      return {};

    Type &type = inputs.emplace_back();
    if (failed(parseKGENType(p, type)))
      return failure();
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type)))
      return failure();
    if (defaultVal)
      defaultParams.emplace_back(defaultVal);
    return success();
  };

  if (failed(KGEN::parseOptionalParamSignature(
          p, inputParamTypes, resultParamTypes, parseInputParam)))
    return failure();

  auto [numPosOnly, numPosOrKw, numKwOnly] = ssParser.getNumPassingKinds();
  paramPassingKinds.append(numPosOnly, PassingKind::PosOnly);
  paramPassingKinds.append(numPosOrKw, PassingKind::PosOrKw);
  paramPassingKinds.append(numKwOnly, PassingKind::KwOnly);

  return success();
}

void LIT::printOptionalParamSignature(AsmPrinter &p,
                                      TypeArrayAttr inputParamTypes,
                                      TypeArrayAttr resultParamTypes,
                                      ArrayRef<StringAttr> paramNames,
                                      ArrayRef<PassingKind> paramPassingKinds,
                                      ArrayRef<TypedAttr> defaultParams) {
  size_t numParams = inputParamTypes.size();
  size_t defaultIdxStart = numParams - defaultParams.size();
  size_t idx = 0;

  StarSlashPrinter ssPrinter(p, numParams, '|');
  auto printWithDefault = [&](Type type) {
    ssPrinter.printOptionalStarSlash(paramPassingKinds[idx], idx);

    if (StringAttr name = paramNames[idx]; !name.empty()) {
      p.printString(name);
      p << ": ";
    }
    printKGENType(p, type);
    if (idx >= defaultIdxStart) {
      p << " = ";
      printParamValue(p, defaultParams[idx - defaultIdxStart]);
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    ssPrinter.printOptionalTrailingSlash(idx++);
  };

  KGEN::printOptionalParamSignature(p, inputParamTypes, resultParamTypes,
                                    printWithDefault);
}

ParseResult
LIT::parseStructParameterSpec(AsmParser &p, ParamDeclArrayAttr &inputParamDecls,
                              StringArrayAttr &paramNames,
                              PassingKindArrayAttr &paramPassingKinds,
                              ParameterExprArrayAttr &defaultParameters) {
  SmallVector<TypedAttr> defaultParams;
  SmallVector<StringAttr> paramNamesArr;
  SmallVector<PassingKind> paramPassingKindsArr;
  ParamDeclArrayAttr resultParams;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseOptionalParameterSpec(p, inputParamDecls, resultParams,
                                 paramNamesArr, paramPassingKindsArr,
                                 defaultParams))
    return failure();
  if (!resultParams.empty())
    return p.emitError(loc, "expected no result parameters");

  MLIRContext *ctx = p.getContext();
  defaultParameters = ParameterExprArrayAttr::get(ctx, defaultParams);
  paramNames = StringArrayAttr::get(ctx, paramNamesArr);
  paramPassingKinds = PassingKindArrayAttr::get(ctx, paramPassingKindsArr);

  return success();
}

void LIT::printStructParameterSpec(AsmPrinter &p, Operation *op,
                                   ArrayRef<ParamDeclAttr> inputParamDecls,
                                   ArrayRef<StringAttr> paramNames,
                                   PassingKindArrayAttr paramPassingKinds,
                                   ParameterExprArrayAttr defaultParameters) {
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(
      p, inputParamDecls,
      /*resultParamDecls=*/{}, paramNames, paramPassingKinds.getValue(),
      defaultParameters ? defaultParameters : ArrayRef<TypedAttr>(), evaluator);
}

ParseResult LIT::parseOptionalName(AsmParser &p, StringAttr &name) {
  std::string argName;
  if (succeeded(p.parseOptionalString(&argName)))
    if (failed(p.parseColon()))
      return failure();
  name = StringAttr::get(p.getContext(), argName);
  return success();
}

OptionalParseResult StarSlashParser::parseOptionalStarSlash(llvm::SMLoc loc) {
  if (succeeded(parser.parseOptionalVerticalBar())) {
    if (foundSlash)
      return parser.emitError(loc, "only one '|' allowed in signature");
    if (foundStar)
      return parser.emitError(loc, "'*' cannot precede '|' in signature");
    numPosOnly = idx;
    foundSlash = true;
    return mlir::success();
  }
  if (succeeded(parser.parseOptionalStar())) {
    if (foundStar)
      return parser.emitError(loc, "only one '*' allowed in signature");
    foundStar = true;
    numPosOrKw = idx - numPosOnly;
    return mlir::success();
  }

  ++idx;
  return std::nullopt;
}

std::tuple<size_t, size_t, size_t> StarSlashParser::getNumPassingKinds() const {
  size_t numPosOrKwSoFar = numPosOrKw;
  if (!foundStar)
    numPosOrKwSoFar = idx - numPosOnly;
  return {numPosOnly, numPosOrKwSoFar, idx - numPosOnly - numPosOrKwSoFar};
}

StarSlashPrinter::StarSlashPrinter(raw_ostream &os, size_t numInputs,
                                   bool suppressSlashAfterSelf, char slash)
    : os(os), numInputs(numInputs), prevPassingKind(PassingKind::PosOnly),
      suppressSlashAfterSelf(suppressSlashAfterSelf), slash(slash) {}

StarSlashPrinter::StarSlashPrinter(AsmPrinter &printer, size_t numInputs,
                                   char slash)
    : StarSlashPrinter(printer.getStream(), numInputs,
                       /*suppressSlashAfterSelf=*/false, slash) {}

void StarSlashPrinter::printOptionalStarSlash(PassingKind passingKind,
                                              size_t idx) {
  if (prevPassingKind == passingKind)
    return;

  switch (prevPassingKind) {
  case PassingKind::PosOnly:
    // Check if we are in the starting state; if no, this was the last
    // positional-only argument. Optionally, we may want to suppress '/' before
    // the second argument.
    if (idx != 0 && (!suppressSlashAfterSelf || idx != 1))
      os << slash << ", ";
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    break;
  case PassingKind::PosOrKw:
    assert(passingKind != PassingKind::PosOnly &&
           "positional-only argument cannot follow positional-or-keyword");
    os << "*, ";
    break;
  case PassingKind::KwOnly:
    llvm_unreachable("keyword-only argument must follow all other arguments");
  }
  prevPassingKind = passingKind;
}

void StarSlashPrinter::printOptionalTrailingSlash(size_t idx) const {
  if (suppressSlashAfterSelf && idx == 0)
    return;
  if (idx == numInputs - 1)
    if (prevPassingKind == PassingKind::PosOnly)
      os << ", " << slash;
}
