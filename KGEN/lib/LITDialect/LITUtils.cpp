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
#include "KGEN/LITDialect/LITAttrs.h"
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
  return isa<AnyTypeType, AnyRegTypeType, MetaTypeType, TraitType,
             ParamRefType>(attr.getType());
}

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

void LIT::printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol) {
  os << symbol.getRootReference().strref();
  for (FlatSymbolRefAttr nestedRef : symbol.getNestedReferences())
    os << "::" << nestedRef.getValue();
}

ParseResult LIT::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                           Type type, bool hasAddress) {
  if (hasAddress) {
    if (auto ptr = dyn_cast<PointerType>(type))
      type = ptr.getElementType();
    else if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementType();
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

  PassingKindParser passingKindParser(p);
  auto parseWithDefault =
      [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    llvm::SMLoc loc = p.getCurrentLocation();

    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash(loc);
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
    } else if (foundDefault && !passingKindParser.isCurrentImplicit()) {
      return p.emitError(loc, "expected parameter with default value");
    }
    return success();
  };

  if (failed(KGEN::parseOptionalParameterSpec(
          p, inputParamDecls, resultParamDecls, parseWithDefault)))
    return failure();

  passingKindParser.populatePassingKinds(paramPassingKinds);
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
  size_t defaultEnd = numParams - countNumImplicitKinds(paramPassingKinds);
  size_t defaultStart = defaultEnd - defaultParams.size();
  size_t idx = 0;

  PassingKindPrinter passingKindPrinter(p, numParams, '|');
  auto printWithDefault = [&](ParamDeclAttr decl) {
    passingKindPrinter.printOptionalStarSlash(paramPassingKinds[idx], idx);

    printParamDecl(p, decl, paramNames[idx]);
    if (idx >= defaultStart && idx < defaultEnd) {
      p << " = ";
      printParamValue(p, cast<TypedAttr>(evaluator.getReboundAttribute(
                             defaultParams[idx - defaultStart])));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
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
  PassingKindParser passingKindPrinter(p);
  auto parseInputParam = [&](SmallVectorImpl<Type> &inputs) -> ParseResult {
    if (OptionalParseResult res =
            passingKindPrinter.parseOptionalStarSlash(p.getCurrentLocation());
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

  passingKindPrinter.populatePassingKinds(paramPassingKinds);
  return success();
}

void LIT::printOptionalParamSignature(AsmPrinter &p,
                                      ArrayRef<Type> inputParamTypes,
                                      ArrayRef<Type> resultParamTypes,
                                      ArrayRef<StringAttr> paramNames,
                                      ArrayRef<PassingKind> paramPassingKinds,
                                      ArrayRef<TypedAttr> defaultParams) {
  size_t numParams = inputParamTypes.size();
  size_t defaultEnd = numParams - countNumImplicitKinds(paramPassingKinds);
  size_t defaultStart = defaultEnd - defaultParams.size();
  size_t idx = 0;

  PassingKindPrinter passingKindPrinter(p, numParams, '|');
  auto printWithDefault = [&](Type type) {
    passingKindPrinter.printOptionalStarSlash(paramPassingKinds[idx], idx);

    if (StringAttr name = paramNames[idx]; !name.empty()) {
      p.printString(name);
      p << ": ";
    }
    printKGENType(p, type);
    if (idx >= defaultStart && idx < defaultEnd) {
      p << " = ";
      printParamValue(p, defaultParams[idx - defaultStart]);
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
  };

  KGEN::printOptionalParamSignature(p, inputParamTypes, resultParamTypes,
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

size_t LIT::countNumPosOnly(ArrayRef<PassingKind> kinds) {
  for (auto [idx, kind] : llvm::enumerate(kinds))
    if (kind != PassingKind::PosOnly)
      return idx;
  return kinds.size();
}

size_t LIT::countNumImplicitKinds(ArrayRef<PassingKind> kinds) {
  size_t num = 0;
  for (PassingKind kind : llvm::reverse(kinds)) {
    if (kind != PassingKind::Implicit)
      break;
    ++num;
  }
  return num;
}

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

OptionalParseResult PassingKindParser::parseOptionalStarSlash(llvm::SMLoc loc) {
  if (succeeded(parser.parseOptionalVerticalBar())) {
    if (foundSlash)
      return parser.emitError(loc, "only one '|' allowed in signature");
    if (foundStar)
      return parser.emitError(loc, "'*' cannot precede '|' in signature");
    if (foundImplicit)
      return parser.emitError(loc, "'?' cannot precede '|' in signature");
    numPosOnly = idx;
    foundSlash = true;
    return mlir::success();
  }
  if (succeeded(parser.parseOptionalStar())) {
    if (foundStar)
      return parser.emitError(loc, "only one '*' allowed in signature");
    if (foundImplicit) {
      return parser.emitError(loc, "'?' cannot precede '*' in signature");
    }
    foundStar = true;
    numPosOrKw = idx - numPosOnly;
    return mlir::success();
  }
  if (succeeded(parser.parseOptionalQuestion())) {
    if (foundImplicit)
      return parser.emitError(loc, "only one '?' allowed in signature");
    foundImplicit = true;
    if (foundStar)
      numKwOnly = idx - numPosOrKw - numPosOnly;
    else
      numPosOrKw = idx - numPosOnly;
    return mlir::success();
  }

  ++idx;
  return std::nullopt;
}

void PassingKindParser::populatePassingKinds(
    SmallVectorImpl<PassingKind> &kinds) const {
  auto [numPosOnly, numPosOrKw, numKwOnly, numImplicit] = getNumPassingKinds();
  kinds.append(numPosOnly, PassingKind::PosOnly);
  kinds.append(numPosOrKw, PassingKind::PosOrKw);
  kinds.append(numKwOnly, PassingKind::KwOnly);
  kinds.append(numImplicit, PassingKind::Implicit);
}

std::tuple<size_t, size_t, size_t, size_t>
PassingKindParser::getNumPassingKinds() const {
  size_t numPosOrKwSoFar = numPosOrKw;
  if (!foundStar && !foundImplicit)
    numPosOrKwSoFar = idx - numPosOnly;
  size_t numKwOnlySoFar = numKwOnly;
  if (!foundImplicit)
    numKwOnlySoFar = idx - numPosOnly - numPosOrKwSoFar;
  return {numPosOnly, numPosOrKwSoFar, numKwOnlySoFar,
          idx - numKwOnlySoFar - numPosOrKwSoFar - numPosOnly};
}

PassingKindPrinter::PassingKindPrinter(raw_ostream &os, size_t numInputs,
                                       bool suppressSlashAfterSelf, char slash)
    : os(os), numInputs(numInputs), prevPassingKind(PassingKind::PosOnly),
      suppressSlashAfterSelf(suppressSlashAfterSelf), slash(slash) {}

PassingKindPrinter::PassingKindPrinter(AsmPrinter &printer, size_t numInputs,
                                       char slash)
    : PassingKindPrinter(printer.getStream(), numInputs,
                         /*suppressSlashAfterSelf=*/false, slash) {}

void PassingKindPrinter::printOptionalStarSlash(PassingKind passingKind,
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
  if (idx == numInputs - 1)
    if (prevPassingKind == PassingKind::PosOnly)
      os << ", " << slash;
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
    // It's a module name if it starts with a leading `$`.
    if (current.starts_with("$"))
      out.moduleNames.push_back(
          StringAttr::get(mangled.getContext(), current.drop_front()));
    else
      out.structNames.push_back(StringAttr::get(mangled.getContext(), current));
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
