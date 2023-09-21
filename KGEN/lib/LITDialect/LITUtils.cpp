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

using namespace M;
using namespace KGEN;
using namespace LIT;

ParseResult LIT::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                           Type type) {
  if (succeeded(p.parseOptionalEqual()))
    return parseParamValue(p, defaultVal, type);
  return success();
}

ParseResult LIT::parseParamDecl(AsmParser &p, ParamDeclAttr &result,
                                TypedAttr &defaultVal) {
  if (failed(KGEN::parseParamDecl(p, result)))
    return failure();
  return parseOptionalDefaultValue(p, defaultVal, result.getType());
}

void LIT::printParamDecl(AsmPrinter &p, ParamDeclAttr decl,
                         TypedAttr defaultVal) {
  KGEN::printParamDecl(p, decl);
  if (defaultVal) {
    p << " = ";
    printParamValue(p, defaultVal);
  }
}

void LIT::printOptionalParameterSpec(AsmPrinter &p,
                                     ArrayRef<ParamDeclAttr> inputParamDecls,
                                     ArrayRef<ParamDeclAttr> resultParamDecls,
                                     ArrayRef<TypedAttr> defaultParams) {
  ssize_t defaultIdx = defaultParams.size() - inputParamDecls.size();
  auto printWithDefault = [&](ParamDeclAttr decl) {
    printParamDecl(p, decl,
                   defaultIdx >= 0 ? defaultParams[defaultIdx] : TypedAttr());
    ++defaultIdx;
  };

  return KGEN::printOptionalParameterSpec(p, inputParamDecls, resultParamDecls,
                                          printWithDefault);
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations, and default values.
/// parameter-decl   ::= identifier (`:` type (`=` expression)? )?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult
LIT::parseOptionalParameterSpec(AsmParser &p,
                                ParamDeclArrayAttr &inputParamDecls,
                                ParamDeclArrayAttr &resultParamDecls,
                                SmallVectorImpl<TypedAttr> &defaultParams) {
  bool foundDefault = false;
  auto parseWithDefault =
      [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    llvm::SMLoc loc = p.getCurrentLocation();
    ParamDeclAttr decl;
    if (failed(parseParamDecl(p, decl)))
      return failure();
    decls.emplace_back(decl);

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

  return KGEN::parseOptionalParameterSpec(p, inputParamDecls, resultParamDecls,
                                          parseWithDefault);
}

ParseResult
LIT::parseOptionalParamSignature(AsmParser &p,
                                 SmallVectorImpl<Type> &inputParamTypes,
                                 SmallVectorImpl<Type> &resultParamTypes,
                                 SmallVectorImpl<TypedAttr> &defaultParams) {
  // Parse the input parameter types and optional default values.
  auto parseInputParam = [&](SmallVectorImpl<Type> &inputs) -> ParseResult {
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

  return KGEN::parseOptionalParamSignature(p, inputParamTypes, resultParamTypes,
                                           parseInputParam);
}

void LIT::printOptionalParamSignature(AsmPrinter &p,
                                      TypeArrayAttr inputParamTypes,
                                      TypeArrayAttr resultParamTypes,
                                      ArrayRef<TypedAttr> defaultParams) {
  ssize_t defaultIdx = defaultParams.size() - inputParamTypes.size();
  auto printWithDefault = [&](Type type) {
    printKGENType(p, type);
    if (defaultIdx >= 0) {
      p << " = ";
      printParamValue(p, defaultParams[defaultIdx]);
    }
    ++defaultIdx;
  };

  KGEN::printOptionalParamSignature(p, inputParamTypes, resultParamTypes,
                                    printWithDefault);
}
