//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions primarily for parsing, printing and
// verifying LIT related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITUTILS_H
#define KGEN_LITDIALECT_LITUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
class TypeArrayAttr;

namespace KGEN {
class ParamDeclAttr;
class ParamDeclArrayAttr;
class ParameterEvaluator;
class ParameterExprArrayAttr;

namespace LIT {
/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type);

/// Parse a ParamDeclAttr which has syntactic form `name (: type (= default)?
/// )?`. `defaultVal` is not modified if a default value was not present.
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result,
                           TypedAttr &defaultVal);

/// Parse a parameter specification in a lit op.
ParseResult
parseOptionalParameterSpec(AsmParser &p, ParamDeclArrayAttr &inputParamDecls,
                           ParamDeclArrayAttr &resultParamDecls,
                           SmallVectorImpl<TypedAttr> &defaultParams);

/// Print a parameter specification in a lit op. A ParameterEvaluator is
/// necessary to substitute parameters into parametric parameters.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> inputParamDecls,
                                ArrayRef<ParamDeclAttr> resultParamDecls,
                                ArrayRef<TypedAttr> defaultParams,
                                ParameterEvaluator &evaluator);

/// Parse a parameter signature (input/result types with optional default
/// values) if present.
ParseResult
parseOptionalParamSignature(AsmParser &p,
                            SmallVectorImpl<Type> &inputParamTypes,
                            SmallVectorImpl<Type> &resultParamTypes,
                            SmallVectorImpl<TypedAttr> &defaultParams);

/// Print the parameter type signature if there are any input or result types,
/// along with the default input parameter values.
void printOptionalParamSignature(AsmPrinter &p, TypeArrayAttr inputParamTypes,
                                 TypeArrayAttr resultParamTypes,
                                 ArrayRef<TypedAttr> defaultParams);

/// StructDeclOp parameter printing/parsing.
ParseResult parseStructParameterSpec(AsmParser &p,
                                     ParamDeclArrayAttr &inputParamDecls,
                                     ParameterExprArrayAttr &defaultParameters);
void printStructParameterSpec(AsmPrinter &p, Operation *op,
                              ArrayRef<ParamDeclAttr> inputParamDecls,
                              ParameterExprArrayAttr defaultParameters);
} // namespace LIT
} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
