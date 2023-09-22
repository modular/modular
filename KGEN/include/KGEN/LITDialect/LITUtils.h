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

namespace LIT {
/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type);

/// Parse a ParamDeclAttr which has syntactic form `name (: type (= default)?
/// )?`. `defaultVal` is not modified if a default value was not present.
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result,
                           TypedAttr &defaultVal);

/// Parse and print a parameter specification in a lit.func.
ParseResult
parseOptionalParameterSpec(AsmParser &p, ParamDeclArrayAttr &inputParamDecls,
                           ParamDeclArrayAttr &resultParamDecls,
                           SmallVectorImpl<TypedAttr> &defaultParams);

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
} // namespace LIT
} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
