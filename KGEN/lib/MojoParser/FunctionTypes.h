//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJO_PARSER_FUNCTIONTYPES_H
#define KGEN_MOJO_PARSER_FUNCTIONTYPES_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
class ASTDecl;
class CValue;
class ExprNode;
class ExprEmitter;
class FuncOp;
class LITSignatureType;
class ValueDest;
class SharedState;

/// Determine whether the function type `actual` can be non-trivially converted
/// to `expected`.
bool canConvertFunctionTypes(LITSignatureType actual,
                             LITSignatureType expected);

/// Emit a non-trivial conversion between two function types. This generates a
/// thunk and passes it on as the converted value.
CValue convertFunctionValue(CValue value, const ExprNode *expr,
                            LITSignatureType expected, ExprEmitter &emitter,
                            ValueDest &dest);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJO_PARSER_FUNCTIONTYPES_H
