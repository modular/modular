//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJO_PARSER_FUNCTIONTYPES_H
#define KGEN_MOJO_PARSER_FUNCTIONTYPES_H

namespace M::KGEN::LIT {
class CValue;
class ExprNode;
class ExprEmitter;
class LITSignatureType;
class ValueDest;
struct TypeCheckScopeInfo;

/// Determine whether the function type `actual` can be non-trivially converted
/// to `expected`.
bool canConvertFunctionTypes(LITSignatureType actual, LITSignatureType expected,
                             const TypeCheckScopeInfo &scopeInfo);

CValue convertFunctionValue(CValue value, const ExprNode *expr,
                            LITSignatureType expected, ExprEmitter &emitter,
                            ValueDest &dest);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJO_PARSER_FUNCTIONTYPES_H
