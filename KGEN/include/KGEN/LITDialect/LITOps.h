//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_LITOPS_H
#define KGEN_KGENDIALECT_LITOPS_H

#include "KGEN/CODialect/COTypes.h"
#include "KGEN/HLCFDialect/HLCFAttrs.h"
#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class NoneType;
class PointerType;
class ReturnOp;

namespace LIT {
enum class SpecialFunctionKind : uint8_t;
class SpecialFunctionInfo;

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under the TryOp region that will handle a 'raise'd
/// error, or if this is in a function that is allowed to raise.  This returns
/// the TryOp or FuncOp if found, or null if raise is not valid.
Operation *findOpProcessingRaise(Block *currentBlock);

/// Given a call or indirect call, return the callee signature type.
LITSignatureType getCalleeType(Operation *op);
/// Given a call or indirect call, return the callee argument values.
ValueRange getCalleeArguments(Operation *op);

/// Return the fully resolved symbol reference for the given declaration,
/// including all scoping that may be needed, making it unique for every
/// declaration.
SymbolRefAttr getFullyResolvedSymbolRef(mlir::SymbolOpInterface op);

/// Returns the user-defined result type of a signature, looking through
/// implicit memory results and stripping off the variant from error throwing
/// results if needed.
Type getSignatureUserResultType(SignatureType sigType, ArrayRef<Type> argTypes,
                                Type resultType);

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<LITSignatureType, ParameterExprArrayAttr>
getUnboundSpecializedSignature(LITSignatureType type,
                               ParameterExprArrayAttr bindings);

/// Get the full signature of a declaration in the given context.
LITSignatureType getFullSignature(Operation *container,
                                  LITSignatureType signature);

} // namespace LIT
} // namespace M::KGEN

namespace M::DebugInfo {
class DIFileAttr;
} // namespace M::DebugInfo

#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.h.inc"

#endif // KGEN_KGENDIALECT_LITOPS_H
