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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/HLCFDialect/HLCFInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class ReturnOp;

namespace POP {
class PointerType;
} // namespace POP

namespace LIT {
class NoneType;

/// Return the fully resolved symbol reference for the given declaration,
/// including all scoping that may be needed, making it unique for every
/// declaration.
SymbolRefAttr getFullyResolvedSymbolRef(mlir::SymbolOpInterface op);

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<SignatureType, ParamBindArrayAttr>
getUnboundSpecializedSignature(SignatureType type, ParamBindArrayAttr bindings);

} // namespace LIT
} // namespace M::KGEN

#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.h.inc"

#endif // KGEN_KGENDIALECT_NLKGENOPS_H
