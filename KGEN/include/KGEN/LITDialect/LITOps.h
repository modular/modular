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

} // namespace LIT
} // namespace M::KGEN

namespace M::DebugInfo {
class DIFileAttr;
} // namespace M::DebugInfo

#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.h.inc"

namespace M::KGEN::LIT {
/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under the try region of a try op.
bool findTryBlock(Block *currentBlock);

/// This class provides a wrapper around a mojo FuncOp that mangles its name (in
/// `mangled`) but also provides all the components of the mangled name. If the
/// func is already mangled, this will pull everything apart.
struct MangledSymbol {
  /// Mangle the symbol for this op by walking upwards and adding struct/module
  /// names.
  static MangledSymbol mangle(mlir::SymbolOpInterface op);
  /// Demangle this mangled name by parsing it into its component parts.
  static FailureOr<MangledSymbol> demangle(StringAttr mangled,
                                           bool parseSignature = true);

  /// The format for a mangled name is roughly:
  ///  $<module name>::<struct name>[::<struct name>]
  ///    ::<function name>[<comma separated params>]
  ///      (<comma-separated args>)<comma-separated results>

  /// The fully mangled name.
  StringAttr mangled;
  /// The various strings that make up the mangled name.
  SmallVector<StringAttr, 1> moduleNames;
  /// We support nested structs, so there may be more than one struct name.
  SmallVector<StringAttr, 1> structNames;
  /// The bare name of the symbol, which may include parameters.
  StringAttr symName;
  /// The bare name of the symbol without parameters.
  StringAttr identifier;
  /// If the symbol has a signature mangled into the name, then it will be here.
  FunctionType signature;
};

/// Print a mangled symbol.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MangledSymbol &ms);
} // namespace M::KGEN::LIT

#endif // KGEN_KGENDIALECT_LITOPS_H
