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

#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
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

namespace M::KGEN::LIT {
/// This class provides a wrapper around a mojo FuncOp that mangles its name (in
/// `mangled`) but also provides all the components of the mangled name. If the
/// func is already mangled, this will pull everything apart.
struct MangledSymbol {
  /// Mangle the symbol for this op by walking upwards and adding struct/module
  /// names.
  static MangledSymbol mangle(mlir::SymbolOpInterface op);
  /// Demangle this mangled name by parsing it into its component parts.
  static MangledSymbol demangle(StringAttr mangled);

  /// The fully mangled name.
  StringAttr mangled;
  /// The various strings that make up the mangled name.
  StringAttr moduleName;
  /// We support nested structs, so there may be more than one struct name.
  SmallVector<StringAttr, 1> structNames;
  /// The bare name of the symbol.
  StringAttr symName;
  /// If the symbol has a signature mangled into the name, then it will be here.
  StringAttr signature;
};

/// Print a mangled symbol.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MangledSymbol &ms);
} // namespace M::KGEN::LIT

#endif // KGEN_KGENDIALECT_NLKGENOPS_H
