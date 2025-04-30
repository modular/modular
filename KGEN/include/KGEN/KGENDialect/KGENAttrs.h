//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the core KGEN attribute classes, provides implementation
// logic for working with them, and helpers for defining operations that take
// them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_H
#define KGEN_KGENDIALECT_KGENATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENEnums.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class OperationName;
} // namespace mlir

namespace M {
class TargetInfoAttr;

namespace KGEN {
class BuildInfoType;
class FuncOp;
class GeneratorOp;
class KGENDType;
class TargetType;
class VariadicType;
class VariadicAttr;
class VTableAttr;
class ConformanceOp;
} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

//===----------------------------------------------------------------------===//
// EmitAsAttr
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class EmitAsAttr : public IntegerAttr {
public:
  using IntegerAttr::IntegerAttr;
  static bool classof(Attribute attr);
  static EmitAsAttr get(MLIRContext *ctx, EmitAs val);
  EmitAs getValue() const;
};
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::ParamDeclRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamDeclRefAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamDeclRefAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Emit an MLIR operation call in a parameter context.
TypedAttr emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
