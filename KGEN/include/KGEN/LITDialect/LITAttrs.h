//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Regex.h"

namespace M::KGEN {
class NoneType;
namespace LIT {
class DeclRefType;
class LifetimeType;
class AnyStructType;
class StructFieldOp;
class UnpackedType;
class FnMetadataAttr;
class RefPackType;
} // namespace LIT
} // namespace M::KGEN

#include "KGEN/LITDialect/LITEnums.h.inc"

namespace M::KGEN::LIT {

/// Given a list of operations, create an array of bools (as a mask) indicating
/// variadic parameters in their concatenated list of parameter declarations.
/// The given operations must all implement DeclInterface.
SmallVector<bool> getContextualVariadicMask(ArrayRef<Operation *> ops);

} // namespace M::KGEN::LIT

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

#endif // KGEN_LITDIALECT_LITATTRS_H
