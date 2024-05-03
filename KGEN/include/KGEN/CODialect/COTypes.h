//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CODIALECT_COTYPES_H
#define KGEN_CODIALECT_COTYPES_H

#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMForwardDecls.h"

namespace M::KGEN {
class SignatureType;
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/CODialect/COTypes.h.inc"

#endif // KGEN_CODIALECT_COTYPES_H
