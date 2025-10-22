//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions primarily for parsing, printing and
// verifying POP related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPUTILS_H
#define KGEN_POPDIALECT_POPUTILS_H

#include "KGEN/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::POP {

/// Verify the conversion between the higher-level type and lower-level type.
LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     SIMDType simd, Type builtinType);
} // namespace M::KGEN::POP

#endif // KGEN_POPDIALECT_POPUTILS_H
