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

/// Verify a conversion between a SIMD type and an MLIR builtin type.
/// Conversions are assumed to be bi-directional. In error messages, the
/// direction of the conversion is controlled by the `fromSimd` parameter.
LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     SIMDType simd, Type builtinType, bool fromSimd);
} // namespace M::KGEN::POP

#endif // KGEN_POPDIALECT_POPUTILS_H
