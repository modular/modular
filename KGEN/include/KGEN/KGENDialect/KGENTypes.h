//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_H
#define KGEN_KGENDIALECT_KGENTYPES_H

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ForwardDecls.h"

namespace M::KGEN {
class KGENDeclInterface;
class ParamBindAttr;
class ParamBindArrayAttr;
class ParamDeclArrayAttr;
class TypeArrayAttr;

/// Return the full signature of this declaration, including parameters from
/// enclosing struct declarations.
SignatureType getFullSignature(KGENDeclInterface decl);

/// This describes the encoding of the first element of the convention specifier
/// in a SignatureType.
enum class FnEffects : uint8_t {
  None = 0,
  // TODO: Throw = 1 << 0
  // TODO: Async = 1 << 1
};

/// This describes the encoding of the value parameter conventions in a
/// SignatureType.
enum class ValueInputConvention : uint8_t {
  ByVal = 0,
  ByRef = 1,
};

} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPES_H
