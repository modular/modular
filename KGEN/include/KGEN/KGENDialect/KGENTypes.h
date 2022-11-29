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

} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPES_H
