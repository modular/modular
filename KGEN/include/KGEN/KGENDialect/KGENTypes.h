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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"

namespace M::KGEN {
class FuncInterface;
class ParamBindAttr;
class ParamBindArrayAttr;
class ParamDeclArrayAttr;
class TypeArrayAttr;

/// Return the full signature of this declaration, including parameters from
/// enclosing struct declarations.
SignatureType getFullSignature(FuncInterface decl);
} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPES_H
