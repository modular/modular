//===- MetaTypes.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/MetaDialect/MetaDialect.td"

#ifndef METATYPES_TD
#define METATYPES_TD

// Base class for other typedefs. Provides dialact-specific defaults.
class MetaType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MetaDialect, name, [
        DeclareTypeInterfaceMethods<OpaqueObjectInterface>
      ] # traits> {
  let mnemonic = typeMnemonic;
}

// Base class for parameterized types.
class ParameterizedMetaType<string name, string typeMnemonic,
                            list<Trait> traits = []>
    : MetaType<name, typeMnemonic, [
        DeclareTypeInterfaceMethods<SubElementTypeInterface>,
      ] # traits>;

#endif // METATYPES_TD
