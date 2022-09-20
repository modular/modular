//===- POPTypes.td --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef POPTYPES_TD
#define POPTYPES_TD

include "KGEN/POPDialect/POPDialect.td"
include "mlir/IR/AttrTypeBase.td"

class POPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<POPDialect, typeName,
            [DeclareTypeInterfaceMethods<SubElementTypeInterface>] # traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def POP_ArrayType : POPType<"Array", "array"> {
  let summary = "parameterized array type";
  let description = [{
    This type represents a statically-sized array of elements. The size and
    element type are parameterized.

    Example:

    ```mlir
    // An array of four scalar floats.
    !pop.array<4, !meta.scalar<f32>>

    // A parameterized array.
    !pop.array<size, type>
    ```
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$elementType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<TypeParamValue>($elementType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size,
                                        "TypedAttr":$elementType)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the SIMD type to an integer.
    Optional<int64_t> resolveSize() const;
    /// Try to narrow the type expression of the pointer type to an MLIR type.
    Type resolveElementType() const;
  }];
}

#endif // POPTYPES_TD
