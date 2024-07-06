//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParameterTypeInterface
//===----------------------------------------------------------------------===//

def ParameterTypeInterface : TypeInterface<"ParameterTypeInterface"> {
  let description = [{
    The `ParameterTypeInterface` can be used by types to plug into various parts
    of the KGEN parameter system, including pretty parsing of attribute values.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Parse an attribute of this type, using pretty syntax if desired. This
      method should return none if the syntax guard does not match the expected
      pretty syntax of an attribute of this type. The KGEN parameter parser
      reserves several tokens to guard generic parameter attributes:

      - `*` denotes the beginning of an escaped parameter reference.
      - `?` denotes an unknown parameter value
      - barewords are parameter references or parameter operators

      The type may use any other token to guard its attribute pretty parsing. If
      the attribute's pretty format contains commas (e.g. a sequence of
      elements), its format must be surrounded by a delimiter to disambiguate
      from operator operand lists.
    }], "OptionalParseResult", "parseValue", (ins "AsmParser &":$p,
                                                  "TypedAttr &":$value)>,
    InterfaceMethod<[{
      Print an attribute of this type, using pretty syntax if desired. Return
      false if the attribute is not one the type knows how to print.
    }], "LogicalResult", "printValue", (ins "AsmPrinter &":$p,
                                            "TypedAttr":$value)>,
    InterfaceMethod<[{
      Return whether the type is a metatype.
    }], "bool", "isMetaType", (ins), [{}], [{ return false; }]>
  ];
}

//===----------------------------------------------------------------------===//
// ParameterScopeTypeInterface
//===----------------------------------------------------------------------===//

def ParameterScopeTypeInterface : TypeInterface<"ParameterScopeTypeInterface"> {
  let description = [{
    The `ParameterScopeTypeInterface` describes a type that declares a nested
    parameter scope within a type expression. It enables `ParamIndexRefAttr`
    values inside the type to reference parameters declared in a scope.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the types of the nested input parameter declarations.
    }], "ArrayRef<Type>", "getInputParamTypes">,
    InterfaceMethod<[{
      Return the types of the nested result parameter declarations.
    }], "ArrayRef<Type>", "getResultParamTypes">
  ];
}

//===----------------------------------------------------------------------===//
// StructTypeInterface
//===----------------------------------------------------------------------===//

def StructTypeInterface : TypeInterface<"StructTypeInterface"> {
  let description = [{
    This interface provides mechanisms for types that refer to a KGEN
    declaration to verify the referenced declaration.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify whether the symbol reference is valid. A symbol table is provided
      to lookup the type definition.
    }], "LogicalResult", "verifySymbolUses",
      (ins "Operation *":$module, "mlir::LockedSymbolTableCollection &":$symtab,
           "Location":$loc)>,
    InterfaceMethod<[{
      Get an optional alias for this symbol reference.
    }], "std::optional<StringRef>", "getAliasName">,
    InterfaceMethod<[{
      Pretty print the symbol reference as a type.
    }], "void", "printSymbol", (ins "AsmPrinter &":$p)>
  ];
}

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
