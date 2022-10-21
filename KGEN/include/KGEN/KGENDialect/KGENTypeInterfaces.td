//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DTypeInterface
//===----------------------------------------------------------------------===//

def DTypeInterface : TypeInterface<"DTypeInterface"> {
  let description = [{
    This interface is used to interact with types that contain a parametric
    data type parameter.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the typed expression representing the data type.",
      "TypedAttr", "getDType">,
    InterfaceMethod<
      /*desc=*/[{
        Try to resolve the data type from the parameter. Returns
        `DType::invalid` if the type could not be resolved.
      }], "Optional<KGEN::KGENDType>", "getResolvedDType", (ins), [{}], [{
        if (auto constant = dyn_cast<DTypeConstantAttr>($_type.getDType()))
          return constant.getDType();
        return {};
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

def DataLayoutInterface : TypeInterface<"DataLayoutInterface"> {
  let description = [{
    This interface is used to query the size and layout of types.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the size in bytes of the type.",
      "Optional<int64_t>", "getTypeSize", (ins "TargetInfoAttr":$target)>,
    InterfaceMethod<
      /*desc=*/"Get the alignment in bytes of the type.",
      "Optional<int64_t>", "getTypeAlign", (ins "TargetInfoAttr":$target)>,
  ];

  let extraClassDeclaration = [{
    /// Try to get the size in bytes of the type.
    static Optional<int64_t>
    getTypeSizeInBytes(TargetInfoAttr target, Type type);

    /// Try to get the aligment in bytes of the type.
    static Optional<int64_t>
    getTypeAlignInBytes(TargetInfoAttr target, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// OpaqueObjectInterface
//===----------------------------------------------------------------------===//

def OpaqueObjectInterface : TypeInterface<"OpaqueObjectInterface"> {
  let description = [{
    This should be used by types to define how to generate an object of their
    type. The object should match a densely-packed LLVM struct in its layout, so
    for example the struct:

    ```mlir
    !llvm.struct<packed (i64, ptr, i8)>
    ```

    should basically do:

    ```c
    int64_t size = ...;
    memcpy(obj, &size, sizeof(int64_t));
    void *ptr = ...;
    memcpy(obj, &ptr, sizeof(void *));
    int8_t type = ...;
    memcpy(obj, &type, sizeof(int8_t));
    ```
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Given a pointer, a tag, and an InputGenKind fill the pointer with an
      object of this type. The tag is the histogram bucket tag - it could be
      a size, a layout, or a combination of the two. It's fully defined by
      the user. Returns failure when the object could not be generated, and
      returns success when it was possible.
    }],
    "::mlir::LogicalResult", "populate",
      (ins "::mlir::Location":$loc, "::M::KGEN::InputGenKind":$kind,
           "::mlir::Attribute":$tag, "void *":$obj)
    >,
    InterfaceMethod<[{
      Given the pointer passed into `populate`, free any additional allocations
      needed to instantiate the object.
    }],
    "void", "destroy", (ins "::mlir::Attribute":$tag, "void *":$obj)
    >,
    InterfaceMethod<[{
      Return the number of bytes required to instantiate an object of this
      type with the given tag. Returns failure when the size in bytes cannot
      be computed.
    }],
    "::mlir::FailureOr<size_t>", "getSizeInBytes",
      (ins "::mlir::Location":$loc, "::mlir::Attribute":$tag)
    >,
    InterfaceMethod<[{
      Given two pointers to objects of this type, return true if they are
      equal, false if not, and return failure if they cannot be compared.
      The data passed in is effectively the data returned by `populate` and
      should be compared accordingly; for an array this means the
      implementation should likely compare the elements of the array, for
      example.
    }],
    "::mlir::FailureOr<bool>", "equals",
      (ins "::mlir::Location":$loc, "::mlir::Attribute":$tag, "void *":$lhsData,
           "void *":$rhsData)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

/// A type whose data type conforms to a constraint.
class ConstrainedDType<string constraint> : ConfinedType<DTypeInterface,
    [CPred<!subst("$_self", "$_self.cast<DTypeInterface>().getResolvedDType()",
                 constraint)>]>;

/// A type whose data type is boolean.
def BoolDType : ConstrainedDType<"$_self && $_self->isBool()">;
/// A type whose data type is address.
def AddressDType : ConstrainedDType<"$_self && $_self->isAddress()">;
/// A type whose data type is either unbound or integral.
def IntegerOrUnboundDType :
    ConstrainedDType<"!$_self || $_self->isInt()">;
/// A type whose data type is either unbound or floating point.
def FloatOrUnboundDType :
    ConstrainedDType<"!$_self || $_self->isFloat()">;

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
