//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Struct Emission.
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTEMITTER_H
#define STRUCTEMITTER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "SharedState.h"

namespace M::KGEN::LIT {

struct GeneratedStubs {
  LIT::FuncOp dtor;
  LIT::FuncOp copyCtr;
  LIT::FuncOp moveCtr;
};

class StructEmitter {
public:
  StructEmitter(SharedState &sharedState) : shared(sharedState) {}
  /// Generate empty stubs for the destructor, copy constructor, and move
  /// constructor on the declOp if they are eligible and do not already exist.
  ///
  /// A struct is eligible for a move constructor if it is memory only.
  ///
  /// A struct is eligible for a copy constructor if it is not register passable
  /// trivial.
  ///
  /// A struct is eligible for a destructor if one of its fields has a
  /// destructor. It is possible that none of the fields of a struct have a
  /// destructor but that struct has an init that allocates heap memory. In this
  /// case set the forceGenerateDestructor flag to true to force destructor
  /// generation.
  GeneratedStubs
  addMissingValueMemberStubsToStruct(StructDeclOp declOp, SMLoc loc,
                                     ASTDecl &parent,
                                     bool forceGenerateDestructor = false);

  /// Populate the function with a field by field copy. This will fail if the
  /// given function does not have the expected signature.
  LogicalResult populateMoveCopy(LIT::FuncOp func, StructDeclOp declOp,
                                 ASTDecl &declScope, SMLoc location,
                                 bool isMove);

private:
  /// Create a FuncOp within the scope of the given struct and add function
  /// terminators.
  LIT::FuncOp addVoidMethod(StructDeclOp selfStruct, StringRef prefix,
                            ArrayRef<Type> argTypes,
                            ArrayRef<ValueInputConvention> argConventions,
                            ArrayRef<StringAttr> argNames,
                            SpecialFunctionKind kind, SMLoc loc);

  /// Create a FuncOp within the scope of the given Struct. The body is not
  /// populated.
  LIT::FuncOp
  synthesizeMethodInStruct(StringRef name, ArrayRef<Type> argTypes,
                           ArrayRef<ValueInputConvention> argConventions,
                           ArrayRef<StringAttr> argNames, Type resultType,
                           StructDeclOp structOp,
                           SpecialFunctionKind specialFnID, SMLoc loc);
  SharedState &shared;
};

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
