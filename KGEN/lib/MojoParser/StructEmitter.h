//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Struct Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_STRUCTEMITTER_H
#define KGEN_MOJOPARSER_STRUCTEMITTER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/SharedState.h"

#include <bitset>

namespace M::KGEN::LIT {

struct GeneratedStubs {
  FnOp dtor;
  FnOp copyCtr;
  FnOp explicitCopy;
  FnOp moveCtr;
  FnOp init;
};

class ValueInfo {
public:
  enum FuncIndex {
    Destruct = 0,
    Move = 1,
    Copy = 2,
    ExplicitCopy = 3,
    FieldwiseInit = 4
  };

  static std::optional<ValueInfo> createValueInfo(ASTDecl &structDecl);
  bool hasNontrivialDestructor() const {
    return existingFunctions[FuncIndex::Destruct];
  }
  bool hasMove() const { return existingFunctions[FuncIndex::Move]; }
  bool hasCopy() const { return existingFunctions[FuncIndex::Copy]; }
  bool hasExplicitCopy() const {
    return existingFunctions[FuncIndex::ExplicitCopy];
  }
  bool hasFieldwiseInit() const {
    return existingFunctions[FuncIndex::FieldwiseInit];
  }

private:
  ValueInfo(const std::bitset<5> &existingFunctions)
      : existingFunctions(existingFunctions) {}
  std::bitset<5> existingFunctions;
};

class StructEmitter : public SharedStateUser {
public:
  StructEmitter(SharedState &sharedState)
      : SharedStateUser(sharedState), noneType(shared.getNoneType()) {}
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
  std::optional<GeneratedStubs>
  addMissingValueMemberStubsToStruct(ASTDecl &structDecl,
                                     bool generateFieldwiseInit,
                                     bool forceGenerateDestructor = false);

  /// Populate the function with a field by field copy. This will fail if the
  /// given function does not have the expected signature.
  LogicalResult populateMoveCopy(ASTDecl &functionDecl, bool isMove);

  /// Create a FnOp within the scope of the given struct and add function
  /// terminators.
  FnOp addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                     ArrayRef<Type> argTypes,
                     ArrayRef<ArgConvention> argConventions,
                     PogListAttr argListAttrs, SpecialFunctionKind kind,
                     ArrayRef<ParamDeclAttr> params,
                     PogListAttr paramListAttrs);
  FnOp addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                     ArrayRef<Type> argTypes,
                     ArrayRef<ArgConvention> argConventions,
                     PogListAttr argListAttrs, SpecialFunctionKind kind);

  /// Given a struct that has no explicitly defined `__del__` member, define a
  /// new one with an empty body. This allows the CheckLifetimes pass to insert
  /// field dels as needed, and makes sure that anything that refers to this
  /// struct properly runs its destructor.
  FnOp synthesizeEmptyDtor(ASTDecl &structDecl);
  /// Add an empty `__moveinit__` or `__copyinit__` stub for this struct, to be
  /// filled in later.
  FnOp synthesizeEmptyMoveOrCopyInit(ASTDecl &structDecl, bool isMove);
  /// Add `copy()` method for this struct.
  FnOp synthesizeExplicitCopy(ASTDecl &structDecl);

  /// Return the initializer method with the specified signature if it exists
  /// and null otherwise. The operands type is not expected to include self.
  FnOp findInitInStruct(StructDeclOp structOp, ArrayRef<Type> operands);

  /// Emit an emtpy function stub at the specified location. The block arguments
  /// are added to the body of the function but no ops are added to the body.
  /// `suffix` is appended to the mangled function name. This adds the
  /// declaration to `parent`.
  std::pair<FnOp, ASTDecl *> synthesizeFunction(
      ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
      ImplicitLocOpBuilder &builder, FnEffects fnEffects = FnEffects(),
      StringRef suffix = "", bool synthetic = true,
      InlineLevel inlineLevel = InlineLevel::Automatic);

  /// This synthesizes an __init__ method that accepts values for every field of
  /// a struct, making it easy for external clients to initialize it.
  /// The `injectedFields` argument can be specified when creating an init
  /// method for memory-only types where not all fields are initialized, though
  /// this requires manual modification of the returned FnOp to initialize any
  /// omitted fields.
  FnOp synthesizeMemberwiseInit(ASTDecl &structDecl, ArrayRef<Type> argTypes,
                                ArrayRef<ArgConvention> argConventions,
                                PogListAttr argListAttrs,
                                // None or Self if register passable.
                                ASTType litReturnType);

  /// Create a FnOp within the scope of the given Struct. The body is not
  /// populated. `suffix` is appended to the mangled function name.
  std::pair<FnOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, ASTDecl &structDecl, SMLoc loc,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);
  std::pair<FnOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, ASTDecl &structDecl, SMLoc loc,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);

private:
  FnOp createFunction(ASTDecl &parent, StringRef name,
                      ArrayRef<ParamDeclAttr> params,
                      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
                      ArrayRef<ArgConvention> argConventions,
                      PogListAttr argListAttrs, Type resultType,
                      SpecialFunctionKind specialFnID, SMLoc loc,
                      ImplicitLocOpBuilder &builder, FnEffects fnEffects,
                      StringRef suffix, bool synthetic,
                      InlineLevel inlineLevel);

protected:
  Type noneType;
};

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
