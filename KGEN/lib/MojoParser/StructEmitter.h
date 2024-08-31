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
  LIT::FuncOp dtor;
  LIT::FuncOp copyCtr;
  LIT::FuncOp moveCtr;
  LIT::FuncOp init;
};

class ValueInfo {
public:
  enum FuncIndex { Destruct = 0, Move = 1, Copy = 2, FieldwiseInit = 3 };

  static std::optional<ValueInfo> createValueInfo(ASTDecl &structDecl,
                                                  SharedState &shared);
  bool hasDestructor() const { return existingFunctions[FuncIndex::Destruct]; }
  bool hasMove() const { return existingFunctions[FuncIndex::Move]; }
  bool hasCopy() const { return existingFunctions[FuncIndex::Copy]; }
  bool hasFieldwiseInit() const {
    return existingFunctions[FuncIndex::FieldwiseInit];
  }

private:
  ValueInfo(const std::bitset<4> &existingFunctions)
      : existingFunctions(existingFunctions) {}
  std::bitset<4> existingFunctions;
};

class StructEmitter : public SharedStateUser {
public:
  StructEmitter(SharedState &sharedState)
      : SharedStateUser(sharedState), noneType(shared.getNoneType()),
        noneAttr(shared.getNoneAttr()) {}
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

  /// Create a FuncOp within the scope of the given struct and add function
  /// terminators.
  LIT::FuncOp addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                            ArrayRef<Type> argTypes,
                            ArrayRef<ArgConvention> argConventions,
                            PogListAttr argListAttrs, SpecialFunctionKind kind,
                            ArrayRef<ParamDeclAttr> params,
                            PogListAttr paramListAttrs);
  LIT::FuncOp addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                            ArrayRef<Type> argTypes,
                            ArrayRef<ArgConvention> argConventions,
                            PogListAttr argListAttrs, SpecialFunctionKind kind);

  /// Given a struct that has no explicitly defined `__del__` member, define a
  /// new one with an empty body. This allows the CheckLifetimes pass to insert
  /// field dels as needed, and makes sure that anything that refers to this
  /// struct properly runs its destructor.
  LIT::FuncOp synthesizeEmptyDtor(ASTDecl &structDecl);
  /// Add an empty `__moveinit__` stub for this struct, to be filled in later.
  LIT::FuncOp synthesizeEmptyMoveInit(ASTDecl &structDecl);
  /// Add an empty `__copyinit__` stub for this struct, to be filled in later.
  LIT::FuncOp synthesizeEmptyCopyInit(ASTDecl &structDecl);

  /// Return the initializer method with the specified signature if it exists
  /// and null otherwise. The operands type is not expected to include self.
  LIT::FuncOp findInitInStruct(StructDeclOp structOp, ArrayRef<Type> operands);

  /// Emit an emtpy function stub at the specified location. The block arguments
  /// are added to the body of the function but no ops are added to the body.
  /// `suffix` is appended to the mangled function name.
  LIT::FuncOp createFunction(
      ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
      ImplicitLocOpBuilder &builder, FnEffects fnEffects = FnEffects(),
      StringRef suffix = "", bool synthetic = true);

  /// This synthesizes an __init__ method that accepts values for every field of
  /// a struct, making it easy for external clients to initialize it.
  /// The `injectedFields` argument can be specified when creating an init
  /// method for memory-only types where not all fields are initialized, though
  /// this requires manual modification of the returned FuncOp to initialize any
  /// omitted fields.
  LIT::FuncOp synthesizeMemberwiseInit(ASTDecl &structDecl,
                                       ArrayRef<Type> argTypes,
                                       ArrayRef<ArgConvention> argConventions,
                                       PogListAttr argListAttrs);

  /// Create a FuncOp within the scope of the given Struct. The body is not
  /// populated. `suffix` is appended to the mangled function name.
  std::pair<LIT::FuncOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, ASTDecl &structDecl,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);
  std::pair<LIT::FuncOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, ASTDecl &structDecl,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);

  /// Given a struct and a trait declaration, make the trait inherit from the
  /// struct if it does not already. This adds the trait decl to the struct's
  /// parent list and all transitive parents that are not already there.
  static void addTraitParent(StructDeclOp structOp, ASTDecl *traitDecl);
  static void appendTraits(SmallVectorImpl<TypeLineageAttr> &parentTypes,
                           ASTDecl *traitDecl);

  /// This adds a default return (lit.return of None, potentially converted
  /// to a variant) and emits a EndFuncOp.
  void appendDefaultReturnAndEndOp(ASTDecl &funcDecl);

protected:
  Type noneType;
  NoneAttr noneAttr;
};

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
