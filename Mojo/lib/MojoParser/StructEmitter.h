//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Struct Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_STRUCTEMITTER_H
#define KGEN_MOJOPARSER_STRUCTEMITTER_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/SharedState.h"

namespace M::KGEN::LIT {

class AliasASTDecl;
class DefASTDecl;
class StructASTDecl;

/// This struct bundles up analysis of the value members of a struct.
class ValueInfo {
public:
  FnOp del;      // __del__
  FnOp copyctor; // __init__(*, copy=)
  FnOp movector; // __init__(*, move=)

  static std::optional<ValueInfo> lookupExisting(ASTDecl &structDecl);

private:
  ValueInfo() {}
};

class FunctionEmitter : public SharedStateUser {
public:
  FunctionEmitter(SharedState &shared) : SharedStateUser(shared) {}

  /// Emit an empty function stub at the specified location. The block arguments
  /// are added to the body of the function but no ops are added to the body.
  /// `suffix` is appended to the mangled function name. This adds the
  /// declaration to `parent`.
  std::pair<FnOp, DefASTDecl *> synthesizeFunction(
      ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
      ImplicitLocOpBuilder &builder, FnEffects fnEffects = FnEffects(),
      StringRef suffix = "", bool synthetic = true,
      InlineLevel inlineLevel = InlineLevel::Automatic,
      bool isNestedOriginsReadOnly = false);
};

class StructEmitter : public FunctionEmitter {
public:
  // The struct we're working on.
  StructASTDecl &structDecl;
  StructDeclOp structDeclOp;

  StructEmitter(StructASTDecl &structDecl);

  /// Generate empty stubs for the destructor, copy constructor, and move
  /// constructor on the declOp if they are eligible and do not already exist.
  ///
  /// A struct is eligible for a destructor if one of its fields has a
  /// destructor. It is possible that none of the fields of a struct have a
  /// destructor but that struct has an init that allocates heap memory. In this
  /// case set the forceGenerateDestructor flag to true to force destructor
  /// generation.
  std::optional<ValueInfo>
  addMissingValueMemberStubsToStruct(bool forceGenerateDestructor = false);

  /// Given a struct that has no explicitly defined `__del__` member, define a
  /// new one with an empty body. This allows the CheckLifetimes pass to insert
  /// field dels as needed, and makes sure that anything that refers to this
  /// struct properly runs its destructor.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause so that overload resolution rejects calls when the
  /// conditional conformance cannot be proven.
  FnOp synthesizeEmptyDtor(ConstraintAttr conformanceConstraint = {});
  /// The two halves of synthesizeEmptyDtor, for a caller that must create the
  /// destructor before the struct's field types are resolved.
  DefASTDecl *
  synthesizeEmptyDtorStub(ConstraintAttr conformanceConstraint = {});
  void diagnoseNonDeinitableFields(DefASTDecl &dtorDecl,
                                   ConstraintAttr conformanceConstraint = {});
  /// Add an empty move/copy ctor stub for this struct, to be filled in later.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause so that overload resolution rejects calls when the
  /// conditional conformance cannot be proven.
  FnOp synthesizeEmptyMoveOrCopyInit(bool isMove,
                                     ConstraintAttr conformanceConstraint = {});
  /// Populate the function with a field by field copy. This will fail if the
  /// given function does not have the expected signature.
  LogicalResult populateMoveCopy(DefASTDecl &fnDecl, bool isMove);

  /// Populate a trait function with its default implementation. Returns
  /// failure() if the meta-type to trait conversion fails; in this case the
  /// provided fnDecl is marked erroneous. Otherwise returns success().
  LogicalResult populateDefaultedTraitFunction(DefASTDecl &fnDecl);

  /// Add a attribute initializer method for this struct with a body.
  FnOp synthesizeFieldwiseInit();

  /// This synthesizes an __init__ method that accepts values for every field of
  /// a struct, making it easy for external clients to initialize it.
  /// The `injectedFields` argument can be specified when creating an init
  /// method for memory-only types where not all fields are initialized, though
  /// this requires manual modification of the returned FnOp to initialize any
  /// omitted fields.
  FnOp synthesizeFieldwiseInit(ArrayRef<Type> argTypes,
                               ArrayRef<ArgConvention> argConventions,
                               PogListAttr argListAttrs,
                               // None or Self if register passable.
                               ASTType litReturnType);

  /// Create a FnOp within the scope of the given Struct. The body is not
  /// populated. `suffix` is appended to the mangled function name.
  std::pair<FnOp, DefASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);
  std::pair<FnOp, DefASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);

  /// Synthesize an unresolved alias into the struct with the specified name .
  AliasASTDecl *synthesizeUnresolvedAlias(StringRef name);

  /// Synthesize the members of an `__enum` from the `lit.enum.case` ops in its
  /// body: the `EnumLike` case lists, the storage field, a `comptime` member
  /// per payloadless case, and a static method for each case with a payload.
  LogicalResult synthesizeEnumMembers();
  /// Synthesize the `EnumLike` aliases `_enum_case_names` and
  /// `_enum_case_types` with the given values, and the enum's only field,
  /// `__enum_storage`, of type `__EnumStorage[caseTypes]`. All three are
  /// registered as resolved declarations and hidden from the docs.
  void synthesizeEnumCaseListsAndStorage(StructDeclOp enumStorageDeclOp,
                                         ParamListAttr caseNames,
                                         ParamListAttr caseTypes);
  /// Synthesize the static method that constructs case `caseName` of an
  /// `__enum` from its `payload` types, documented by the case's `docString`.
  LogicalResult synthesizeEnumCaseConstructor(StringRef caseName,
                                              ArrayRef<Type> payload,
                                              SMLoc caseLoc,
                                              DocStringAttr docString);
  /// Populate the body of a case constructor synthesized by
  /// `synthesizeEnumCaseConstructor`.
  LogicalResult populateEnumCaseConstructor(DefASTDecl &fnDecl);
  /// Compute the value of `aliasDecl`, the `comptime` member that
  /// `synthesizeEnumMembers` declared for a case without a payload.
  LogicalResult resolveEnumCaseAlias(AliasASTDecl &aliasDecl);
  TypedAttr populateSpecialFnIsTrivial(SpecialFunctionKind kind);

  /// Like synthesizeMethodInStruct but accepts higher-level signature
  /// information and extracts the necessary components internally. Also handles
  /// all the post-creation setup specific to default trait method wrappers.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause constraint on the synthesized function so that overload
  /// resolution rejects calls when the conditional conformance cannot be
  /// proven.
  FnOp synthesizeDefaultTraitMethodWrapper(
      DefASTDecl &existingDecl, StringRef name,
      FnTypeGeneratorType wrapperSignature, FnOp traitFn, ASTDecl *traitFnDecl,
      ImplicitLocOpBuilder &builder, StringRef suffix = "",
      ConstraintAttr conformanceConstraint = {});
};

/// Look up `name` in the standard library module that implements enums,
/// emitting an error if it is missing.
ASTDecl *lookupEnumSupportDecl(SharedState &shared, StringRef name, SMLoc loc);

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
