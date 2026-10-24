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

#ifndef KGEN_MOJOPARSER_PARAMBINDINGS_H
#define KGEN_MOJOPARSER_PARAMBINDINGS_H

#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/MojoParser/CallOperands.h"

namespace M::KGEN {
class ParameterExprArrayAttr;
class PogListAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
using llvm::SMLoc;
class DeclResolver;
class ExprNode;
class FnOp;
class FnTypeGeneratorType;
class PValue;
class StructDeclOp;
class TypeSignatureType;
class ParamInf;
class VerifiedParamBindings;

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

/// This class holds a work-in-progress set of parameter bindings for a type or
/// function declaration.  Some of the bindings may be pre-checked, others may
/// not be.  They are eventually resolved and diagnosed via ParamInf inference.
///
/// Consider something like one of these:
///    SomeType[param1].method[param2](...args...)
///    OuterType[param1].InnerType[param2]
///
/// The type parameters (param1) will be bound typed-checked, and the param2
/// will be bound as the TypedAttr value of param2.  We cannot type check the
/// bindings until overload resolution has resolved which 'method' we are
/// talking about and when inference is complete, so we keep a flag.
class ParamBindings {
public:
  enum BindingKind {
    // Following the standard binding rules: the parameter binding should
    // produce the a concrete type. Every parameter must be bound explicitly
    // (either with a concrete value or a `_`).
    kStandard = 0,
    // Allow producing partially instantiated type even in the absence of `...`
    // or `_`, but STILL INSTALL default value unless unbound explicitly.
    kContextual = 1,
    // This typically means that there is explicit `...` in the parameter list.
    // Allow producing partially bound type, also NOT install default values.
    kWithEllipsis = 2,
  };

  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  /// Initialize ParamBindings with a declscope to perform lookups against
  /// and a notion of shared context.
  ParamBindings(ASTDecl &declScope, const ExprNode *expr);
  ParamBindings(const ParamBindings &) = delete;
  ParamBindings(ParamBindings &&other) = default;

  /// Replace our bindings with another set.
  void operator=(ParamBindings &&other);

  /// The overall expression this was formed for.
  const ExprNode *getExpr() const { return parameters.callExpr; }
  SMLoc getExprLoc() const;

  /// Return whether there are any bindings given.
  bool empty() const { return parameters.empty(); }

  // Provide access to the parameter list this represents.
  const CallOperands &getParameters() const { return parameters; }

  /// Add a bound value for a positional parameter binding.
  void add(const ExprNode *expr, AnyValue value) {
    add(expr, value, StringAttr());
  }
  /// Add a bound value for a keyword parameter binding. The caller is
  /// responsible for ensuring the keyword is not already present.
  void add(const ExprNode *expr, AnyValue value, StringAttr name);

  /// Append the bindings in `other` after ours and adopt its binding kind.
  void append(const ParamBindings &other);

  /// Require `constraints` to be satisfied by the bindings, in addition to the
  /// callee's own.
  void addConstraints(ArrayRef<ConstraintAttr> constraints) {
    llvm::append_range(additionalConstraints, constraints);
  }

  /// Ensure the binding kind is at least the given kind.
  void relaxBindingKindTo(BindingKind kind) {
    this->bindingKind = std::max(this->bindingKind, kind);
  }

  /// Describe how closely the given parameter bindings match the specified
  /// parameters and call operands.
  struct Fitness {
    /// Any unprovable constraints encountered during verification.
    SmallVector<ConstraintAttr> unprovableConstraints;
  };

  /// Method for debugging.
  LLVM_DUMP_METHOD void dump() const;

  BindingKind bindingKind = kStandard;

private:
  /// This contains the values that are bound into this parameter list.
  CallOperands parameters;

  /// This is a list of additional constraints that need to be satisfied by the
  /// bindings, consider:
  /// ```
  /// comptime T where condition = MyStruct
  /// _ = T()
  /// ```
  /// In order to emit the call `T()`, we need `condition` to be satisfied.
  SmallVector<ConstraintAttr> additionalConstraints;

  friend class ParamInf;
  friend class CallParamInf;
  friend TypedAttr getBoundConstAttrForFn(ASTDecl &, const ParamBindings &);
};

/// Emit diagnostics for unprovable constraints from a Fitness result.
void emitUnprovableConstraintsFromFitness(
    ArrayRef<ConstraintAttr> unprovableConstraints, SharedState &shared,
    SMLoc exprLoc, ASTDecl *declIfKnown);

/// Utility function to perform substitutions of the bindings into the symbol
/// for the given function declaration. It returns the resultant
/// SymbolConstantAttr or produces an error message and returns null.
TypedAttr getBoundConstAttrForFn(ASTDecl &fnDecl,
                                 const ParamBindings &unverifiedBindings);
TypedAttr getBoundConstAttrForFn(ASTDecl &fnDecl, SharedState &shared,
                                 const VerifiedParamBindings &verifiedBindings,
                                 ASTDecl &useScope, SMLoc useLoc);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMBINDINGS_H
