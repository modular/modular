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
// This file defines the context type used while emitting a signature to defer
// unprovable body constraints instead of raising a hard error, and to declare
// implicit parameters for `_` holes as they are resolved.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H
#define KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

#include <optional>
#include <utility>

namespace M::KGEN::LIT {

class ASTDecl;
using llvm::SMLoc;

/// An unprovable constraint that was deferred during emission. The caller that
/// installed the context is responsible for proving (or diagnosing) these
/// constraints at a later point.
struct DeferredConstraint {
  /// The unprovable body constraint. Its own `getLoc()` points at the `where`
  /// clause that declared the constraint.
  ConstraintAttr constraint;
  /// Source location of the emission site that produced this deferred
  /// constraint (the call expression, struct-binding expression, etc.).
  SMLoc deferralLoc;
};

/// Declares and manages implicit parameters declared for `_` holes
/// (auto-parameterization).
class ImplicitParamManager {
public:
  /// An implicit parameter declared for a hole.
  struct PendingParam {
    ParamDeclAttr decl;
    SMLoc loc;
  };

  ImplicitParamManager(ASTDecl &declScope, StringAttr prefix)
      : declScope(declScope), prefix(prefix) {}

  /// Declare an implicit parameter and record it as pending.
  ParamDeclAttr declare(Type type, StringRef name, SMLoc loc);

  size_t getNumPending() const { return pending.size(); }

  /// Drop the pending implicit parameters declared after the first `numParams`.
  void truncate(size_t numParams) { pending.truncate(numParams); }

  /// Return the pending implicit parameters, to be added to the parameter list.
  SmallVector<PendingParam> takePending() { return std::exchange(pending, {}); }

private:
  /// The declaration under which implicit parameters will be declared. Implicit
  /// parameter names are mangled against it.
  ASTDecl &declScope;
  /// Implicit parameters are named `<prefix>.<param>`.
  StringAttr prefix;

  /// Implicit parameters declared so far, in declaration order.
  SmallVector<PendingParam> pending;
};

/// Context for deferred typing decisions during the IR emission of one part of
/// a signature, such as an argument's type. A `DeferredTypingScope` owns it for
/// the duration of that emission.
class DeferredTypingContext {
public:
  /// Record a constraint that is inconclusive during emission.
  void addInconclusive(ConstraintAttr constraint, SMLoc loc) {
    deferredConstraints.push_back({constraint, loc});
  }

  bool canDeclareImplicitParams() const { return implicitParams.has_value(); }

  ParamDeclAttr declareImplicitParam(Type type, StringRef name, SMLoc loc) {
    assert(canDeclareImplicitParams() && "implicit parameters are disabled");
    return implicitParams->declare(type, name, loc);
  }

  /// Sizes of the recorded state, to roll back to.
  struct Checkpoint {
    size_t numDeferredConstraints = 0;
    size_t numImplicitParams = 0;
  };

  Checkpoint getCheckpoint() const {
    return {deferredConstraints.size(),
            implicitParams ? implicitParams->getNumPending() : 0};
  }

  /// Drop everything recorded since `checkpoint`, e.g. when the binding that
  /// recorded it failed.
  void rollback(const Checkpoint &checkpoint) {
    deferredConstraints.truncate(checkpoint.numDeferredConstraints);
    if (implicitParams)
      implicitParams->truncate(checkpoint.numImplicitParams);
  }

private:
  friend class TypeCheckedParamList;

  /// Unprovable constraints deferred during emission.
  SmallVector<DeferredConstraint> deferredConstraints;

  /// Present if holes may be declared as implicit parameters.
  std::optional<ImplicitParamManager> implicitParams;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H
