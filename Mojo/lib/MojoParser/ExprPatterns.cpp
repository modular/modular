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
// This file implements `emitMatch` for expression nodes used as match
// patterns.
//
//===----------------------------------------------------------------------===//

#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/CallOperands.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"

#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// Defined in ExprNodes.cpp; used to project tuple elements for matching.
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ExprDest &dest,
                                IREmitter &emitter);
static LogicalResult emitEnumCaseNameMatch(IREmitter &emitter, CValue subject,
                                           const ExprNode *expr,
                                           StringRef caseName,
                                           const ExprNode *typeBase);

//===----------------------------------------------------------------------===//
// Per-ExprNode Support for Matching.
//===----------------------------------------------------------------------===//

LogicalResult ExprNode::emitMatch(IREmitter &emitter, CValue subject,
                                  PatternDeclKind patternKind,
                                  SmallVectorImpl<BoundName> &bindings) const {
  emitter.emitError(getLoc(), "expression is not a valid match pattern");
  return failure();
}

/// Drive match CF from a success predicate: on mismatch take `hlcf.match.next`,
/// on match fall through via `hlcf.yield`. Known-true predicates are a no-op;
/// known-false predicates emit only `match.next`.
///
/// Dynamic form (equivalent to `elif !matches { next } else { yield }`):
///   hlcf.elif matches {
///     hlcf.yield
///   } else {
///     hlcf.match.next
///   }
static LogicalResult emitMatchOutcome(IREmitter &emitter, Location loc,
                                      CValue matches, const ExprNode *expr) {
  if (!matches)
    return failure();

  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return failure();
  }

  auto asBoolAttr = sugarDynCastIfPresent<SIMDAttr>(matches.getIfPValue());
  if (asBoolAttr) {
    if (!asBoolAttr.getAsBool())
      HLCF::MatchNextOp::create(*emitter.builder, loc);
    return success();
  }

  SRValue matchesSR = emitter.emitSRValue({matches, expr}, EC_BoolCondition);
  if (!matchesSR)
    return failure();
  HLCF::ElifOp::create(
      *emitter.builder, loc, TypeRange(), matchesSR,
      [&]() -> LogicalResult {
        HLCF::YieldOp::create(*emitter.builder, loc);
        return success();
      },
      [&]() -> LogicalResult {
        HLCF::MatchNextOp::create(*emitter.builder, loc);
        return success();
      });
  return success();
}

/// Match a literal / attribute pattern by emitting it as the subject's type
/// and comparing with `__eq__`. Mismatch advances via `hlcf.match.next`:
///
///   hlcf.elif subject != lit {   // equivalently: elif eq { yield } else
///     hlcf.match.next
///   } else {
///     hlcf.yield
///   }
static LogicalResult emitMatchAgainstValue(const ExprNode *expr,
                                           IREmitter &emitter, CValue subject) {
  // Emit this literal as a value of the subject's type, then compare.
  ExprDest litDest(subject.getRValueType(), EC_MatchSubject);
  AnyValue litValue = emitter.emitExpr(expr, litDest);
  if (!litValue)
    return failure();

  CValue eqResult = emitter.emitNamedMethodCall(
      "__eq__",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_BoolCondition),
                   {{AnyValue(subject), expr}, {litValue, expr}}));
  if (!eqResult)
    return failure();

  // Convert Bool (or other boolable) to scalar<bool> / i1 for hlcf.elif.
  CValue matches = emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition);
  return emitMatchOutcome(emitter, expr->getLocation(emitter), matches, expr);
}

LogicalResult
SimpleLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                             PatternDeclKind patternKind,
                             SmallVectorImpl<BoundName> &bindings) const {
  // `_` always succeeds and introduces no bindings.
  if (kind == kDiscardLiteral)
    return success();

  return ExprNode::emitMatch(emitter, subject, patternKind, bindings);
}

LogicalResult
BoolLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                           PatternDeclKind patternKind,
                           SmallVectorImpl<BoundName> &bindings) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult
IntLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                          PatternDeclKind patternKind,
                          SmallVectorImpl<BoundName> &bindings) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult
FloatLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                            PatternDeclKind patternKind,
                            SmallVectorImpl<BoundName> &bindings) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult
StringLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                             PatternDeclKind patternKind,
                             SmallVectorImpl<BoundName> &bindings) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult
DeclRefNode::emitMatch(IREmitter &emitter, CValue subject,
                       PatternDeclKind patternKind,
                       SmallVectorImpl<BoundName> &bindings) const {
  // Bare identifiers are only valid match patterns when nested under a `var`
  // or `ref` binding (or top-level bind mode). Do not treat them as "match
  // this existing value".
  if (patternKind == PatternDeclKind::kNone) {
    emitter.emitError(getLoc(), "bare identifier '")
        << spelling << "' is not a valid match pattern; use 'var " << spelling
        << "' or 'ref " << spelling << "' to bind a name";
    return failure();
  }

  // Binding patterns are irrefutable: declare `spelling` under the requested
  // mode and initialize it from the subject. Match subjects are borrowed
  // (BValues), so `var` bindings copy and `ref` bindings borrow — same
  // machinery as `var x = ...` / `ref x = ...` assignment. Declarations live
  // in the enclosing match case region and dominate the case body.
  ExprDest declDest(LValueInitializerType{subject.getRValueType()}, EC_VarInit);
  declDest.setPatternDeclKind(patternKind);
  LValue bindingLV = emitter.emitExprLValue(this, declDest);
  if (!bindingLV)
    return failure();

  ExprDest storeDest(bindingLV, EC_VarInit);
  if (!emitter.emitCResult(subject, this, storeDest))
    return failure();

  bindings.push_back({spelling, subject, patternKind});
  return success();
}

LogicalResult
AttributeRefNode::emitMatch(IREmitter &emitter, CValue subject,
                            PatternDeclKind patternKind,
                            SmallVectorImpl<BoundName> &bindings) const {
  // `Optional.None` / `Type.Case` against an EnumLike subject is a discriminant
  // pattern (no payload). Other attribute refs keep value-equality matching.
  ASTType subjectType = subject.getRValueType();
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumCaseNameMatch(emitter, subject, this, spelling, base);
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult InferredAttributeRefNode::emitMatch(
    IREmitter &emitter, CValue subject, PatternDeclKind patternKind,
    SmallVectorImpl<BoundName> &bindings) const {
  // `.None` / `.Case` against an EnumLike subject is a discriminant pattern.
  ASTType subjectType = subject.getRValueType();
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumCaseNameMatch(emitter, subject, this, spelling,
                                 /*typeBase=*/nullptr);
  // Resolve `.member` against the subject's type (e.g. `.red` → `Color.red`),
  // then compare for equality like other literal patterns.
  return emitMatchAgainstValue(this, emitter, subject);
}

LogicalResult ParenNode::emitMatch(IREmitter &emitter, CValue subject,
                                   PatternDeclKind patternKind,
                                   SmallVectorImpl<BoundName> &bindings) const {
  return subExpr->emitMatch(emitter, subject, patternKind, bindings);
}

LogicalResult BinOpNode::emitMatch(IREmitter &emitter, CValue subject,
                                   PatternDeclKind patternKind,
                                   SmallVectorImpl<BoundName> &bindings) const {
  if (kind == kOr)
    return emitOrMatch(emitter, subject, patternKind, bindings);
  if (kind == kAsPat)
    return emitAsMatch(emitter, subject, patternKind, bindings);
  return ExprNode::emitMatch(emitter, subject, patternKind, bindings);
}

/// Collect VarDeclOps registered in `scope`, keyed by binding name.
static void
collectPatternBindings(ASTDecl &scope,
                       SmallVectorImpl<std::pair<StringAttr, VarDeclOp>> &out) {
  for (auto &[name, decls] : scope.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto varDecl = dyn_cast_or_null<VarDeclOp>(decl->getIfOperation());
      if (!varDecl)
        continue;
      out.push_back({name, varDecl});
    }
  }
}

/// Verify LHS/RHS or-pattern alternatives bind the same names with matching
/// kinds and types, rewrite RHS stores to use the LHS VarDecls, erase the
/// duplicate RHS VarDeclOps, and promote the LHS bindings into `parentScope`.
static LogicalResult mergeOrPatternBindings(IREmitter &emitter,
                                            ASTDecl &parentScope,
                                            ASTDecl &lhsScope,
                                            ASTDecl *rhsScope, SMLoc loc) {
  SmallVector<std::pair<StringAttr, VarDeclOp>, 4> lhsBindings;
  collectPatternBindings(lhsScope, lhsBindings);

  SmallVector<std::pair<StringAttr, VarDeclOp>, 4> rhsBindings;
  if (rhsScope)
    collectPatternBindings(*rhsScope, rhsBindings);

  // No bindings on either side — nothing to promote.
  if (lhsBindings.empty() && rhsBindings.empty())
    return success();

  // RHS was never emitted (e.g. LHS constant-folded true). Promote LHS only.
  if (!rhsScope) {
    parentScope.mergeDeclsFrom(lhsScope);
    return success();
  }

  if (lhsBindings.empty() || rhsBindings.empty()) {
    StringAttr missing = lhsBindings.empty() ? rhsBindings.front().first
                                             : lhsBindings.front().first;
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << missing.getValue() << "' is bound in one alternative but "
        << "not the other";
    return failure();
  }

  llvm::DenseMap<StringAttr, VarDeclOp> rhsByName;
  for (auto &[name, varDecl] : rhsBindings)
    rhsByName[name] = varDecl;

  for (auto &[name, lhsVar] : lhsBindings) {
    auto it = rhsByName.find(name);
    if (it == rhsByName.end()) {
      emitter.emitError(loc, "or-pattern alternatives must bind the same names")
          << "; '" << name.getValue() << "' is bound in one alternative but "
          << "not the other";
      return failure();
    }
    VarDeclOp rhsVar = it->second;
    if (lhsVar.getKind() != rhsVar.getKind()) {
      emitter.emitError(loc, "or-pattern binding '")
          << name.getValue() << "' must use the same 'var'/'ref' kind in each "
          << "alternative";
      return failure();
    }
    // VarDecl types are `!lit.ref[decl] T`. Each alternative creates its own
    // decl, so the self-origin always differs even when `T` matches. Compare
    // the element types (and address space) instead.
    ASTType lhsType = lhsVar.getType().getElementType();
    ASTType rhsType = rhsVar.getType().getElementType();
    if (!lhsType.isEqualCanon(rhsType)) {
      auto diag = emitter.emitError(loc, "or-pattern binding '")
                  << name.getValue()
                  << "' has incompatible types across alternatives";
      diag.attachNote(loc) << "left alternative has type " << lhsType
                           << ", right has type " << rhsType;
      return failure();
    }

    // Both alternatives write the same name; keep the LHS VarDecl and retarget
    // RHS initializers to it.
    rhsVar.getResult().replaceAllUsesWith(lhsVar.getResult());
    rhsVar->erase();
    rhsByName.erase(it);
  }

  if (!rhsByName.empty()) {
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << rhsByName.begin()->first.getValue()
        << "' is bound in one alternative but not the other";
    return failure();
  }

  parentScope.mergeDeclsFrom(lhsScope);
  return success();
}

/// Merge BoundName sets from or-pattern alternatives into `out`. Both sides
/// must bind the same names with the same `PatternDeclKind`. The surviving
/// `CValue` currently comes from the LHS only.
static LogicalResult mergeOrPatternBoundNames(
    IREmitter &emitter, SMLoc loc, ArrayRef<ExprNode::BoundName> lhsBindings,
    ArrayRef<ExprNode::BoundName> rhsBindings, bool rhsEmitted,
    SmallVectorImpl<ExprNode::BoundName> &out) {
  if (lhsBindings.empty() && rhsBindings.empty())
    return success();

  // RHS was never emitted (e.g. LHS constant-folded true). Take LHS only.
  if (!rhsEmitted) {
    out.append(lhsBindings.begin(), lhsBindings.end());
    return success();
  }

  if (lhsBindings.empty() || rhsBindings.empty()) {
    StringRef missing = lhsBindings.empty() ? rhsBindings.front().name
                                            : lhsBindings.front().name;
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << missing << "' is bound in one alternative but not the "
        << "other";
    return failure();
  }

  llvm::StringMap<const ExprNode::BoundName *> rhsByName;
  for (const ExprNode::BoundName &bn : rhsBindings)
    rhsByName[bn.name] = &bn;

  for (const ExprNode::BoundName &lhsBN : lhsBindings) {
    auto it = rhsByName.find(lhsBN.name);
    if (it == rhsByName.end()) {
      emitter.emitError(loc, "or-pattern alternatives must bind the same names")
          << "; '" << lhsBN.name << "' is bound in one alternative but not "
          << "the other";
      return failure();
    }
    const ExprNode::BoundName &rhsBN = *it->second;
    if (lhsBN.patternKind != rhsBN.patternKind) {
      emitter.emitError(loc, "or-pattern binding '")
          << lhsBN.name
          << "' must use the same 'var'/'ref' kind in each alternative";
      return failure();
    }

    // FIXME: Merge LHS/RHS CValues (e.g. via select/phi) instead of keeping
    // only the LHS subject value.
    (void)rhsBN;
    out.push_back(lhsBN);
    rhsByName.erase(it);
  }

  if (!rhsByName.empty()) {
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << rhsByName.begin()->first()
        << "' is bound in one alternative but not the other";
    return failure();
  }
  return success();
}

LogicalResult
BinOpNode::emitOrMatch(IREmitter &emitter, CValue subject,
                       PatternDeclKind patternKind,
                       SmallVectorImpl<BoundName> &bindings) const {
  // `pat1 | pat2` matches if either alternative matches. Lowered as a nested
  // `hlcf.match` whose cases try each alternative: success completes the
  // nested match (then the enclosing case continues); failing both
  // alternatives uses `hlcf.match.next` in the else region so control
  // advances the *enclosing* match case.  This ensures that code this dominates
  // will initialize the pattern bindings in the LHS/RHS consistently on the
  // fallthrough.
  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(this);
    return failure();
  }

  auto createBindingScope = [&](SMLoc scopeLoc) -> ASTDecl & {
    return emitter.getDeclResolver().addFullyResolvedDecl(
        /*declVal=*/nullptr, StringAttr(), scopeLoc, &emitter.declScope);
  };

  // TODO: Look for other "or" patterns and merge them into a single match.
  Location loc = getLocation(emitter);
  auto matchOp = HLCF::MatchOp::create(*emitter.builder, loc, TypeRange(),
                                       /*caseRegionsCount=*/2);
  matchOp.getElseRegion().emplaceBlock();
  for (Region &region : matchOp.getCaseRegions())
    region.emplaceBlock();

  // Case 0: try the LHS alternative. `emitMatch` advances with `match.next`
  // on failure; on success fall through and complete this nested match.
  emitter.builder->setInsertionPointToStart(
      &matchOp.getCaseRegions()[0].front());
  ASTDecl &lhsScope = createBindingScope(lhs->getLoc());
  IREmitter lhsEmitter(lhsScope, *emitter.builder);
  SmallVector<BoundName, 4> lhsBindings;
  if (failed(lhs->emitMatch(lhsEmitter, subject, patternKind, lhsBindings)))
    return failure();
  emitter.builder = lhsEmitter.builder;
  HLCF::MatchCompleteOp::create(*emitter.builder, loc);

  // Case 1: try the RHS alternative.
  emitter.builder->setInsertionPointToStart(
      &matchOp.getCaseRegions()[1].front());
  ASTDecl *rhsScope = &createBindingScope(rhs->getLoc());
  SmallVector<BoundName, 4> rhsBindings;
  {
    IREmitter rhsEmitter(*rhsScope, *emitter.builder);
    if (failed(rhs->emitMatch(rhsEmitter, subject, patternKind, rhsBindings)))
      return failure();
    emitter.builder = rhsEmitter.builder;
  }
  HLCF::MatchCompleteOp::create(*emitter.builder, loc);

  // Both alternatives failed: advance the enclosing match case.
  emitter.builder->setInsertionPointToStart(&matchOp.getElseRegion().front());
  HLCF::MatchNextOp::create(*emitter.builder, loc);

  // Surviving binding decls are created inside case regions; hoist them so
  // they dominate the enclosing case body after this nested match completes.
  SmallVector<std::pair<StringAttr, VarDeclOp>, 4> lhsVarDecls;
  collectPatternBindings(lhsScope, lhsVarDecls);

  if (failed(mergeOrPatternBoundNames(emitter, getLoc(), lhsBindings,
                                      rhsBindings, /*rhsEmitted=*/true,
                                      bindings)))
    return failure();
  if (failed(mergeOrPatternBindings(emitter, emitter.declScope, lhsScope,
                                    rhsScope, getLoc())))
    return failure();

  for (auto &[name, varDecl] : lhsVarDecls) {
    (void)name;
    if (varDecl->getParentOp() == matchOp.getOperation())
      varDecl->moveBefore(matchOp);
  }

  emitter.builder->setInsertionPointAfter(matchOp);
  return success();
}

LogicalResult
BinOpNode::emitAsMatch(IREmitter &emitter, CValue subject,
                       PatternDeclKind patternKind,
                       SmallVectorImpl<BoundName> &bindings) const {
  // `pattern as name` applies `pattern` and binds `name` to the whole
  // subject without copying. Memory values use `ref`; register-passable
  // (trivial) values have no address, so they use `bind` instead.
  auto *name = dyn_cast<DeclRefNode>(rhs);
  if (!name) {
    emitter.emitError(rhs->getLoc(), "expected a name after 'as'");
    return failure();
  }

  PatternDeclKind bindKind =
      subject.isMValue() ? PatternDeclKind::kRef : PatternDeclKind::kBind;
  if (failed(name->emitMatch(emitter, subject, bindKind, bindings)))
    return failure();
  return lhs->emitMatch(emitter, subject, patternKind, bindings);
}

LogicalResult
UnaryOpNode::emitMatch(IREmitter &emitter, CValue subject,
                       PatternDeclKind patternKind,
                       SmallVectorImpl<BoundName> &bindings) const {
  // `var`/`ref` patterns are unary wrappers that set the binding mode for
  // their subpattern (e.g. `case var x:` / `case ref (a, b):`).
  if (kind != kVarPat && kind != kRefPat)
    return ExprNode::emitMatch(emitter, subject, patternKind, bindings);

  // Nested specifiers like `var ref x` are redundant; keep going with the
  // innermost kind after warning, matching assignment-pattern behavior.
  if (patternKind != PatternDeclKind::kNone &&
      patternKind != PatternDeclKind::kBind) {
    emitter.emitWarning(getLoc()) << "nested 'var' or 'ref' patterns are "
                                     "redundant, remove the outer pattern";
  }

  PatternDeclKind subKind =
      kind == kVarPat ? PatternDeclKind::kVar : PatternDeclKind::kRef;
  return subExpr->emitMatch(emitter, subject, subKind, bindings);
}

LogicalResult TupleNode::emitMatch(IREmitter &emitter, CValue subject,
                                   PatternDeclKind patternKind,
                                   SmallVectorImpl<BoundName> &bindings) const {
  ASTType subjectType = subject.getRValueType();
  ASTType tupleType = emitter.shared.lookupBuiltinType(
      "Tuple", emitter.getDeclScope(), getLoc());

  if (!tupleType.isEqualCanon(
          subjectType.getWithoutParameters(emitter.shared))) {
    emitter.emitError(getLoc(), "expected a tuple type to match against, got ")
        << subjectType << getRange();
    return failure();
  }

  assert(subjectType.getParamBindings().size() == 2 &&
         "Tuple has two parameters");
  auto vaAttr = sugarCast<ParamListAttr>(subjectType.getParamBindings()[0]);
  if (vaAttr.getValues().size() != exprs.size()) {
    emitter.emitError(getLoc(), "cannot match value of ")
        << subjectType << " of " << vaAttr.getValues().size() << " element"
        << plural(vaAttr.getValues().size()) << " against a pattern with "
        << exprs.size() << " element" << plural(exprs.size()) << getRange();
    return failure();
  }

  // Empty tuple pattern `()` always matches an empty `Tuple[]`.
  if (exprs.empty())
    return success();

  // Borrow the subject so each element access can reuse it.
  BValue subjectBVal = emitter.emitBValue({subject, this}, EC_MatchSubject);
  if (!subjectBVal)
    return failure();

  // Extract `subject[i]` the same way comptime tuple destructuring does —
  // via a synthesized subscript that prefers `__getitem_param__`.
  auto getTupleItem = [&](ASTType eltType, unsigned index) -> CValue {
    ExprDest eltDest(eltType, EC_TupleElement);
    TypedAttr indexAttr =
        IntegerAttr::get(IndexType::get(emitter.getContext()), index);
    CValue intIndexCValue =
        emitter.emitInt(ASTExprAnd<PValue>{PValue(indexAttr), this},
                        ExprContext::EC_CallParamValue);
    if (!intIndexCValue)
      return {};
    PValue intIndex = intIndexCValue.getIfPValue();
    assert(intIndex && "Int must be PValue when constructed from int attr");

    SyntheticNode indexExpr(getLoc(), intIndex);
    Operand exprOperand(&indexExpr, getLoc(), ArgUnpackStyle::kPositional);
    SubscriptNode subscript(this, this->getLoc(), {}, this->getLoc());
    auto elem = emitGetterSetterAccess(&subscript, {subjectBVal, this},
                                       exprOperand, eltDest, emitter);
    if (!elem) {
      eltDest.resetForError(emitter);
      return {};
    }
    return emitter.emitCValue({elem, this}, EC_TupleElement);
  };

  // Match elements sequentially. Earlier mismatches take `match.next` and
  // skip later element tests at runtime.
  for (unsigned i = 0, e = exprs.size(); i != e; ++i) {
    CValue eltVal = getTupleItem(ASTType(vaAttr.getValues()[i]), i);
    if (!eltVal)
      return failure();
    if (failed(exprs[i]->emitMatch(emitter, eltVal, patternKind, bindings)))
      return failure();
  }
  return success();
}

LogicalResult CallNode::emitMatch(IREmitter &emitter, CValue subject,
                                  PatternDeclKind patternKind,
                                  SmallVectorImpl<BoundName> &bindings) const {
  ASTType subjectType = subject.getRValueType();

  // `Optional.Some(ref elt)` / `Type.Case(...)`: when the subject is
  // `EnumLike`, deep-match the named case and its payload subpatterns instead
  // of treating the call as a struct field pattern.
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumMatch(emitter, subject, patternKind, bindings);

  // `Type(field=pat, ...)` is a struct pattern: the callee names the expected
  // type, and each operand is a subpattern for a stored field. Positional
  // operands bind in field-declaration order; keywords select by name.
  ASTType patternType = emitter.emitExprType(callee);
  if (!patternType)
    return failure();

  if (!patternType.isEqualCanon(subjectType)) {
    emitter.emitError(getLoc(), "cannot match value of type ")
        << subjectType << " against pattern type " << patternType << getRange();
    return failure();
  }

  auto structType =
      dyn_cast<StructType>(SugarAttr::strip(subjectType.mlirType));
  if (!structType) {
    emitter.emitError(getLoc(), "expected a struct type to match against, got ")
        << subjectType << getRange();
    return failure();
  }

  ASTDecl *typeDecl = subjectType.getDecl(emitter.shared);
  if (!typeDecl) {
    emitter.emitError(getLoc(), "cannot match fields of ")
        << subjectType << getRange();
    return failure();
  }

  SmallVector<StructFieldOp, 8> storedFields;
  if (auto structDecl =
          dyn_cast_or_null<StructDeclOp>(typeDecl->getIfOperation())) {
    for (auto field : structDecl.getFieldDecls())
      storedFields.push_back(field);
  }

  // Resolve every operand to a field first so unknown/duplicate names are
  // diagnosed even when an earlier subpattern is statically false.
  struct FieldPattern {
    const Operand *operand;
    StructFieldOp fieldOp;
  };
  SmallVector<FieldPattern, 8> fieldPatterns;
  llvm::SmallPtrSet<Attribute, 8> seenFields;

  for (const Operand &operand : operands) {
    if (!operand.isKeyword()) {
      emitter.emitError(
          operand.getLoc(),
          "struct patterns do not support positional or unpacked arguments")
          << operand.expr->getRange();
      return failure();
    }

    StringAttr fieldName = operand.name;
    LookupResult lookup = emitter.shared.lookupAndResolveDecl(
        fieldName.getValue(), operand.getLoc(), *typeDecl,
        /*searchParentScopes=*/false);
    if (lookup.isErroneous())
      return failure();
    if (!lookup.isSuccess() || lookup.getIfSuccess().size() != 1) {
      emitter.emitError(operand.getLoc(), "'")
          << fieldName.getValue() << "' is not a field of " << subjectType
          << operand.expr->getRange();
      return failure();
    }
    auto fieldOp = dyn_cast_or_null<StructFieldOp>(
        lookup.getIfSuccess().front()->getIfOperation());
    if (!fieldOp) {
      emitter.emitError(operand.getLoc(), "'")
          << fieldName.getValue() << "' is not a stored field of "
          << subjectType << operand.expr->getRange();
      return failure();
    }

    if (!seenFields.insert(fieldName).second) {
      emitter.emitError(operand.getLoc(), "duplicate field '")
          << fieldName.getValue() << "' in struct pattern"
          << operand.expr->getRange();
      return failure();
    }
    fieldPatterns.push_back({&operand, fieldOp});
  }

  if (fieldPatterns.empty())
    return success();

  BValue subjectBVal = emitter.emitBValue({subject, this}, EC_MatchSubject);
  if (!subjectBVal)
    return failure();

  // Match fields sequentially. Earlier mismatches take `match.next`.
  for (const FieldPattern &fp : fieldPatterns) {
    StructFieldOp fieldOp = fp.fieldOp;
    ASTType fieldType = fieldOp.getReboundType(
        structType, &emitter.shared.getEvaluationContext());
    ExprDest fieldDest(fieldType, EC_AttributeRefBase);
    CValue fieldVal = AttributeRefNode::emitStoredFieldRef(
        {subjectBVal, this}, fieldOp, fp.operand->expr, fieldDest, emitter);
    if (!fieldVal)
      return failure();
    if (failed(fp.operand->expr->emitMatch(emitter, fieldVal, patternKind,
                                           bindings)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EnumLike Matching.
//===----------------------------------------------------------------------===//

/// When processing `Type.Case` patterns, require them to be the subject's
/// nominal type.  Allow unbound types like `Optional` to match `Optional[Int]`.
/// Return failure if we emit an error.
static LogicalResult checkEnumCaseTypeBase(IREmitter &emitter, CValue subject,
                                           const ExprNode *typeBase,
                                           const ExprNode *expr) {
  ASTType baseType = emitter.emitExprType(typeBase, /*allowUnbound=*/true);
  if (!baseType)
    return failure();
  ASTType baseNominal = baseType.getWithoutParameters(emitter.shared);
  ASTType subjectNominal =
      subject.getRValueType().getWithoutParameters(emitter.shared);
  if (baseNominal.isEqualCanon(subjectNominal))
    return success();
  emitter.emitError(expr->getLoc(), "cannot match value of type ")
      << subject.getRValueType() << " against enum case of type " << baseType
      << expr->getRange();
  return failure();
}

/// Look up `caseName` in `SubjectType._enum_case_names`. Returns nullopt after
/// emitting an error when the name is not a case.
static std::optional<size_t> lookupEnumCaseIndex(IREmitter &emitter,
                                                 ASTType subjectType,
                                                 StringRef caseName,
                                                 const ExprNode *expr) {
  SyntheticNode typeNode(expr->getLoc(), PValue(subjectType));
  AttributeRefNode namesRef(&typeNode, expr->getLoc(), "_enum_case_names");
  PValue namesPV = emitter.emitExprPValue(&namesRef, EC_AttributeRefBase);
  if (!namesPV)
    return std::nullopt;

  auto namesList = dyn_cast<ParamListAttr>(getCanonicalAttr(namesPV.get()));
  if (!namesList) {
    emitter.emitError(expr->getLoc(), "cannot match on a parametric enum type")
        << expr->getRange();
    return std::nullopt;
  }

  for (auto [idx, nameAttr] : llvm::enumerate(namesList.getValues())) {
    auto nameStr = dyn_cast<StringAttr>(nameAttr);
    if (nameStr && nameStr.getValue() == caseName)
      return idx;
  }
  emitter.emitError(expr->getLoc(), "'")
      << caseName << "' is not a case of " << subjectType << expr->getRange();
  return std::nullopt;
}

/// Return the payload type for case `caseIndex` from `_enum_case_types`. This
/// returns null if an error is emitted.
static ASTType getEnumCasePayloadType(IREmitter &emitter, ASTType subjectType,
                                      unsigned caseIndex,
                                      const ExprNode *expr) {
  SyntheticNode typeNode(expr->getLoc(), PValue(subjectType));
  AttributeRefNode typesRef(&typeNode, expr->getLoc(), "_enum_case_types");
  PValue typesPV = emitter.emitExprPValue(&typesRef, EC_AttributeRefBase);
  if (!typesPV)
    return {};

  auto typesList = dyn_cast<ParamListAttr>(getCanonicalAttr(typesPV.get()));
  if (!typesList || caseIndex >= typesList.getValues().size()) {
    emitter.emitError(expr->getLoc(),
                      "'_enum_case_types' must be a parameter list covering "
                      "every case")
        << expr->getRange();
    return {};
  }

  TypedAttr payloadAttr = typesList.getValues()[caseIndex];
  if (!LIT::isTypeExpr(payloadAttr)) {
    emitter.emitError(expr->getLoc(),
                      "'_enum_case_types' elements must be types")
        << expr->getRange();
    return {};
  }
  return ASTType(payloadAttr);
}

/// True when the case payload is Mojo `NoneType` (no associated value).
static bool isEnumCaseWithoutPayload(IREmitter &emitter, ASTType payloadType,
                                     const ExprNode *expr) {
  if (payloadType.isNoneType())
    return true;
  ASTType noneType = emitter.shared.lookupBuiltinType(
      "NoneType", emitter.getDeclScope(), expr->getLoc());
  if (!noneType)
    return false;
  return payloadType.getWithoutParameters(emitter.shared)
      .isEqualCanon(noneType.getWithoutParameters(emitter.shared));
}

/// Emit `_get_enum_discriminant() == caseIndex` as a scalar bool predicate.
/// This returns the bool result as well as the case number as an Int.
static std::pair<CValue, CValue>
emitEnumDiscriminantMatch(IREmitter &emitter, CValue subject,
                          unsigned caseIndex, const ExprNode *expr) {
  BValue subjectBVal = emitter.emitBValue({subject, expr}, EC_MatchSubject);
  if (!subjectBVal)
    return {{}, {}};

  CValue discriminant = emitter.emitNamedMethodCall(
      "_get_enum_discriminant",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_MatchSubject),
                   {{AnyValue(subjectBVal), expr}}));
  if (!discriminant)
    return {{}, {}};

  TypedAttr indexAttr =
      IntegerAttr::get(IndexType::get(emitter.getContext()), caseIndex);
  CValue caseIdxInt = emitter.emitInt(
      ASTExprAnd<PValue>{PValue(indexAttr), expr}, EC_CallParamValue);
  if (!caseIdxInt)
    return {{}, {}};

  CValue eqResult = emitter.emitNamedMethodCall(
      "__eq__",
      CallOperands(
          CallSyntax::kMethodCall, expr, ExprDest(EC_BoolCondition),
          {{AnyValue(discriminant), expr}, {AnyValue(caseIdxInt), expr}}));
  if (!eqResult)
    return {{}, {}};

  return {emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition),
          caseIdxInt};
}

/// Match `Type.Case` / `.Case` (no parentheses) against an EnumLike subject.
/// "expr" may be either an AttributeRefNode or an InferredAttributeRefNode.
/// typeBase is null in the later case.
static LogicalResult emitEnumCaseNameMatch(IREmitter &emitter, CValue subject,
                                           const ExprNode *expr,
                                           StringRef caseName,
                                           const ExprNode *typeBase) {
  if (typeBase &&
      failed(checkEnumCaseTypeBase(emitter, subject, typeBase, expr)))
    return failure();
  std::optional<size_t> caseIndex =
      lookupEnumCaseIndex(emitter, subject.getRValueType(), caseName, expr);
  if (!caseIndex)
    return failure();

  CValue discMatch =
      emitEnumDiscriminantMatch(emitter, subject, *caseIndex, expr).first;
  return emitMatchOutcome(emitter, expr->getLocation(emitter), discMatch, expr);
}

//===----------------------------------------------------------------------===//
// EnumLike call patterns (`Type.Case(payload)`).
//===----------------------------------------------------------------------===//

LogicalResult
CallNode::emitEnumMatch(IREmitter &emitter, CValue subject,
                        PatternDeclKind patternKind,
                        SmallVectorImpl<BoundName> &bindings) const {
  // `Optional.Some(ref elt)` / `.Some(pat)`: call form carries payload
  // subpatterns. Cases with no associated value must use `Optional.None` /
  // `.None` without parentheses.
  StringRef caseName;
  if (auto *attr = dyn_cast<AttributeRefNode>(callee)) {
    caseName = attr->spelling;
    if (failed(checkEnumCaseTypeBase(emitter, subject, attr->base, this)))
      return failure();
  } else if (auto *inferred = dyn_cast<InferredAttributeRefNode>(callee)) {
    caseName = inferred->spelling;
  } else {
    emitter.emitError(getLoc(),
                      "enum case pattern must be written as 'Type.Case(...)' "
                      "or '.Case(...)'")
        << callee->getRange();
    return failure();
  }

  // Figure out what case we're matching against, and the payload type.
  ASTType subjectType = subject.getRValueType();
  auto caseIndex = lookupEnumCaseIndex(emitter, subjectType, caseName, this);
  if (!caseIndex)
    return failure();
  ASTType payloadType =
      getEnumCasePayloadType(emitter, subjectType, *caseIndex, this);
  if (!payloadType)
    return failure();

  // Reject attempts to pattern match on a None case.
  if (isEnumCaseWithoutPayload(emitter, payloadType, this)) {
    emitter.emitError(getLoc(), "enum case '")
        << caseName << "' has no associated value" << getParenRange();
    return failure();
  }

  // Empty `Type.Case()` is never valid: no-payload cases omit parentheses,
  // and payload cases need a subpattern.
  if (operands.empty()) {
    emitter.emitError(getLoc(), "enum case '")
        << caseName << "' requires a payload pattern inside the parentheses"
        << getParenRange();
    return failure();
  }

  // Reject unsupported unpacking and keyword arguments.
  for (const Operand &operand : operands) {
    if (operand.unpackStyle == ArgUnpackStyle::kKeyword) {
      emitter.emitError(operand.getLoc(),
                        "enum case patterns do not support keyword arguments")
          << operand.expr->getRange();
      return failure();
    }
    if (operand.unpackStyle != ArgUnpackStyle::kPositional) {
      emitter.emitError(operand.getLoc(),
                        "enum case patterns do not support unpacked arguments")
          << operand.expr->getRange();
      return failure();
    }
  }

  // A single subpattern matches the whole payload.
  // TODO: Support .Case(a, b) as a nested tuple pattern.
  if (operands.size() != 1) {
    emitter.emitError(operands[1].getLoc(),
                      "enum case patterns currently support at most one "
                      "payload subpattern")
        << operands[1].expr->getRange();
    return failure();
  }

  // Check the discriminant; on mismatch `match.next` skips payload extraction.
  auto [discMatch, caseIdxInt] =
      emitEnumDiscriminantMatch(emitter, subject, *caseIndex, this);
  if (!discMatch || !caseIdxInt)
    return failure();
  if (failed(emitMatchOutcome(emitter, getLocation(emitter), discMatch, this)))
    return failure();

  // Extract the payload and deep-match the subpattern.
  SyntheticNode subjectNode(getLoc(), subject);
  AttributeRefNode payloadMethod(&subjectNode, getLoc(),
                                 "_unsafe_get_enum_payload");
  SyntheticNode indexNode(getLoc(), caseIdxInt);
  Operand indexOperand(&indexNode, getLoc(), ArgUnpackStyle::kPositional);
  SubscriptNode subscript(&payloadMethod, getLoc(), indexOperand, getLoc());
  CallNode payloadCall(&subscript, getLoc(), /*operands=*/{}, getLoc());
  CValue payload = emitter.emitExprCValue(&payloadCall, EC_MatchSubject);
  if (!payload)
    return failure();

  return operands[0].expr->emitMatch(emitter, payload, patternKind, bindings);
}
