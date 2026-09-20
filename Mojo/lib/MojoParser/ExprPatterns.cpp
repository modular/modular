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
// This file implements match-pattern lowering: a command-list / access-path IR
// (`buildCheckList`) and emission via `PatternEmitState`.
//
//===----------------------------------------------------------------------===//

#include "PatternMatchIR.h"

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
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// PatternMatchBuilder
//===----------------------------------------------------------------------===//

PatternMatchBuilder::PatternMatchBuilder(ASTDecl &declScope,
                                         ExprContext paramContext)
    : declScope(declScope), paramContext(paramContext),
      shared(declScope.getShared()) {}

IREmitter PatternMatchBuilder::getParamEmitter() {
  return IREmitter(declScope, paramContext);
}

MojoInflightDiag PatternMatchBuilder::emitError(SMLoc loc,
                                                const Twine &message) {
  return shared.emitError(loc, message);
}

MojoInflightDiag PatternMatchBuilder::emitWarning(SMLoc loc,
                                                  const Twine &message) {
  return shared.emitWarning(loc, message);
}

//===----------------------------------------------------------------------===//
// Pattern command-list dumping
//===----------------------------------------------------------------------===//

static StringRef stringifyPatternDeclKind(PatternDeclKind kind) {
  switch (kind) {
  case PatternDeclKind::kNone:
    return "none";
  case PatternDeclKind::kVar:
    return "var";
  case PatternDeclKind::kRef:
    return "ref";
  case PatternDeclKind::kBind:
    return "bind";
  }
  llvm_unreachable("unknown PatternDeclKind");
}

void PatternPath::print(raw_ostream &os) const {
  SmallVector<const PatternPath *, 4> chain;
  for (const PatternPath *p = this; p; p = p->parent)
    chain.push_back(p);
  std::reverse(chain.begin(), chain.end());

  for (const PatternPath *p : chain) {
    switch (p->kind) {
    case Root:
      os << "$";
      break;
    case TupleElement:
      os << "[" << p->index << "]";
      break;
    case StructField:
      os << "." << p->fieldName.getValue();
      break;
    case EnumPayload:
      os << "#payload(" << p->index << ")";
      break;
    }
  }
  os << " : " << type;
}

void PatternPath::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void PatternCommand::print(raw_ostream &os, unsigned indent) const {
  os.indent(indent);
  switch (kind) {
  case Equal:
    os << "equal ";
    if (path)
      path->print(os);
    else
      os << "<null-path>";
    break;
  case EnumTag:
    os << "enum_tag ";
    if (path)
      path->print(os);
    else
      os << "<null-path>";
    os << " case=" << enumCaseIndex;
    break;
  case Bind:
    os << "bind " << stringifyPatternDeclKind(declKind) << " " << bindName
       << " at ";
    if (path)
      path->print(os);
    else
      os << "<null-path>";
    break;
  case Or:
    os << "or ";
    if (path)
      path->print(os);
    else
      os << "<null-path>";
    os << " {\n";
    for (auto [altIdx, alt] : llvm::enumerate(orAlternatives)) {
      os.indent(indent + 2) << "alt #" << altIdx << ":\n";
      unsigned altIndent = indent + 4;
      if (alt.empty()) {
        os.indent(altIndent) << "<empty>\n";
        continue;
      }
      for (const PatternCommand *cmd : alt) {
        if (cmd)
          cmd->print(os, altIndent);
        else
          os.indent(altIndent) << "<null-command>\n";
      }
    }
    os.indent(indent) << "}";
    break;
  }
  os << "\n";
}

void PatternCommand::dump() const { print(llvm::errs()); }
//===----------------------------------------------------------------------===//
// Per-ExprNode Support for Matching.
//===----------------------------------------------------------------------===//

LogicalResult
ExprNode::buildCheckList(PatternMatchBuilder &builder, CValue subject,
                         const PatternPath *path,
                         SmallVectorImpl<const PatternCommand *> &out) const {
  builder.emitError(getLoc(), "expression is not a valid match pattern");
  return failure();
}

StringRef ExprNode::getLiteralSpelling() const { return {}; }

StringRef IntLiteralNode::getLiteralSpelling() const { return spelling; }

StringRef FloatLiteralNode::getLiteralSpelling() const { return spelling; }

StringRef BoolLiteralNode::getLiteralSpelling() const {
  return value ? "True" : "False";
}

StringRef StringLiteralNode::getLiteralSpelling() const {
  // Only a single source token has a stable spelling StringRef; concatenated
  // literals fall back to pointer equality at the grouping site.
  return spellings.size() == 1 ? spellings.front() : StringRef();
}

LogicalResult SimpleLiteralNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  // `_` is irrefutable: no check, no binding.
  if (kind == kDiscardLiteral)
    return success();
  return ExprNode::buildCheckList(builder, subject, path, out);
}

LogicalResult BoolLiteralNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  out.push_back(builder.createEqual(path, this));
  return success();
}

LogicalResult IntLiteralNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  out.push_back(builder.createEqual(path, this));
  return success();
}

LogicalResult FloatLiteralNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  out.push_back(builder.createEqual(path, this));
  return success();
}

LogicalResult StringLiteralNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  out.push_back(builder.createEqual(path, this));
  return success();
}

LogicalResult DeclRefNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  PatternDeclKind patternKind = builder.getPatternKind();
  if (patternKind == PatternDeclKind::kNone) {
    builder.emitError(getLoc(), "bare identifier '")
        << spelling << "' is not a valid match pattern; use 'var " << spelling
        << "' or 'ref " << spelling << "' to bind a name";
    return failure();
  }
  out.push_back(builder.createBind(path, this, spelling, patternKind));
  return success();
}

LogicalResult
ParenNode::buildCheckList(PatternMatchBuilder &builder, CValue subject,
                          const PatternPath *path,
                          SmallVectorImpl<const PatternCommand *> &out) const {
  return subExpr->buildCheckList(builder, subject, path, out);
}

// If this is the root of (1|2)|(3|4), dig out all the alternatives to
// generate a single structure (reducing IR bloat).
static void
addOrPatternAlternatives(SmallVectorImpl<const ExprNode *> &alternatives,
                         const ExprNode *node) {
  if (auto *orNode = dyn_cast<BinOpNode>(node);
      orNode && orNode->kind == ExprNode::kOr) {
    addOrPatternAlternatives(alternatives, orNode->lhs);
    addOrPatternAlternatives(alternatives, orNode->rhs);
  } else if (auto *parenNode = dyn_cast<ParenNode>(node)) {
    addOrPatternAlternatives(alternatives, parenNode->subExpr);
  } else {
    alternatives.push_back(node);
  }
}

LogicalResult
BinOpNode::buildCheckList(PatternMatchBuilder &builder, CValue subject,
                          const PatternPath *path,
                          SmallVectorImpl<const PatternCommand *> &out) const {
  if (kind == kOr)
    return buildOrCheckList(builder, subject, path, out);
  if (kind == kAsPat)
    return buildAsCheckList(builder, subject, path, out);
  return ExprNode::buildCheckList(builder, subject, path, out);
}

LogicalResult BinOpNode::buildOrCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  SmallVector<const ExprNode *, 2> alternatives;
  addOrPatternAlternatives(alternatives, this);

  // Each alternative is its own command list (including Bind steps). Binding
  // agreement across arms is verified when the or is emitted, not here.
  SmallVector<PatternCommandList, 2> altLists;
  altLists.reserve(alternatives.size());
  for (const ExprNode *alternative : alternatives) {
    SmallVector<const PatternCommand *, 8> altCmds;
    if (failed(alternative->buildCheckList(builder, subject, path, altCmds)))
      return failure();
    altLists.push_back(builder.internCommandList(altCmds));
  }

  out.push_back(builder.createOr(path, this, altLists));
  return success();
}

LogicalResult BinOpNode::buildAsCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  auto *name = dyn_cast<DeclRefNode>(rhs);
  if (!name) {
    builder.emitError(rhs->getLoc(), "expected a name after 'as'");
    return failure();
  }

  PatternDeclKind bindKind = PatternDeclKind::kRef;
  if (subject)
    bindKind =
        subject.isMValue() ? PatternDeclKind::kRef : PatternDeclKind::kBind;
  {
    llvm::SaveAndRestore restoreKind(builder.patternKindRef());
    builder.setPatternKind(bindKind);
    if (failed(name->buildCheckList(builder, subject, path, out)))
      return failure();
  }
  return lhs->buildCheckList(builder, subject, path, out);
}

LogicalResult UnaryOpNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  if (kind != kVarPat && kind != kRefPat)
    return ExprNode::buildCheckList(builder, subject, path, out);

  if (builder.getPatternKind() != PatternDeclKind::kNone &&
      builder.getPatternKind() != PatternDeclKind::kBind) {
    builder.emitWarning(getLoc()) << "nested 'var' or 'ref' patterns are "
                                     "redundant, remove the outer pattern";
  }

  llvm::SaveAndRestore restoreKind(builder.patternKindRef());
  builder.setPatternKind(kind == kVarPat ? PatternDeclKind::kVar
                                         : PatternDeclKind::kRef);
  return subExpr->buildCheckList(builder, subject, path, out);
}

LogicalResult
TupleNode::buildCheckList(PatternMatchBuilder &builder, CValue subject,
                          const PatternPath *path,
                          SmallVectorImpl<const PatternCommand *> &out) const {
  ASTType subjectType = path->type;
  ASTType tupleType =
      builder.shared.lookupBuiltinType("Tuple", builder.declScope, getLoc());

  if (!tupleType.isEqualCanon(
          subjectType.getWithoutParameters(builder.shared))) {
    builder.emitError(getLoc(), "expected a tuple type to match against, got ")
        << subjectType << getRange();
    return failure();
  }

  assert(subjectType.getParamBindings().size() == 2 &&
         "Tuple has two parameters");
  auto vaAttr = sugarCast<ParamListAttr>(subjectType.getParamBindings()[0]);
  if (vaAttr.getValues().size() != exprs.size()) {
    builder.emitError(getLoc(), "cannot match value of ")
        << subjectType << " of " << vaAttr.getValues().size() << " element"
        << plural(vaAttr.getValues().size()) << " against a pattern with "
        << exprs.size() << " element" << plural(exprs.size()) << getRange();
    return failure();
  }

  if (exprs.empty())
    return success();

  for (unsigned i = 0, e = exprs.size(); i != e; ++i) {
    ASTType eltType = ASTType(vaAttr.getValues()[i]);
    const PatternPath *eltPath = builder.getTupleElement(path, i, eltType);
    // Nested subjects are not projected yet; subpatterns use path types.
    if (failed(exprs[i]->buildCheckList(builder, CValue(), eltPath, out)))
      return failure();
  }
  return success();
}

LogicalResult
CallNode::buildCheckList(PatternMatchBuilder &builder, CValue subject,
                         const PatternPath *path,
                         SmallVectorImpl<const PatternCommand *> &out) const {
  ASTType subjectType = path->type;

  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               builder.shared, {}))
    return buildEnumCheckList(builder, subject, path, out);

  IREmitter emitter = builder.getParamEmitter();
  ASTType patternType = emitter.emitExprType(callee);
  if (!patternType)
    return failure();

  if (!patternType.isEqualCanon(subjectType)) {
    builder.emitError(getLoc(), "cannot match value of type ")
        << subjectType << " against pattern type " << patternType << getRange();
    return failure();
  }

  auto structType =
      dyn_cast<StructType>(SugarAttr::strip(subjectType.mlirType));
  if (!structType) {
    builder.emitError(getLoc(), "expected a struct type to match against, got ")
        << subjectType << getRange();
    return failure();
  }

  ASTDecl *typeDecl = subjectType.getDecl(builder.shared);
  if (!typeDecl) {
    builder.emitError(getLoc(), "cannot match fields of ")
        << subjectType << getRange();
    return failure();
  }

  struct FieldPattern {
    const Operand *operand;
    StructFieldOp fieldOp;
  };
  SmallVector<FieldPattern, 8> fieldPatterns;
  llvm::SmallPtrSet<Attribute, 8> seenFields;

  for (const Operand &operand : operands) {
    if (!operand.isKeyword()) {
      builder.emitError(
          operand.getLoc(),
          "struct patterns do not support positional or unpacked arguments")
          << operand.expr->getRange();
      return failure();
    }

    StringAttr fieldName = operand.name;
    LookupResult lookup = builder.shared.lookupAndResolveDecl(
        fieldName.getValue(), operand.getLoc(), *typeDecl,
        /*searchParentScopes=*/false);
    if (lookup.isErroneous())
      return failure();
    if (!lookup.isSuccess() || lookup.getIfSuccess().size() != 1) {
      builder.emitError(operand.getLoc(), "'")
          << fieldName.getValue() << "' is not a field of " << subjectType
          << operand.expr->getRange();
      return failure();
    }
    auto fieldOp = dyn_cast_or_null<StructFieldOp>(
        lookup.getIfSuccess().front()->getIfOperation());
    if (!fieldOp) {
      builder.emitError(operand.getLoc(), "'")
          << fieldName.getValue() << "' is not a stored field of "
          << subjectType << operand.expr->getRange();
      return failure();
    }

    if (!seenFields.insert(fieldName).second) {
      builder.emitError(operand.getLoc(), "duplicate field '")
          << fieldName.getValue() << "' in struct pattern"
          << operand.expr->getRange();
      return failure();
    }
    fieldPatterns.push_back({&operand, fieldOp});
  }

  for (const FieldPattern &fp : fieldPatterns) {
    StructFieldOp fieldOp = fp.fieldOp;
    ASTType fieldType = fieldOp.getReboundType(
        structType, &builder.shared.getEvaluationContext());
    const PatternPath *fieldPath =
        builder.getStructField(path, fp.operand->name, fieldType);
    if (failed(fp.operand->expr->buildCheckList(builder, CValue(), fieldPath,
                                                out)))
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
                                      size_t caseIndex, const ExprNode *expr) {
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

/// Given a value of EnumLike type, extract the discriminant from it.
static CValue emitGetEnumDiscriminant(IREmitter &emitter, CValue subject,
                                      const ExprNode *expr) {
  BValue subjectBVal = emitter.emitBValue({subject, expr}, EC_MatchSubject);
  if (!subjectBVal)
    return {};
  return emitter.emitNamedMethodCall(
      "_get_enum_discriminant",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_MatchSubject),
                   {{AnyValue(subjectBVal), expr}}));
}

LogicalResult CallNode::buildEnumCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  IREmitter emitter = builder.getParamEmitter();
  StringRef caseName;
  if (auto *attr = dyn_cast<AttributeRefNode>(callee)) {
    caseName = attr->spelling;
    if (subject &&
        failed(checkEnumCaseTypeBase(emitter, subject, attr->base, this)))
      return failure();
  } else if (auto *inferred = dyn_cast<InferredAttributeRefNode>(callee)) {
    caseName = inferred->spelling;
  } else {
    builder.emitError(getLoc(),
                      "enum case pattern must be written as 'Type.Case(...)' "
                      "or '.Case(...)'")
        << callee->getRange();
    return failure();
  }

  ASTType subjectType = path->type;
  auto caseIndex = lookupEnumCaseIndex(emitter, subjectType, caseName, this);
  if (!caseIndex)
    return failure();
  ASTType payloadType =
      getEnumCasePayloadType(emitter, subjectType, *caseIndex, this);
  if (!payloadType)
    return failure();

  if (isEnumCaseWithoutPayload(emitter, payloadType, this)) {
    builder.emitError(getLoc(), "enum case '")
        << caseName << "' has no associated value" << getParenRange();
    return failure();
  }

  if (operands.empty()) {
    builder.emitError(getLoc(), "enum case '")
        << caseName << "' requires a payload pattern inside the parentheses"
        << getParenRange();
    return failure();
  }

  for (const Operand &operand : operands) {
    if (operand.unpackStyle == ArgUnpackStyle::kKeyword) {
      builder.emitError(operand.getLoc(),
                        "enum case patterns do not support keyword arguments")
          << operand.expr->getRange();
      return failure();
    }
    if (operand.unpackStyle != ArgUnpackStyle::kPositional) {
      builder.emitError(operand.getLoc(),
                        "enum case patterns do not support unpacked arguments")
          << operand.expr->getRange();
      return failure();
    }
  }

  if (operands.size() != 1) {
    builder.emitError(operands[1].getLoc(),
                      "enum case patterns currently support at most one "
                      "payload subpattern")
        << operands[1].expr->getRange();
    return failure();
  }

  out.push_back(builder.createEnumTag(path, this, *caseIndex));
  const PatternPath *payloadPath =
      builder.getEnumPayload(path, *caseIndex, payloadType);
  return operands[0].expr->buildCheckList(builder, CValue(), payloadPath, out);
}

LogicalResult AttributeRefNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  ASTType subjectType = path->type;
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               builder.shared, {})) {
    IREmitter emitter = builder.getParamEmitter();
    if (subject && failed(checkEnumCaseTypeBase(emitter, subject, base, this)))
      return failure();
    auto caseIndex = lookupEnumCaseIndex(emitter, subjectType, spelling, this);
    if (!caseIndex)
      return failure();
    out.push_back(builder.createEnumTag(path, this, *caseIndex));
    return success();
  }
  out.push_back(builder.createEqual(path, this));
  return success();
}

LogicalResult InferredAttributeRefNode::buildCheckList(
    PatternMatchBuilder &builder, CValue subject, const PatternPath *path,
    SmallVectorImpl<const PatternCommand *> &out) const {
  ASTType subjectType = path->type;
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               builder.shared, {})) {
    IREmitter emitter = builder.getParamEmitter();
    auto caseIndex = lookupEnumCaseIndex(emitter, subjectType, spelling, this);
    if (!caseIndex)
      return failure();
    out.push_back(builder.createEnumTag(path, this, *caseIndex));
    return success();
  }
  out.push_back(builder.createEqual(path, this));
  return success();
}

//===----------------------------------------------------------------------===//
// PatternCommandList emission
//===----------------------------------------------------------------------===//

CValue PatternEmitState::getPathValue(OpBuilder &builder,
                                      const PatternPath *path,
                                      const ExprNode *expr) {
  assert(path && "null pattern path");
  if (auto it = pathValues.find(path); it != pathValues.end())
    return it->second;

  if (path->kind == PatternPath::Root) {
    assert(path == rootPath && "unexpected root path");
    return pathValues[path] = rootSubject;
  }

  CValue parent = getPathValue(builder, path->parent, expr);
  if (!parent)
    return {};

  IREmitter emitter(curDeclScope, builder);

  switch (path->kind) {
  case PatternPath::Root:
    llvm_unreachable("handled above");
  case PatternPath::TupleElement: {
    // Emit `parent[idx]` as a subscript expression.
    TypedAttr indexAttr =
        IntegerAttr::get(IndexType::get(emitter.getContext()), path->index);
    CValue intIndex = emitter.emitInt(
        ASTExprAnd<PValue>{PValue(indexAttr), expr}, EC_CallParamValue);
    if (!intIndex)
      return {};
    SyntheticNode baseNode(expr->getLoc(), parent);
    SyntheticNode indexNode(expr->getLoc(), intIndex);
    Operand indexOperand(&indexNode, expr->getLoc(),
                         ArgUnpackStyle::kPositional);
    SubscriptNode subscript(&baseNode, expr->getLoc(), indexOperand,
                            expr->getLoc());
    CValue elt = emitter.emitExprCValue(&subscript, EC_TupleElement);
    return pathValues[path] = elt;
  }
  case PatternPath::StructField: {
    // Synthesize `parent.field` and emit it as a normal attribute reference.
    SyntheticNode baseNode(expr->getLoc(), parent);
    AttributeRefNode fieldRef(&baseNode, expr->getLoc(),
                              path->fieldName.getValue());
    CValue fieldVal = emitter.emitExprCValue(&fieldRef, EC_AttributeRefBase);
    if (!fieldVal)
      return {};
    return pathValues[path] = fieldVal;
  }
  case PatternPath::EnumPayload: {
    TypedAttr indexAttr =
        IntegerAttr::get(IndexType::get(emitter.getContext()), path->index);
    CValue caseIdxInt = emitter.emitInt(
        ASTExprAnd<PValue>{PValue(indexAttr), expr}, EC_CallParamValue);
    if (!caseIdxInt)
      return {};
    SyntheticNode subjectNode(expr->getLoc(), parent);
    AttributeRefNode payloadMethod(&subjectNode, expr->getLoc(),
                                   "_unsafe_get_enum_payload");
    SyntheticNode indexNode(expr->getLoc(), caseIdxInt);
    Operand indexOperand(&indexNode, expr->getLoc(),
                         ArgUnpackStyle::kPositional);
    SubscriptNode subscript(&payloadMethod, expr->getLoc(), indexOperand,
                            expr->getLoc());
    CallNode payloadCall(&subscript, expr->getLoc(), /*operands=*/{},
                         expr->getLoc());
    CValue payload = emitter.emitExprCValue(&payloadCall, EC_MatchSubject);
    if (!payload)
      return {};
    return pathValues[path] = payload;
  }
  }
  llvm_unreachable("unknown PatternPath kind");
}

/// Emit `__eq__` of `value` against the literal or enum tag in `command`, as a
/// scalar bool. `value` is the subject for `Equal` and the discriminant for
/// `EnumTag`.
SRValue PatternEmitState::emitTestForValue(OpBuilder &builder,
                                           const PatternCommand *command,
                                           CValue value) {
  assert(command->kind == PatternCommand::Equal ||
         command->kind == PatternCommand::EnumTag);
  IREmitter emitter(curDeclScope, builder);
  const ExprNode *expr = command->expr;
  AnyValue rhs;
  if (command->kind == PatternCommand::EnumTag) {
    TypedAttr indexAttr = IntegerAttr::get(IndexType::get(emitter.getContext()),
                                           command->enumCaseIndex);
    CValue caseIdxInt = emitter.emitInt(
        ASTExprAnd<PValue>{PValue(indexAttr), expr}, EC_CallParamValue);
    if (!caseIdxInt)
      return {};
    rhs = caseIdxInt;
  } else {
    ExprDest litDest(value.getRValueType(), EC_MatchSubject);
    rhs = emitter.emitExpr(expr, litDest);
    if (!rhs)
      return {};
  }

  CValue eqResult = emitter.emitNamedMethodCall(
      "__eq__",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_BoolCondition),
                   {{AnyValue(value), expr}, {rhs, expr}}));
  auto resultSB = emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition);
  auto result = emitter.emitSRValue({resultSB, expr}, EC_BoolCondition);

  // The emitter may have moved the insertion point, keep the caller up to date.
  builder = *emitter.builder;
  return result;
}

/// Given an Equal/EnumTag command, emit the subject and (if an enum) extract
/// the discriminant.
CValue PatternEmitState::emitTestableValue(OpBuilder &builder,
                                           const PatternCommand *command) {
  assert(command->kind == PatternCommand::Equal ||
         command->kind == PatternCommand::EnumTag);
  // Emit the subject for equals and enum tests both.
  CValue subject = getPathValue(builder, command->path, command->expr);
  if (!subject || command->kind == PatternCommand::Equal)
    return subject;

  // Enum tests need the discriminant of the enum, not the whole value.
  IREmitter emitter(curDeclScope, builder);
  return emitGetEnumDiscriminant(emitter, subject, command->expr);
}

LogicalResult
PatternEmitState::emitCommands(OpBuilder &builder,
                               ArrayRef<const PatternCommand *> commands,
                               SmallVectorImpl<PatternBoundName> &bindings) {
  for (const PatternCommand *cmd : commands) {
    assert(cmd && "null pattern command");
    switch (cmd->kind) {
    case PatternCommand::Equal:
    case PatternCommand::EnumTag: {
      CValue value = emitTestableValue(builder, cmd);
      if (!value)
        return failure();
      SRValue matches = emitTestForValue(builder, cmd, value);
      if (!matches)
        return failure();

      /// Emit a dynamic test:
      ///   hlcf.elif matches {
      ///     hlcf.yield
      ///   } else {
      ///     hlcf.match.next
      ///   }
      Location loc =
          curDeclScope.getShared().translateLocation(cmd->expr->getLoc());
      HLCF::ElifOp::create(
          builder, loc, TypeRange(), matches,
          [&]() -> LogicalResult {
            HLCF::YieldOp::create(builder, loc);
            return success();
          },
          [&]() -> LogicalResult {
            HLCF::MatchNextOp::create(builder, loc);
            return success();
          });
      break;
    }
    case PatternCommand::Bind: {
      CValue subject = getPathValue(builder, cmd->path, cmd->expr);
      if (!subject)
        return failure();
      bindings.push_back({cmd->bindName, subject, cmd->declKind});
      break;
    }
    case PatternCommand::Or:
      if (failed(emitOr(builder, *cmd, bindings)))
        return failure();
      break;
    }
  }
  return success();
}

/// Consider an "or" pattern like (1, x)|(x, 2).  This is handled by emitting
/// temporary vardecls for the bound value, which is then initialized on each
/// arm.  This sets up the temporary VarDecls to use for the intermediates.
static LogicalResult
initOrBindings(const OpBuilder &builder, HLCF::MatchOp matchOp, SMLoc loc,
               SmallVectorImpl<PatternBoundName> &aggregateBindings,
               ArrayRef<PatternBoundName> caseBindings, ASTDecl &curDeclScope) {
  if (caseBindings.empty())
    return success();

  // Create one VarDecl per binding before the nested or-match so each
  // alternative can store into the same slots and the result dominates the
  // enclosing case.
  IREmitter emitter(curDeclScope, builder);
  emitter.builder->setInsertionPoint(matchOp);
  Location mlirLoc = emitter.shared.translateLocation(loc);

  for (const PatternBoundName &bn : caseBindings) {
    ASTType varType = bn.value.getRValueType();
    bool isRefOrBind = bn.bindingKind == PatternDeclKind::kRef ||
                       bn.bindingKind == PatternDeclKind::kBind;
    VarDeclKind declKind;
    switch (bn.bindingKind) {
    default:
      assert(false && "unhandled pattern decl kind");
      return failure();
    case PatternDeclKind::kRef:
      declKind = VarDeclKind::Ref;
      break;
    case PatternDeclKind::kVar:
      declKind = VarDeclKind::Var;
      break;
    case PatternDeclKind::kBind:
      declKind = VarDeclKind::Bind;
      break;
    }

    // Match DeclRefNode pattern bindings: ref/bind wrap with a placeholder
    // origin replaced when the value is stored.
    if (isRefOrBind)
      varType = RefType::getAnyOrigin(varType, /*isMut=*/true);

    // Temporary slots only — do not register in the AST scope. The enclosing
    // match case materializes the user-visible bindings from these values.
    VarDeclOp varDecl =
        emitter.emitVarDecl(bn.name, varType, mlirLoc, declKind);
    if (!varDecl)
      return failure();

    CValue slot =
        isRefOrBind ? CValue(RLValue(varDecl)) : CValue(MLValue(varDecl));
    aggregateBindings.push_back({bn.name, slot, bn.bindingKind});
  }
  return success();
}

/// Verify each alternative binds the same names/kinds/types as the aggregate
/// slots, and copy each case binding into its aggregate VarDecl.
static LogicalResult checkOrBindings(
    OpBuilder &builder, SMLoc loc, ArrayRef<PatternBoundName> aggregateBindings,
    ArrayRef<PatternBoundName> caseBindings, ASTDecl &curDeclScope) {
  if (aggregateBindings.empty() && caseBindings.empty())
    return success();

  llvm::StringMap<const PatternBoundName *> caseByName;
  for (const PatternBoundName &bn : caseBindings)
    caseByName[bn.name] = &bn;

  SharedState &shared = curDeclScope.getShared();
  for (const PatternBoundName &bn : aggregateBindings) {
    auto it = caseByName.find(bn.name);
    if (it == caseByName.end()) {
      shared.emitError(loc, "or-pattern alternatives must bind the same names")
          << "; '" << bn.name << "' is bound in one alternative but not "
          << "the other";
      return failure();
    }
    const PatternBoundName &caseBN = *it->second;
    if (bn.bindingKind != caseBN.bindingKind) {
      shared.emitError(loc, "or-pattern binding '")
          << bn.name
          << "' must use the same 'var'/'ref' kind in each alternative";
      return failure();
    }

    ASTType lhsType = bn.value.getRValueType();
    ASTType rhsType = caseBN.value.getRValueType();
    // Aggregate ref/bind slots are `!lit.ref[any] T`; compare `T` to the case
    // subject's type.
    if ((bn.bindingKind == PatternDeclKind::kRef ||
         bn.bindingKind == PatternDeclKind::kBind) &&
        sugarIsa<RefType>(lhsType))
      lhsType = ASTType(sugarCast<RefType>(lhsType).getElementType());
    if (!lhsType.isEqualCanon(rhsType)) {
      auto diag = shared.emitError(loc, "or-pattern binding '")
                  << bn.name << "' has incompatible types across alternatives";
      diag.attachNote(loc) << "first alternative has type " << lhsType
                           << ", this alternative has type " << rhsType;
      return failure();
    }

    // Copy/borrow the case binding into the shared aggregate slot.
    LValue destLV;
    if (MLValue ml = bn.value.getIfMLValue()) {
      destLV = LValue(ml);
    } else {
      RLValue rl = bn.value.getIfRLValue();
      assert(rl && "aggregate or-pattern bindings are ML/RLValues");
      destLV = LValue(rl);
    }
    ExprDest storeDest(destLV, EC_VarInit);
    SyntheticNode locExpr(loc);
    IREmitter emitter(curDeclScope, builder);
    if (!emitter.emitCResult(caseBN.value, &locExpr, storeDest))
      return failure();
    builder = *emitter.builder; // Keep builder in sync.
    caseByName.erase(it);
  }

  if (!caseByName.empty()) {
    shared.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << caseByName.begin()->first()
        << "' is bound in one alternative but not the other";
    return failure();
  }
  return success();
}

LogicalResult
PatternEmitState::emitOr(OpBuilder &builder, const PatternCommand &cmd,
                         SmallVectorImpl<PatternBoundName> &bindings) {

  SharedState &shared = curDeclScope.getShared();

  auto createBindingScope = [&](SMLoc scopeLoc) -> ASTDecl & {
    return shared.getDeclResolver().addFullyResolvedDecl(
        /*declVal=*/nullptr, StringAttr(), scopeLoc, &curDeclScope);
  };

  Location loc = shared.translateLocation(cmd.expr->getLoc());
  auto matchOp =
      HLCF::MatchOp::create(builder, loc, TypeRange(),
                            /*caseRegionsCount=*/cmd.orAlternatives.size());

  SmallVector<PatternBoundName, 4> aggregateBindings;
  for (auto [idx, alt] : llvm::enumerate(cmd.orAlternatives)) {
    Block &block = matchOp.getCaseRegions()[idx].emplaceBlock();
    builder.setInsertionPointToStart(&block);

    ASTDecl &scope = createBindingScope(cmd.expr->getLoc());
    PatternEmitState altState{scope, rootSubject, rootPath, matchLocation,
                              DenseMap<const PatternPath *, CValue>()};
    SmallVector<PatternBoundName, 4> caseBindings;
    if (failed(altState.emitCommands(builder, alt, caseBindings)))
      return failure();

    if (idx == 0) {
      if (failed(initOrBindings(builder, matchOp, cmd.expr->getLoc(),
                                aggregateBindings, caseBindings, curDeclScope)))
        return failure();
    }
    if (failed(checkOrBindings(builder, cmd.expr->getLoc(), aggregateBindings,
                               caseBindings, curDeclScope)))
      return failure();
    HLCF::MatchCompleteOp::create(builder, loc);
  }

  Block &elseBlock = matchOp.getElseRegion().emplaceBlock();
  builder.setInsertionPointToStart(&elseBlock);
  HLCF::MatchNextOp::create(builder, loc);
  builder.setInsertionPointAfter(matchOp);

  for (const PatternBoundName &bn : aggregateBindings) {
    CValue resultBinding;
    if (MLValue ml = bn.value.getIfMLValue()) {
      resultBinding = MRValue(ml);
    } else {
      RLValue rl = bn.value.getIfRLValue();
      Value refVal = RefLoadOp::create(builder, loc, rl);
      resultBinding = CValue::getMValueForRef(refVal);
    }
    bindings.push_back({bn.name, resultBinding, bn.bindingKind});
  }
  return success();
}
