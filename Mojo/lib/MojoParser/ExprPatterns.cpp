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
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/CallOperands.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/POPDialect/POPAttrs.h"
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

  for (const PatternPath *p : llvm::reverse(chain)) {
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

/// Read `SubjectType._enum_case_names` as a concrete ParamListAttr value list.
static std::optional<ArrayRef<TypedAttr>>
getEnumCaseNames(IREmitter &emitter, ASTType subjectType, SMLoc loc) {
  SyntheticNode typeNode(loc, PValue(subjectType));
  AttributeRefNode namesRef(&typeNode, loc, "_enum_case_names");
  TypedAttr namesAttr = emitter.emitExprPValue(&namesRef, EC_AttributeRefBase);
  if (auto namesList =
          dyn_cast_or_null<ParamListAttr>(getCanonicalAttr(namesAttr)))
    return namesList.getValues();
  return std::nullopt;
}

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
  std::optional<ArrayRef<TypedAttr>> caseNames =
      getEnumCaseNames(emitter, subjectType, expr->getLoc());
  if (!caseNames) {
    emitter.emitError(expr->getLoc(), "cannot match on a parametric enum type")
        << expr->getRange();
    return std::nullopt;
  }

  for (auto [idx, nameAttr] : llvm::enumerate(*caseNames)) {
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
  // Parametric subjects stay as PValues; dynamic ones borrow as BValues.
  AnyValue selfArg = subject;
  if (!subject.getIfPValue()) {
    BValue subjectBVal = emitter.emitBValue({subject, expr}, EC_MatchSubject);
    if (!subjectBVal)
      return {};
    selfArg = subjectBVal;
  }
  return emitter.emitNamedMethodCall("_get_enum_discriminant",
                                     CallOperands(CallSyntax::kMethodCall, expr,
                                                  ExprDest(EC_MatchSubject),
                                                  {{selfArg, expr}}));
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

CValue PatternEmitState::getPathValue(IREmitter &emitter,
                                      const PatternPath *path,
                                      const ExprNode *expr) {
  assert(path && "null pattern path");
  if (auto it = pathValues.find(path); it != pathValues.end())
    return it->second;

  if (path->kind == PatternPath::Root) {
    assert(path == rootPath && "unexpected root path");
    return pathValues[path] = rootSubject;
  }

  CValue parent = getPathValue(emitter, path->parent, expr);
  if (!parent)
    return {};

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
RValue PatternEmitState::emitTestForValue(IREmitter &emitter,
                                          const PatternCommand *command,
                                          CValue value) {
  assert(command->kind == PatternCommand::Equal ||
         command->kind == PatternCommand::EnumTag);
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
  return emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition);
}

/// Given an Equal/EnumTag command, emit the subject and (if an enum) extract
/// the discriminant.
CValue PatternEmitState::emitTestableValue(IREmitter &emitter,
                                           const PatternCommand *command) {
  assert(command->kind == PatternCommand::Equal ||
         command->kind == PatternCommand::EnumTag);
  // Emit the subject for equals and enum tests both.
  CValue subject = getPathValue(emitter, command->path, command->expr);
  if (!subject || command->kind == PatternCommand::Equal)
    return subject;

  // Enum tests need the discriminant of the enum, not the whole value.
  return emitGetEnumDiscriminant(emitter, subject, command->expr);
}

/// Emit a comptime `Or` pattern: OR alternative conditions and Cond-select
/// each binding value with the same condition shape as the or:
///   or(c0, c1) = cond(c0, c0, c1)
///   val        = cond(c0, v0, v1)
/// Appends the merged bindings to `bindings`.
static FailureOr<TypedAttr>
emitComptimeOrCondition(PatternEmitState &state, const PatternCommand &cmd,
                        SmallVectorImpl<PatternBoundName> &bindings) {
  assert(cmd.kind == PatternCommand::Or && "expected Or command");
  assert(!cmd.orAlternatives.empty() && "or-pattern needs alternatives");

  struct AltResult {
    TypedAttr cond;
    SmallVector<PatternBoundName, 4> bindings;
  };
  SmallVector<AltResult, 2> alts;
  alts.reserve(cmd.orAlternatives.size());
  for (PatternCommandList alt : cmd.orAlternatives) {
    SmallVector<PatternBoundName, 4> altBindings;
    FailureOr<TypedAttr> altCond =
        state.emitComptimeCondition(alt, altBindings);
    if (failed(altCond))
      return failure();
    alts.push_back({*altCond, std::move(altBindings)});
  }

  // Agree on names/kinds/types across alternatives (same rules as runtime).
  SharedState &shared = state.curDeclScope.getShared();
  SMLoc orLoc = cmd.expr->getLoc();
  ArrayRef<PatternBoundName> firstBindings = alts.front().bindings;
  for (const AltResult &alt : llvm::drop_begin(alts)) {
    llvm::StringMap<const PatternBoundName *> byName;
    for (const PatternBoundName &bn : alt.bindings)
      byName[bn.name] = &bn;
    for (const PatternBoundName &bn : firstBindings) {
      auto it = byName.find(bn.name);
      if (it == byName.end()) {
        shared.emitError(orLoc,
                         "or-pattern alternatives must bind the same names")
            << "; '" << bn.name
            << "' is bound in one alternative but not the other";
        return failure();
      }
      const PatternBoundName &other = *it->second;
      if (bn.bindingKind != other.bindingKind) {
        shared.emitError(orLoc, "or-pattern binding '")
            << bn.name
            << "' must use the same 'var'/'ref' kind in each alternative";
        return failure();
      }
      if (!bn.value.getRValueType().isEqualCanon(other.value.getRValueType())) {
        auto diag = shared.emitError(orLoc, "or-pattern binding '")
                    << bn.name
                    << "' has incompatible types across alternatives";
        diag.attachNote(orLoc)
            << "first alternative has type " << bn.value.getRValueType()
            << ", this alternative has type " << other.value.getRValueType();
        return failure();
      }
      byName.erase(it);
    }
    if (!byName.empty()) {
      shared.emitError(orLoc,
                       "or-pattern alternatives must bind the same names")
          << "; '" << byName.begin()->first()
          << "' is bound in one alternative but not the other";
      return failure();
    }
  }

  // Left-fold: Cond-select values with the pre-Or condition, then Or.
  TypedAttr combinedCond = alts.front().cond;
  SmallVector<PatternBoundName, 4> merged = alts.front().bindings;
  for (const AltResult &alt : llvm::drop_begin(alts)) {
    for (PatternBoundName &bn : merged) {
      const PatternBoundName *other = nullptr;
      for (const PatternBoundName &obn : alt.bindings) {
        if (obn.name == bn.name) {
          other = &obn;
          break;
        }
      }
      assert(other && "binding agreement checked above");
      PValue prevPV = bn.value.getIfPValue();
      PValue nextPV = other->value.getIfPValue();
      if (!prevPV || !nextPV) {
        IREmitter emitter(state.curDeclScope, EC_MatchSubject);
        emitter.emitError(orLoc)
            << "variable binding '" << bn.name
            << "' in 'comptime match' must be a compile-time value"
            << cmd.expr->getRange();
        return failure();
      }
      bn.value =
          ParamOperatorAttr::get(POC::Cond, {combinedCond, prevPV, nextPV});
    }
    combinedCond = ParamOperatorAttr::getLogicalOr(combinedCond, alt.cond);
  }
  bindings.append(merged.begin(), merged.end());
  return combinedCond;
}

/// AND together Equal/EnumTag tests in `commands` into one compile-time
/// bool attribute. Or alternatives are OR'd with `getLogicalOr` (same as
/// comptime `or`). Bind commands append to `bindings` (like `emitCommands`)
/// without contributing to the condition.
FailureOr<TypedAttr> PatternEmitState::emitComptimeCondition(
    ArrayRef<const PatternCommand *> commands,
    SmallVectorImpl<PatternBoundName> &bindings) {
  // Use a parameter emitter so everything is comptime.
  IREmitter emitter(curDeclScope, EC_MatchSubject);

  // Empty command list is irrefutable (`_`).
  if (commands.empty())
    return TypedAttr(SIMDAttr::getScalarBool(emitter.getContext(), true));

  TypedAttr combined;
  for (const PatternCommand *cmd : commands) {
    assert(cmd && "null pattern command");
    TypedAttr newTest; // The new test to merge in
    switch (cmd->kind) {
    case PatternCommand::Bind: {
      CValue subject = getPathValue(emitter, cmd->path, cmd->expr);
      if (!subject)
        return failure();
      bindings.push_back({cmd->bindName, subject, cmd->declKind});
      continue;
    }
    case PatternCommand::Or: {
      FailureOr<TypedAttr> orCond =
          emitComptimeOrCondition(*this, *cmd, bindings);
      if (failed(orCond))
        return failure();
      newTest = *orCond;
      break;
    }
    case PatternCommand::Equal:
    case PatternCommand::EnumTag: {
      CValue value = emitTestableValue(emitter, cmd);
      if (!value)
        return failure();
      RValue matches = emitTestForValue(emitter, cmd, value);
      if (!matches)
        return failure();
      newTest = matches.getIfPValue();
      if (!newTest) {
        emitter.emitError(cmd->expr->getLoc())
            << "'comptime match' case condition must be evaluable at "
               "compile-time"
            << cmd->expr->getRange();
        return failure();
      }
      break;
    }
    }

    assert(newTest && "Should have produced a new test to merge in");
    if (combined)
      combined = ParamOperatorAttr::getLogicalAnd(combined, newTest);
    else
      combined = newTest;
  }
  // Bind-only patterns are irrefutable: no tests, just names to materialize.
  if (!combined)
    return TypedAttr(SIMDAttr::getScalarBool(emitter.getContext(), true));
  return combined;
}

LogicalResult
PatternEmitState::emitCommands(OpBuilder &builder,
                               ArrayRef<const PatternCommand *> commands,
                               SmallVectorImpl<PatternBoundName> &bindings) {
  IREmitter emitter(curDeclScope, builder);
  for (const PatternCommand *cmd : commands) {
    assert(cmd && "null pattern command");
    switch (cmd->kind) {
    case PatternCommand::Equal:
    case PatternCommand::EnumTag: {
      CValue value = emitTestableValue(emitter, cmd);
      if (!value)
        return failure();
      RValue matchesR = emitTestForValue(emitter, cmd, value);
      SRValue matches = emitter.emitSRValue({AnyValue(matchesR), cmd->expr},
                                            EC_BoolCondition);
      if (!matches)
        return failure();

      /// Emit a dynamic test:
      ///   hlcf.if matches {
      ///     hlcf.yield
      ///   } else {
      ///     hlcf.match.next
      ///   }
      Location loc =
          curDeclScope.getShared().translateLocation(cmd->expr->getLoc());
      HLCF::IfOp::create(
          *emitter.builder, loc, TypeRange(), matches,
          [&]() -> LogicalResult {
            HLCF::YieldOp::create(*emitter.builder, loc);
            return success();
          },
          [&]() -> LogicalResult {
            HLCF::MatchNextOp::create(*emitter.builder, loc);
            return success();
          });
      break;
    }
    case PatternCommand::Bind: {
      CValue subject = getPathValue(emitter, cmd->path, cmd->expr);
      if (!subject)
        return failure();
      bindings.push_back({cmd->bindName, subject, cmd->declKind});
      break;
    }
    case PatternCommand::Or:
      if (emitter.builder)
        builder = *emitter.builder;
      if (failed(emitOr(builder, *cmd, bindings)))
        return failure();
      emitter.builder = builder;
      break;
    }
  }
  if (emitter.builder)
    builder = *emitter.builder;
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

//===----------------------------------------------------------------------===//
// Match exhaustivity / unreachability
//===----------------------------------------------------------------------===//

namespace {

/// Cap on flattened product cells. Larger products become TooComplex.
constexpr size_t kMaxProductCells = 1024;

/// One EnumLike dimension in a flattened FiniteCtors product (tuple leaves).
struct MatchCoveringDim {
  const PatternPath *path = nullptr;
  size_t numCtors = 0;
  ArrayRef<TypedAttr> caseNames;
};

/// One covered Equal-literal spelling in a LiteralSet space (bump-linked).
struct CoveredLiteralNode {
  StringRef spelling;
  CoveredLiteralNode *next = nullptr;
};

/// Lazy covering model for one `match`.
/// Bump-allocated so nested / product spaces can share the builder's arena.
///
/// Product of EnumLike leaves is flattened into a cell bitset so correlated
/// patterns (`True, True` / `_, False`) cover the right combinations —
/// independent per-field bitsets would wrongly treat `True,True` +
/// `False,False` as total. Applies to tuples and structs of EnumLike fields.
///
/// LiteralSet tracks root Equal spellings (`4`, `"foo"`) for duplicate /
/// unreachability only — the domain is open, so covering literals never
/// proves exhaustivity (only a catch-all closes the space).
struct MatchCoveringSpace {
  enum class Kind : uint8_t {
    Opaque,
    FiniteCtors,
    Product,
    /// Tuple/struct of EnumLike leaves whose flattened cell count exceeds
    /// `kMaxProductCells`. Not tracked cell-by-cell; requires a catch-all.
    TooComplex,
    /// Open universe of root Equal literals (Int, String, …).
    LiteralSet,
    Closed
  };
  Kind kind = Kind::Opaque;

  // FiniteCtors (EnumLike root):
  size_t numCtors = 0;
  bool *remaining = nullptr;
  ArrayRef<TypedAttr> caseNames;

  // Product (tuple/struct of EnumLike leaves, flattened):
  size_t numDims = 0;
  MatchCoveringDim *dims = nullptr;
  size_t numCells = 0;
  bool *cells = nullptr; ///< `true` = that combination is still open.

  // LiteralSet (open Equal-literal universe):
  CoveredLiteralNode *coveredLiterals = nullptr;
  llvm::BumpPtrAllocator *allocator = nullptr;

  bool hasOpenCtor() const {
    if (kind != Kind::FiniteCtors || !remaining)
      return false;
    for (size_t i = 0; i < numCtors; ++i)
      if (remaining[i])
        return true;
    return false;
  }

  bool hasOpenCell() const {
    if (kind != Kind::Product || !cells)
      return false;
    for (size_t i = 0; i < numCells; ++i)
      if (cells[i])
        return true;
    return false;
  }

  bool isFullyCovered() const {
    return kind == Kind::Closed ||
           (kind == Kind::FiniteCtors && !hasOpenCtor()) ||
           (kind == Kind::Product && !hasOpenCell());
  }

  void closeAll() { kind = Kind::Closed; }

  void coverCtor(size_t index) {
    assert(kind == Kind::FiniteCtors && remaining && index < numCtors);
    remaining[index] = false;
  }

  bool isCtorOpen(size_t index) const {
    return kind == Kind::FiniteCtors && remaining && index < numCtors &&
           remaining[index];
  }

  bool isLiteralCovered(StringRef spelling) const {
    for (auto *node = coveredLiterals; node; node = node->next)
      if (node->spelling == spelling)
        return true;
    return false;
  }

  void coverLiteral(StringRef spelling) {
    assert(kind == Kind::LiteralSet && allocator);
    void *mem = allocator->Allocate(sizeof(CoveredLiteralNode),
                                    alignof(CoveredLiteralNode));
    coveredLiterals = new (mem) CoveredLiteralNode{spelling, coveredLiterals};
  }
};

/// Look up `spelling` in an EnumLike `_enum_case_names` list (e.g. `"True"` →
/// index of that ctor). Used for Bool-style `Equal` patterns and witnesses.
static std::optional<size_t> findCaseNameIndex(ArrayRef<TypedAttr> names,
                                               StringRef spelling) {
  if (spelling.empty())
    return std::nullopt;
  for (auto [idx, nameAttr] : llvm::enumerate(names)) {
    auto nameStr = dyn_cast<StringAttr>(nameAttr);
    if (nameStr && nameStr.getValue() == spelling)
      return idx;
  }
  return std::nullopt;
}

/// Spelling for ctor `index` in diagnostics (`"False"`, `"Some"`, ...).
static StringRef caseNameSpelling(ArrayRef<TypedAttr> names, size_t index) {
  if (index < names.size())
    if (auto str = dyn_cast<StringAttr>(names[index]))
      return str.getValue();
  return "?";
}

/// True when `type` is the builtin `Tuple` (ignoring element parameters).
static bool isTupleType(PatternMatchBuilder &builder, ASTType type, SMLoc loc) {
  ASTType tupleType =
      builder.shared.lookupBuiltinType("Tuple", builder.declScope, loc);
  if (!tupleType)
    return false;
  return tupleType.isEqualCanon(type.getWithoutParameters(builder.shared));
}

/// Append EnumLike leaf dimensions under `path`. Returns false if any leaf is
/// not a known FiniteCtors type (caller falls back to Opaque).
///
/// Walks `Tuple` elements and struct stored fields; the Product covering logic
/// is path-kind agnostic (`TupleElement` vs `StructField`).
static bool collectFiniteDims(PatternMatchBuilder &builder,
                              const PatternPath *path, SMLoc loc,
                              SmallVectorImpl<MatchCoveringDim> &dims) {
  ASTType type = path->type;
  if (type.provenConformsToBuiltinTrait(
          "EnumLike", loc, builder.shared,
          ASTDecl::getAssumptionsFromScope(&builder.declScope))) {
    IREmitter emitter = builder.getParamEmitter();
    auto names = getEnumCaseNames(emitter, type, loc);
    if (!names || names->empty())
      return false;
    dims.push_back({path, names->size(), *names});
    return true;
  }

  if (isTupleType(builder, type, loc)) {
    assert(type.getParamBindings().size() == 2 && "Tuple has two parameters");
    auto vaAttr = sugarCast<ParamListAttr>(type.getParamBindings()[0]);
    for (unsigned i = 0, e = vaAttr.getValues().size(); i != e; ++i) {
      ASTType eltType = ASTType(vaAttr.getValues()[i]);
      const PatternPath *eltPath = builder.getTupleElement(path, i, eltType);
      if (!collectFiniteDims(builder, eltPath, loc, dims))
        return false;
    }
    return true;
  }

  // Struct of EnumLike (or nested tuple/struct) fields — same product model.
  auto structType = dyn_cast<LIT::StructType>(SugarAttr::strip(type.mlirType));
  ASTDecl *typeDecl = type.getDecl(builder.shared);
  if (!structType || !typeDecl)
    return false;
  auto structDeclOp =
      dyn_cast_or_null<LIT::StructDeclOp>(typeDecl->getIfOperation());
  if (!structDeclOp)
    return false;

  for (LIT::StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getReboundType(
        structType, &builder.shared.getEvaluationContext());
    const PatternPath *fieldPath =
        builder.getStructField(path, fieldOp.getNameAttr(), fieldType);
    if (!collectFiniteDims(builder, fieldPath, loc, dims))
      return false;
  }
  return true;
}

/// Build the initial covering space for `rootPath->type`.
static MatchCoveringSpace *createRootCoveringSpace(PatternMatchBuilder &builder,
                                                   const PatternPath *rootPath,
                                                   SMLoc matchLoc) {
  auto *space = builder.create<MatchCoveringSpace>();
  ASTType subjectType = rootPath->type;

  // EnumLike root → FiniteCtors.
  if (subjectType.provenConformsToBuiltinTrait(
          "EnumLike", matchLoc, builder.shared,
          ASTDecl::getAssumptionsFromScope(&builder.declScope))) {
    IREmitter emitter = builder.getParamEmitter();
    auto names = getEnumCaseNames(emitter, subjectType, matchLoc);
    if (!names || names->empty())
      return space; // Opaque — incomplete reflection metadata

    space->kind = MatchCoveringSpace::Kind::FiniteCtors;
    space->numCtors = names->size();
    space->caseNames = *names;
    space->remaining = builder.allocator.Allocate<bool>(space->numCtors);
    std::fill(space->remaining, space->remaining + space->numCtors, true);
    return space;
  }

  // Tuple / struct of EnumLike leaves → flattened Product cell bitset.
  SmallVector<MatchCoveringDim, 4> dims;
  if (collectFiniteDims(builder, rootPath, matchLoc, dims) && !dims.empty()) {
    size_t numCells = 1;
    bool tooLarge = false;
    for (const MatchCoveringDim &dim : dims) {
      if (dim.numCtors == 0 || numCells > kMaxProductCells / dim.numCtors) {
        tooLarge = true;
        break;
      }
      numCells *= dim.numCtors;
    }
    if (tooLarge || numCells > kMaxProductCells) {
      space->kind = MatchCoveringSpace::Kind::TooComplex;
      return space;
    }

    space->kind = MatchCoveringSpace::Kind::Product;
    space->numDims = dims.size();
    space->dims = builder.allocator.Allocate<MatchCoveringDim>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
      space->dims[i] = dims[i];
    space->numCells = numCells;
    space->cells = builder.allocator.Allocate<bool>(numCells);
    std::fill(space->cells, space->cells + numCells, true);
    return space;
  }

  // Everything else: open Equal-literal universe (Int, String, …). Track
  // covered spellings for duplicate / unreachability; never prove exhaustivity.
  space->kind = MatchCoveringSpace::Kind::LiteralSet;
  space->allocator = &builder.allocator;
  return space;
}

/// True if `ancestor` is `path` or any parent of `path` in the PatternPath
/// tree. Used when classifying a command relative to product dimensions:
///   - Bind on an ancestor of several dims → those dims stay wildcards
///   - Commands under a dim / root ctor (payload refine) are ignored for
///     tag-level FiniteCtors / product covering
static bool isAncestorOrEqual(const PatternPath *ancestor,
                              const PatternPath *path) {
  for (const PatternPath *p = path; p; p = p->parent)
    if (p == ancestor)
      return true;
  return false;
}

/// If these Or-free commands cover exactly one root constructor (EnumTag /
/// Bool-style Equal), return that index. Payload refinements under that ctor
/// (binds or nested tests) are ignored for tag-level credit — v1 treats the
/// EnumLike tag set as the universe. Otherwise nullopt.
static std::optional<size_t>
getSimpleRootCtorCoverage(PatternCommandList commands,
                          const PatternPath *rootPath,
                          ArrayRef<TypedAttr> caseNames) {
  std::optional<size_t> ctor;
  for (const PatternCommand *cmd : commands) {
    assert(cmd->kind != PatternCommand::Or && "expand Or before covering");
    if (cmd->kind == PatternCommand::Bind)
      continue;
    // Payload / nested refinements under the matched root ctor: ignore for
    // tag-level FiniteCtors credit (`Some(0)` still covers `Some`).
    if (cmd->path != rootPath) {
      if (isAncestorOrEqual(rootPath, cmd->path))
        continue;
      return std::nullopt;
    }

    if (cmd->kind == PatternCommand::EnumTag) {
      if (ctor)
        return std::nullopt;
      ctor = cmd->enumCaseIndex;
      continue;
    }
    if (cmd->kind == PatternCommand::Equal) {
      assert(cmd->expr);
      auto idx = findCaseNameIndex(caseNames, cmd->expr->getLiteralSpelling());
      if (!idx || ctor)
        return std::nullopt;
      ctor = *idx;
      continue;
    }
    return std::nullopt;
  }
  return ctor;
}

/// Expand `Or` commands into a disjunction of Or-free command lists. Each
/// alternative is covered independently (union). Nested / multiple Ors are
/// flattened by substituting one Or at a time.
static void
expandOrCommands(PatternCommandList commands,
                 SmallVectorImpl<SmallVector<const PatternCommand *, 8>> &out) {
  for (size_t i = 0, e = commands.size(); i != e; ++i) {
    if (commands[i]->kind != PatternCommand::Or)
      continue;

    for (PatternCommandList alt : commands[i]->orAlternatives) {
      SmallVector<const PatternCommand *, 8> combined;
      combined.reserve(i + alt.size() + (e - i - 1));
      combined.append(commands.begin(), commands.begin() + i);
      combined.append(alt.begin(), alt.end());
      combined.append(commands.begin() + i + 1, commands.end());
      expandOrCommands(combined, out);
    }
    return;
  }
  out.emplace_back(commands.begin(), commands.end());
}

/// If `path` is exactly one flattened product dimension (an EnumLike leaf
/// under the match root), return its index in `space.dims`. Commands that
/// `Equal`/`EnumTag`/`Bind` that leaf constrain or wildcard that dimension.
static std::optional<size_t> findDimIndex(const MatchCoveringSpace &space,
                                          const PatternPath *path) {
  for (size_t i = 0; i < space.numDims; ++i)
    if (space.dims[i].path == path)
      return i;
  return std::nullopt;
}

/// Map an Equal / EnumTag command to a ctor index on `dim` (via enum case
/// index or literal spelling like `"True"`).
static std::optional<size_t> ctorIndexForCommand(const PatternCommand *cmd,
                                                 const MatchCoveringDim &dim) {
  if (cmd->kind == PatternCommand::EnumTag)
    return cmd->enumCaseIndex;
  if (cmd->kind == PatternCommand::Equal) {
    assert(cmd->expr);
    return findCaseNameIndex(dim.caseNames, cmd->expr->getLiteralSpelling());
  }
  return std::nullopt;
}

/// Unpack a flat cell id into per-dimension ctor indices. Layout is row-major
/// with the last dimension varying fastest: for Bool×Bool (False=0, True=1),
/// cell 2 → `(True, False)`.
static void decodeCell(const MatchCoveringSpace &space, size_t cell,
                       SmallVectorImpl<size_t> &out) {
  out.resize(space.numDims);
  size_t rest = cell;
  for (size_t i = space.numDims; i > 0; --i) {
    size_t dim = i - 1;
    size_t n = space.dims[dim].numCtors;
    out[dim] = rest % n;
    rest /= n;
  }
}

/// Inverse of `decodeCell`: pack per-dimension ctor indices into a cell id.
static size_t encodeCell(const MatchCoveringSpace &space,
                         ArrayRef<size_t> indices) {
  size_t cell = 0;
  for (size_t i = 0; i < space.numDims; ++i)
    cell = cell * space.dims[i].numCtors + indices[i];
  return cell;
}

/// Render an open cell as a user-facing witness pattern, e.g. `"True, False"`.
static std::string formatProductWitness(const MatchCoveringSpace &space,
                                        size_t cell) {
  SmallVector<size_t, 4> indices;
  decodeCell(space, cell, indices);
  std::string result;
  llvm::raw_string_ostream os(result);
  for (size_t i = 0; i < space.numDims; ++i) {
    if (i)
      os << ", ";
    os << caseNameSpelling(space.dims[i].caseNames, indices[i]);
  }
  return result;
}

/// If these Or-free commands are exactly one root Equal with a known literal
/// spelling (plus optional Binds), return that spelling. Used for LiteralSet
/// covering (`case 4:`, `case "foo":`).
static std::optional<StringRef>
getSimpleRootLiteralCoverage(PatternCommandList commands,
                             const PatternPath *rootPath) {
  std::optional<StringRef> literal;
  for (const PatternCommand *cmd : commands) {
    assert(cmd->kind != PatternCommand::Or && "expand Or before covering");
    if (cmd->kind == PatternCommand::Bind)
      continue;
    if (cmd->path != rootPath || cmd->kind != PatternCommand::Equal ||
        !cmd->expr)
      return std::nullopt;
    StringRef spelling = cmd->expr->getLiteralSpelling();
    if (spelling.empty() || literal)
      return std::nullopt;
    literal = spelling;
  }
  return literal;
}

/// Cover LiteralSet with an Or-free command list. Returns false if the pattern
/// cannot be credited; `hitOpen` if a new spelling was recorded or the space
/// was closed by a bind-only alternative.
static bool coverLiteralCommands(MatchCoveringSpace &space,
                                 PatternCommandList commands,
                                 const PatternPath *rootPath, bool &hitOpen) {
  hitOpen = false;
  assert(space.kind == MatchCoveringSpace::Kind::LiteralSet);

  // Irrefutable alternative (`_` / bind-only): closes the open universe.
  if (llvm::all_of(commands, [](const PatternCommand *cmd) {
        return cmd->kind == PatternCommand::Bind;
      })) {
    space.closeAll();
    hitOpen = true;
    return true;
  }

  std::optional<StringRef> literal =
      getSimpleRootLiteralCoverage(commands, rootPath);
  if (!literal)
    return false;
  if (space.isLiteralCovered(*literal))
    return true;
  space.coverLiteral(*literal);
  hitOpen = true;
  return true;
}

/// Cover FiniteCtors with an Or-free command list. Returns false if the
/// pattern cannot be credited; `hitOpen` if a still-open ctor was cleared.
static bool coverFiniteCtorsCommands(MatchCoveringSpace &space,
                                     PatternCommandList commands,
                                     const PatternPath *rootPath,
                                     bool &hitOpen) {
  hitOpen = false;
  assert(space.kind == MatchCoveringSpace::Kind::FiniteCtors);

  // Irrefutable alternative (`_` / bind-only, including empty): closes every
  // remaining ctor. Needed for `case True | _:` where `_` is an Or arm.
  if (llvm::all_of(commands, [](const PatternCommand *cmd) {
        return cmd->kind == PatternCommand::Bind;
      })) {
    for (size_t i = 0; i < space.numCtors; ++i) {
      if (space.remaining[i]) {
        space.remaining[i] = false;
        hitOpen = true;
      }
    }
    return true;
  }

  std::optional<size_t> ctor =
      getSimpleRootCtorCoverage(commands, rootPath, space.caseNames);
  if (!ctor)
    return false;
  if (space.isCtorOpen(*ctor)) {
    space.coverCtor(*ctor);
    hitOpen = true;
  }
  return true;
}

/// Cover product cells matched by an Or-free command list. Returns:
///   true  — pattern was interpretable; `hitOpen` says whether any open cell
///           was cleared (false ⇒ already covered for this alternative).
///   false — cannot credit this alternative (unknown path / bad literal).
static bool coverProductCommands(MatchCoveringSpace &space,
                                 PatternCommandList commands, bool &hitOpen) {
  hitOpen = false;
  assert(space.kind == MatchCoveringSpace::Kind::Product);

  // Per-dimension constraint: nullopt = wildcard, else required ctor index.
  SmallVector<std::optional<size_t>, 4> constraints(space.numDims,
                                                    std::nullopt);

  for (const PatternCommand *cmd : commands) {
    assert(cmd->kind != PatternCommand::Or && "expand Or before covering");

    if (cmd->kind == PatternCommand::Bind) {
      // Bind on a dim path (or ancestor of dims) is a wildcard — already
      // represented by nullopt. Bind on a payload under a dim is ignored.
      if (auto dim = findDimIndex(space, cmd->path))
        continue;
      bool underSomeDim = false;
      for (size_t i = 0; i < space.numDims; ++i) {
        if (isAncestorOrEqual(space.dims[i].path, cmd->path) &&
            cmd->path != space.dims[i].path) {
          underSomeDim = true;
          break;
        }
      }
      if (underSomeDim)
        continue;
      // Bind on an intermediate tuple path that owns several dims: those dims
      // stay wildcards (nullopt). Any dim whose path is under cmd->path is OK.
      bool coversDims = false;
      for (size_t i = 0; i < space.numDims; ++i) {
        if (isAncestorOrEqual(cmd->path, space.dims[i].path)) {
          coversDims = true;
          break;
        }
      }
      if (coversDims)
        continue;
      return false;
    }

    if (auto dim = findDimIndex(space, cmd->path)) {
      auto ctor = ctorIndexForCommand(cmd, space.dims[*dim]);
      if (!ctor || constraints[*dim])
        return false;
      constraints[*dim] = *ctor;
      continue;
    }

    // Payload refinements under a dim (`Some(0)` on an Optional leaf): ignore
    // for tag-level product covering — the dim constraint already recorded the
    // enum ctor.
    bool underSomeDim = false;
    for (size_t i = 0; i < space.numDims; ++i) {
      if (isAncestorOrEqual(space.dims[i].path, cmd->path) &&
          cmd->path != space.dims[i].path) {
        underSomeDim = true;
        break;
      }
    }
    if (underSomeDim)
      continue;
    return false;
  }

  // Walk matching cells. For small products a full scan is fine.
  SmallVector<size_t, 4> indices(space.numDims, 0);
  // Initialize wildcards to 0; fixed constraints to their value.
  for (size_t d = 0; d < space.numDims; ++d)
    if (constraints[d])
      indices[d] = *constraints[d];

  while (true) {
    size_t cell = encodeCell(space, indices);
    if (space.cells[cell]) {
      space.cells[cell] = false;
      hitOpen = true;
    }

    // Increment the first wildcard dimension (odometer over free dims).
    size_t d = 0;
    for (; d < space.numDims; ++d) {
      if (constraints[d])
        continue;
      ++indices[d];
      if (indices[d] < space.dims[d].numCtors)
        break;
      indices[d] = 0;
    }
    if (d == space.numDims)
      break;
  }
  return true;
}

/// Cover `commands` against `space`, expanding Or into a union of alternatives.
/// Sets `hitOpen` if any alternative cleared still-open coverage. Returns false
/// only when every alternative is uninterpretable (no credit at all).
///
/// Unreachability: when every alternative is interpretable and none hit open
/// space, the whole Or (or Or-free pattern) is redundant. If any alternative
/// cannot be credited, we still cover the ones we can but do not treat the
/// case as unreachable — a complex alt might still match at runtime.
static bool coverCommands(MatchCoveringSpace &space,
                          PatternCommandList commands,
                          const PatternPath *rootPath, bool &hitOpen,
                          bool &allAltsInterpreted) {
  hitOpen = false;
  allAltsInterpreted = true;

  SmallVector<SmallVector<const PatternCommand *, 8>, 4> alts;
  expandOrCommands(commands, alts);

  bool anyInterpreted = false;
  for (ArrayRef<const PatternCommand *> alt : alts) {
    bool altHit = false;
    bool ok = false;
    if (space.kind == MatchCoveringSpace::Kind::FiniteCtors)
      ok = coverFiniteCtorsCommands(space, alt, rootPath, altHit);
    else if (space.kind == MatchCoveringSpace::Kind::Product)
      ok = coverProductCommands(space, alt, altHit);
    else if (space.kind == MatchCoveringSpace::Kind::LiteralSet)
      ok = coverLiteralCommands(space, alt, rootPath, altHit);
    else
      return false;

    if (!ok) {
      allAltsInterpreted = false;
      continue;
    }
    anyInterpreted = true;
    hitOpen |= altHit;
  }
  return anyInterpreted;
}

} // namespace

void PatternMatchBuilder::checkCaseExhaustivityAndUnreachability(
    MutableArrayRef<MatchCaseEntry> caseEntries, const PatternPath *rootPath,
    SMLoc matchLoc) {
  // Covering model (see MatchCoveringSpace):
  //   - EnumLike root → FiniteCtors bitset.
  //   - Tuple/struct of EnumLike leaves → flattened Product cell bitset.
  //   - Product larger than kMaxProductCells → TooComplex (needs `_`).
  //   - Else LiteralSet: track root Equal spellings for duplicates only
  //     (no exhaustivity — Int/String/… are open universes).
  //   - Irrefutable unguarded cases close the space.
  //   - Or patterns expand to a union of alternatives; each is covered
  //     independently (guards never shrink the remaining space).
  MatchCoveringSpace *space =
      createRootCoveringSpace(*this, rootPath, matchLoc);

  for (MatchCaseEntry &entry : caseEntries) {
    // Already fully covered → every subsequent case is unreachable.
    if (space->isFullyCovered()) {
      entry.isUnreachable = true;
      emitWarning(entry.patternExpr->getLoc(),
                  "case is unreachable; previous cases cover every "
                  "value of the match subject")
          << entry.patternExpr->getRange();
      continue;
    }

    // Irrefutable unguarded pattern (`_`, bare bind): closes everything.
    if (entry.alwaysMatches()) {
      space->closeAll();
      entry.isConcluding = true;
      continue;
    }

    // Guards never shrink the remaining space.
    if (entry.guardExpr)
      continue;

    if (space->kind != MatchCoveringSpace::Kind::FiniteCtors &&
        space->kind != MatchCoveringSpace::Kind::Product &&
        space->kind != MatchCoveringSpace::Kind::LiteralSet)
      continue;

    bool hitOpen = false;
    bool allAltsInterpreted = false;
    if (!coverCommands(*space, entry.commandList, rootPath, hitOpen,
                       allAltsInterpreted))
      continue;

    if (!hitOpen && allAltsInterpreted) {
      entry.isUnreachable = true;
      // Prefer a ctor/literal-specific message for simple root patterns
      // (with optional Bind residuals), but not for Or.
      bool hasOr =
          llvm::any_of(entry.commandList, [](const PatternCommand *cmd) {
            return cmd->kind == PatternCommand::Or;
          });
      if (!hasOr && space->kind == MatchCoveringSpace::Kind::FiniteCtors) {
        if (auto ctor = getSimpleRootCtorCoverage(entry.commandList, rootPath,
                                                  space->caseNames)) {
          emitWarning(entry.patternExpr->getLoc(), "case is unreachable; ")
              << "'" << caseNameSpelling(space->caseNames, *ctor)
              << "' is already covered by a previous case"
              << entry.patternExpr->getRange();
          continue;
        }
      }
      if (!hasOr && space->kind == MatchCoveringSpace::Kind::LiteralSet) {
        if (auto literal =
                getSimpleRootLiteralCoverage(entry.commandList, rootPath)) {
          emitWarning(entry.patternExpr->getLoc(), "case is unreachable; ")
              << "'" << *literal << "' is already covered by a previous case"
              << entry.patternExpr->getRange();
          continue;
        }
      }
      emitWarning(entry.patternExpr->getLoc(),
                  "case is unreachable; previous cases cover every "
                  "value of the match subject")
          << entry.patternExpr->getRange();
    } else if (hitOpen && space->isFullyCovered()) {
      entry.isConcluding = true;
    }
  }

  if (space->kind == MatchCoveringSpace::Kind::FiniteCtors &&
      space->hasOpenCtor()) {
    size_t witness = 0;
    while (witness < space->numCtors && !space->remaining[witness])
      ++witness;
    assert(witness < space->numCtors);
    emitWarning(matchLoc, "'match' is not exhaustive; missing case for ")
        << "'" << caseNameSpelling(space->caseNames, witness) << "'";
  } else if (space->kind == MatchCoveringSpace::Kind::Product &&
             space->hasOpenCell()) {
    size_t witness = 0;
    while (witness < space->numCells && !space->cells[witness])
      ++witness;
    assert(witness < space->numCells);
    emitWarning(matchLoc, "'match' is not exhaustive; missing case for ")
        << "'" << formatProductWitness(*space, witness) << "'";
  } else if (space->kind == MatchCoveringSpace::Kind::TooComplex) {
    // A catch-all would have closed the space via alwaysMatches().
    emitWarning(matchLoc, "'match' is too complex to check for exhaustivity; "
                          "add a '_' case");
  }
}
