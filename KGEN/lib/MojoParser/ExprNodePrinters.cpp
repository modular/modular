//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines `print` methods for all the expression nodes.
//
//===----------------------------------------------------------------------===//

#include "ExprNodes.h"
#include "Signatures.h"
#include "mlir/Support/IndentedOstream.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

using mlir::raw_indented_ostream;

static StringRef stringifyExprKind(ExprNode::Kind kind) {
  switch (kind) {
  case ExprNode::kSynthetic:
    return "Synthetic";
  case ExprNode::kIntLiteral:
    return "IntLiteral";
  case ExprNode::kFloatLiteral:
    return "FloatLiteral";
  case ExprNode::kBoolLiteral:
    return "BoolLiteral";
  case ExprNode::kSelfLiteral:
    return "SelfLiteral";
  case ExprNode::kStringLiteral:
    return "StringLiteral";
  case ExprNode::kNoneLiteral:
    return "NoneLiteral";
  case ExprNode::kDiscardLiteral:
    return "DiscardLiteral";
  case ExprNode::kDeclRef:
    return "DeclRef";
  case ExprNode::kAttributeRef:
    return "AttributeRef";
  case ExprNode::kParen:
    return "Paren";
  case ExprNode::kTuple:
    return "Tuple";
  case ExprNode::kList:
    return "List";
  case ExprNode::kDictionary:
    return "Dictionary";
  case ExprNode::kCall:
    return "Call";
  case ExprNode::kSubscript:
    return "Subscript";
  case ExprNode::kSubscriptArrow:
    return "SubscriptArrow";
  case ExprNode::kSlice:
    return "Slice";
  case ExprNode::kDictSubscript:
    return "DictSubscript";
  case ExprNode::kChainedCmp:
    return "ChainedCmp";
  case ExprNode::kFunctionType:
    return "FunctionType";
  case ExprNode::kGetMValueAsLitRef:
    return "GetMValueAsLitRef";
  case ExprNode::kGetLitRefAsMValue:
    return "GetLitRefAsMValue";
  case ExprNode::kGetAddressAsUninitLValue:
    return "GetAddressAsUninitLValue";
  case ExprNode::kGetAddressAsOwned:
    return "GetAddressAsOwned";
  case ExprNode::kGetNearestErrorSlot:
    return "GetNearestErrorSlot";
  case ExprNode::kOriginOf:
    return "OriginOf";
  case ExprNode::kTypeOf:
    return "TypeOf";
  case ExprNode::kNeg:
    return "Neg";
  case ExprNode::kPos:
    return "Pos";
  case ExprNode::kInvert:
    return "Invert";
  case ExprNode::kUnpack:
    return "Unpack";
  case ExprNode::kBoolNot:
    return "BoolNot";
  case ExprNode::kAwait:
    return "Await";
  case ExprNode::kTransfer:
    return "Transfer";
  case ExprNode::kAdd:
    return "Add";
  case ExprNode::kSub:
    return "Sub";
  case ExprNode::kMul:
    return "Mul";
  case ExprNode::kMatMul:
    return "MatMul";
  case ExprNode::kTrueDiv:
    return "TrueDiv";
  case ExprNode::kFloorDiv:
    return "FloorDiv";
  case ExprNode::kMod:
    return "Mod";
  case ExprNode::kBoolOr:
    return "BoolOr";
  case ExprNode::kBoolAnd:
    return "BoolAnd";
  case ExprNode::kCmpIn:
    return "CmpIn";
  case ExprNode::kCmpNotIn:
    return "CmpNotIn";
  case ExprNode::kCmpIs:
    return "CmpIs";
  case ExprNode::kCmpIsNot:
    return "CmpIsNot";
  case ExprNode::kCmpLT:
    return "CmpLT";
  case ExprNode::kCmpLE:
    return "CmpLE";
  case ExprNode::kCmpGT:
    return "CmpGT";
  case ExprNode::kCmpGE:
    return "CmpGE";
  case ExprNode::kCmpNE:
    return "CmpNE";
  case ExprNode::kCmpEQ:
    return "CmpEQ";
  case ExprNode::kOr:
    return "Or";
  case ExprNode::kXor:
    return "Xor";
  case ExprNode::kAnd:
    return "And";
  case ExprNode::kLShift:
    return "LShift";
  case ExprNode::kRShift:
    return "RShift";
  case ExprNode::kPow:
    return "Pow";
  case ExprNode::kWalrus:
    return "Walrus";
  case ExprNode::kAssign:
    return "Assign";
  case ExprNode::kIAdd:
    return "IAdd";
  case ExprNode::kISub:
    return "ISub";
  case ExprNode::kIMul:
    return "IMul";
  case ExprNode::kIMatMul:
    return "IMatMul";
  case ExprNode::kITrueDiv:
    return "ITrueDiv";
  case ExprNode::kIFloorDiv:
    return "IFloorDiv";
  case ExprNode::kIMod:
    return "IMod";
  case ExprNode::kIAnd:
    return "IAnd";
  case ExprNode::kIOr:
    return "IOr";
  case ExprNode::kIXor:
    return "IXor";
  case ExprNode::kILShift:
    return "ILShift";
  case ExprNode::kIRShift:
    return "IRShift";
  case ExprNode::kIPow:
    return "IPow";
  case ExprNode::kIfElse:
    return "IfElse";
  case ExprNode::kInvalid:
    return "Invalid";
  }
}

void ExprNode::dump() const {
  mlir::raw_indented_ostream os(llvm::errs());
  print(os);
}

void SyntheticNode::print(raw_indented_ostream &os) const {
  os << "Synthetic { " << irValue << " }\n";
}

void IntLiteralNode::print(raw_indented_ostream &os) const {
  os << "IntLiteral { " << spelling << " }\n";
}

void FloatLiteralNode::print(raw_indented_ostream &os) const {
  os << "FloatLiteral { " << spelling << " }\n";
}

void BoolLiteralNode::print(raw_indented_ostream &os) const {
  os << "BoolLiteral { " << (value ? "True" : "False") << " }\n";
}

void SimpleLiteralNode::print(raw_indented_ostream &os) const {
  os << "SimpleLiteral { " << stringifyExprKind(kind) << " }\n";
}

void StringLiteralNode::print(raw_indented_ostream &os) const {
  os << "StringLiteral {";
  if (spellings.size() == 1) {
    os << " \"" << spellings.front() << "\" }\n";
    return;
  }
  for (StringRef spelling : spellings)
    os << "  \"" << spelling << "\"\n";
  os << "}\n";
}

static void printIdentifier(raw_indented_ostream &os, const Identifier &id) {
  if (id.isEscaped)
    os << '`';
  os << id.spelling;
  if (id.isEscaped)
    os << '`';
}

void DeclRefNode::print(raw_indented_ostream &os) const {
  os << "DeclRef { ";
  printIdentifier(os, *this);
  os << " }\n";
}

void AttributeRefNode::print(raw_indented_ostream &os) const {
  os << "AttributeRef {\n";
  os.indent() << "base: ";
  base->print(os);
  os << "attr: ";
  printIdentifier(os, *this);
  os << "\n";
  os.unindent() << "}\n";
}

void Operand::print(raw_indented_ostream &os) const {
  auto stringifyPassKind = [&] {
    switch (passKind) {
    case Operand::kPositional:
      return "positional";
    case Operand::kStar:
      return "star";
    case Operand::kKeyword:
      return "keyword";
    case Operand::kStarStar:
      return "starstar";
    }
  };

  os << "{\n";
  os.indent() << "expr: ";
  expr->print(os);
  os << "passKind: " << stringifyPassKind() << "\n";
  os << "name: " << name << "\n";
  os.unindent() << "}\n";
}

void Operand::dump() const {
  raw_indented_ostream os(llvm::errs());
  print(os);
}

void CallNode::print(raw_indented_ostream &os) const {
  os << "Call {\n";
  os.indent() << "callee: ";
  callee->print(os);
  os << "operands: [";
  os.indent();
  for (const Operand &operand : operands)
    operand.print(os);
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

void SubscriptNode::print(raw_indented_ostream &os) const {
  os << "Subscript {\n";
  os.indent() << "base: ";
  base->print(os);
  os << "operands: [";
  os.indent();
  for (const Operand &operand : operands)
    operand.print(os);
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

static void printNullableExpr(raw_indented_ostream &os, const ExprNode *expr) {
  if (!expr)
    os << "<NULL>\n";
  else
    expr->print(os);
}

void SliceNode::print(raw_indented_ostream &os) const {
  os << "Slice {\n";
  os.indent() << "lower: ";
  printNullableExpr(os, lower);
  os << "upper: ";
  printNullableExpr(os, upper);
  os << "stride: ";
  printNullableExpr(os, stride);
  os.unindent() << "}\n";
}

void ParenNode::print(raw_indented_ostream &os) const {
  os << "Paren {\n";
  os.indent() << "subExpr: ";
  subExpr->print(os);
  os.unindent() << "}\n";
}

void TupleNode::print(raw_indented_ostream &os) const {
  os << "Tuple {\n";
  os.indent() << "exprs: [\n";
  for (const ExprNode *expr : exprs)
    expr->print(os);
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

void ListNode::print(raw_indented_ostream &os) const {
  os << "List {\n";
  os.indent() << "exprs: [\n";
  for (const ExprNode *expr : exprs)
    expr->print(os);
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

void DictionaryNode::print(raw_indented_ostream &os) const {
  os << "Dictionary {\n";
  os.indent() << "values: [\n";
  for (auto [name, expr] : values) {
    os << "{\n";
    os.indent() << "name: ";
    name->print(os);
    os << "value: ";
    expr->print(os);
    os.unindent() << "}\n";
  }
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

void DictSubscriptNode::print(raw_indented_ostream &os) const {
  os << "DictSubscript {\n";
  os.indent() << "base: ";
  base->print(os);
  os << "indices: ";
  indices->print(os);
  os.unindent() << "}\n";
}

void IfElseOpNode::print(raw_indented_ostream &os) const {
  os << "IfElseOp {\n";
  os.indent() << "trueExpr: ";
  trueExpr->print(os);
  os << "condExpr: ";
  condExpr->print(os);
  os << "falseExpr: ";
  falseExpr->print(os);
  os.unindent() << "}\n";
}

void BinOpNode::print(raw_indented_ostream &os) const {
  os << "BinOp {\n";
  os.indent() << "kind: " << stringifyExprKind(kind) << "\n";
  os << "lhs: ";
  lhs->print(os);
  os << "rhs: ";
  rhs->print(os);
  os.unindent() << "}\n";
}

void UnaryOpNode::print(raw_indented_ostream &os) const {
  os << "UnaryOp {\n";
  os.indent() << "kind: " << stringifyExprKind(kind) << "\n";
  os << "subExpr: ";
  subExpr->print(os);
  os.unindent() << "}\n";
}

void ChainedCmpOpNode::print(raw_indented_ostream &os) const {
  os << "ChainedCmpOp {\n";
  os.indent() << "exprs: [\n";
  os.indent();
  for (const ExprNode *expr : exprs)
    expr->print(os);
  os.unindent() << "]\n";
  os << "ops: [\n";
  os.indent();
  for (ExprNode::Kind op : ops)
    os << stringifyExprKind(op) << "\n";
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}

void ParsedArgument::print(raw_indented_ostream &os) const {
  auto stringifyConvention = [&] {
    switch (convention) {
    case ParsedArgument::kConventionUnspec:
      return "Unspec";
    case ParsedArgument::kConventionMut:
      return "Mut";
    case ParsedArgument::kConventionOwned:
      return "Owned";
    case ParsedArgument::kConventionRead:
      return "Borrowed";
    case ParsedArgument::kConventionRef:
      return "Ref";
    case ParsedArgument::kConventionByRefResult:
      return "RefResult";
    case ParsedArgument::kConventionOut:
      return "Out";
    }
  };

  auto stringifyKWArgHandling = [&] {
    switch (kwArgHandling) {
    case KWArgHandling::kInferred:
      return "Inferred";
    case KWArgHandling::kPositionalOnly:
      return "PositionalOnly";
    case KWArgHandling::kPositionalOrKeyword:
      return "PositionalOrKeyword";
    case KWArgHandling::kKeywordOnly:
      return "KeywordOnly";
    }
  };

  // The argument might not be type checked yet.
  os << "{\n";
  os.indent() << "convention: " << stringifyConvention() << "\n";
  os << "vararg: " << stringifyVariadicKind(variadicKind) << "\n";
  os << "name: " << name << "\n";
  os << "typeExpr: ";
  printNullableExpr(os, typeExpr);
  os << "initExpr: ";
  printNullableExpr(os, initExpr);
  os << "refOriginExpr: ";
  printNullableExpr(os, refOriginExpr);
  os << "kwArgHandling: " << stringifyKWArgHandling() << "\n";
  os.unindent() << "}\n";
}

void ParsedArgument::dump() const {
  raw_indented_ostream os(llvm::errs());
  print(os);
}

void FunctionTypeNode::print(raw_indented_ostream &os) const {
  os << "FunctionType {\n";
  os << "params: [\n";
  os.indent();
  for (const ParsedArgument &param : parsedParams)
    param.print(os);
  os.unindent() << "]\n";

  os << "args: [\n";
  os.indent();
  for (const ParsedArgument &arg : parsedArgs)
    arg.print(os);
  os.unindent() << "]\n";

  os << "result: [\n";
  os.indent();
  resultArg.print(os);
  os.unindent() << "]\n";

  os << "effects: " << stringifyFnEffects(effects.getImpl()) << "\n";
  os << "originExpr: ";
  originExpr->print(os);

  os.unindent() << "}\n";
}

void MagicFunctionNode::print(raw_indented_ostream &os) const {
  os << "MagicFunction {\n";
  os.indent() << "kind: " << stringifyExprKind(kind) << "\n";
  os << "subExprs: [\n";
  os.indent();
  for (const ExprNode *expr : subExprs)
    expr->print(os);
  os.unindent() << "]\n";
  os.unindent() << "}\n";
}
