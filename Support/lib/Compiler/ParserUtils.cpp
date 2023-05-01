//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/ParserUtils.h"

using namespace M;

/// Parse a parenthesized operand list with types included, e.g.
/// `(%a: i32, %b: f32)`.
ParseResult
M::parseParenOperandList(OpAsmParser &parser, OperationState &result,
                         SmallVectorImpl<OpAsmParser::Argument> &argumentInfo) {
  auto parseOperandFn = [&]() -> ParseResult {
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/true) ||
        parser.resolveOperand(arg.ssaName, arg.type, result.operands))
      return failure();
    argumentInfo.push_back(arg);
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
}
