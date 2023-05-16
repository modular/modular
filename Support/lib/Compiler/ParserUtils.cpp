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
                         SmallVectorImpl<OpAsmParser::Argument> &argumentInfo,
                         Type optDefaultType) {
  auto parseOperandFn = [&]() -> ParseResult {
    OpAsmParser::Argument arg;

    // parse argument name
    NamedAttrList attrs;
    if (failed(parser.parseOperand(arg.ssaName, /*allowResultNumber=*/false)))
      return failure();

    // parse optional type annotation
    arg.type = optDefaultType;
    if (succeeded(parser.parseOptionalColon())) {
      if (failed(parser.parseType(arg.type)))
        return failure();
    } else if (!optDefaultType) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected type annotation");
    }
    arg.attrs = attrs.getDictionary(parser.getContext());

    // resolve operand
    if (failed(parser.resolveOperand(arg.ssaName, arg.type, result.operands)))
      return failure();
    argumentInfo.push_back(arg);
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
}
