//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_PARSINGUTILS_H
#define SUPPORT_COMPILER_PARSINGUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectImplementation.h"

/// Overload the attribute parameter parser for optional int64_ts.
template <>
struct mlir::FieldParser<llvm::Optional<int64_t>> {
  static FailureOr<Optional<int64_t>> parse(AsmParser &parser) {
    int64_t value = 0;
    OptionalParseResult result = parser.parseOptionalInteger(value);
    if (result.has_value()) {
      if (succeeded(*result))
        return {Optional<int64_t>(value)};
      return failure();
    }
    return {llvm::None};
  }
};

#endif // SUPPORT_COMPILER_PARSINGUTILS_H
