//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONSOURCECODE_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONSOURCECODE_H

#include "Support/LLVMForwardDecls.h"
#include <string>

namespace M::KGEN::Mojo {

/// Class that disects the input user expression code into chunks that can be
/// used to generate runnable JIT code.
class MojoExpressionSourceCode {
public:
  MojoExpressionSourceCode(StringRef exprText);

  /// Return the code that should be executed at the top level. It is
  /// eol-terminated or empty.
  const std::string &getTopLevelCode() const { return topLevelCode; }

  /// Return the code that should be executed in the main body of the JITted
  /// function. It is eol-terminated or empty.
  const std::string &getMainBodyCode() const { return mainBodyCode; }

private:
  std::string topLevelCode;
  std::string mainBodyCode;
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONSOURCECODE_H
