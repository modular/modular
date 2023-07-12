//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_H

#define MOJO_EXPR_LOG(...)                                                     \
  do {                                                                         \
    Log *__logPtr = GetLog(LLDBLog::Expressions);                              \
    LLDB_LOG(__logPtr, "[mojo] " __VA_ARGS__);                                 \
  } while (0)

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_H
