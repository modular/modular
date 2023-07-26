//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This class provides hooks for performing code completion within a given Mojo
// source file.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CODECOMPLETE_H
#define KGEN_MOJOPARSER_CODECOMPLETE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <string>

namespace M::KGEN {
class CompilationOptions;
} // namespace M::KGEN
namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::Mojo {
/// This class represents a code completion result.
struct CodeCompletionResult {
  enum Kind {
    kUnknown,
    kPackage,
    kModule,
    kStruct,
    kFunction,
    kField,
  };

  CodeCompletionResult() = default;
  CodeCompletionResult(StringRef label, Kind kind)
      : label(label.str()), kind(kind) {}

  /// The label of this completion item.
  std::string label;

  /// The documentation of this completion item.
  std::string documentation;

  /// The kind of this completion item.
  Kind kind = Kind::kUnknown;
};

/// Returns the code completion results for the given buffer at the given
/// completion position.
std::vector<CodeCompletionResult>
codeComplete(llvm::MemoryBufferRef buffer, uint64_t completionPosition,
             MLIRContext *context, LLCL::Runtime &runtime,
             const KGEN::CompilationOptions &options);
} // namespace M::KGEN::Mojo

#endif // KGEN_MOJOPARSER_CODECOMPLETE_H
