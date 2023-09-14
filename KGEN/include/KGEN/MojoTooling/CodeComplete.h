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

#ifndef KGEN_MOJOTOOLING_CODECOMPLETE_H
#define KGEN_MOJOTOOLING_CODECOMPLETE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

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
    kVariable,
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

} // namespace M::KGEN::Mojo

#endif // KGEN_MOJOTOOLING_CODECOMPLETE_H
