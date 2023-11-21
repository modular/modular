//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This class provides hooks for performing code completion and signature help
// within a given Mojo source file.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_CODECOMPLETE_H
#define KGEN_MOJOTOOLING_CODECOMPLETE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN::Mojo {
//===----------------------------------------------------------------------===//
// CodeCompletionResult
//===----------------------------------------------------------------------===//

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
    kTrait,
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

//===----------------------------------------------------------------------===//
// SignatureHelpResult
//===----------------------------------------------------------------------===//

/// This class represents a signature help result.
struct SignatureHelpResult {
  /// This class represents a parameter within a signature.
  struct Parameter {
    /// The offset of this parameter within the signature label.
    std::pair<unsigned, unsigned> labelOffset;

    /// The documentation of this parameter.
    std::string documentation;
  };
  /// This class represents a signature.
  struct Signature {
    /// The label of this signature.
    std::string label;

    /// The documentation of this signature.
    std::string documentation;

    /// The parameters of this signature.
    std::vector<Parameter> parameters;
  };

  /// The signatures of this signature help.
  std::vector<Signature> signatures;

  /// The index of the active signature.
  unsigned activeSignature = 0;

  /// The index of the active parameter.
  unsigned activeParameter = 0;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_MOJOTOOLING_CODECOMPLETE_H
