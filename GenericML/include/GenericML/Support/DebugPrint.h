//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_DEBUGPRINT_H
#define GENERICML_SUPPORT_DEBUGPRINT_H

#include "ArraySupport/Tensor.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"

#include <string>

namespace M {

/// Options, conveyed via a runtime context object, which control the output
/// format for debug printing of tensor values.
struct DebugTensorPrintOptions {
  /// Format to use.
  ResultOutputStyle style = ResultOutputStyle::kCompact;
  /// Precision of textual floating point numbers.
  unsigned precision = 6;
  /// If outputStyle is binary, the directory in which to create the files,
  /// using the 'label' for the basename.
  std::string binaryDir;

  DebugTensorPrintOptions() = default;
  DebugTensorPrintOptions(ResultOutputStyle style, unsigned precision,
                          std::string binaryDir)
      : style(style), precision(precision), binaryDir(std::move(binaryDir)) {}

  /// Prints the tensor with CPU-hosted buffer contents and spec, following
  /// the options of this object, and using label if given to disambiguate
  /// the result.
  ErrorOrSuccess printTensor(const void *buffer, const TensorSpec &spec,
                             StringRef label);
};

} // namespace M

#endif // GENERICML_SUPPORT_DEBUGPRINT_H
