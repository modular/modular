//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

const char *CompiledFrameworkLabel::getAsOpNameOrNull() const {
  switch (value) {
  case kUnknown:
    return nullptr;
  case kPyTorchModel:
  case kModularModel:
    return "mgp.model";
  }
  llvm::report_fatal_error("missing case");
}

const char *CompiledFrameworkLabel::getAsFrameworkNameOrNull() const {
  return asLabelString(value);
}

bool CompiledFrameworkLabel::isValidOpName(StringRef opName) {
  return opName == "mgp.model";
}

bool CompiledFrameworkLabel::isValidFrameworkName(StringRef frameworkName) {
  return llvm::is_contained(
      {"tf",
       "mgp", // TODO(#6190): "mgp" isn't really a framework, replace with faux.
       "pytorch", "mof"},
      frameworkName);
}

CompiledFrameworkLabel
CompiledFrameworkLabel::getLabelForOpName(StringRef opName,
                                          StringRef frameworkName) {
  if (opName == "mgp.model") {
    if (frameworkName == "pytorch")
      return CompiledFrameworkLabel{kPyTorchModel};
    if (frameworkName == "mof")
      return CompiledFrameworkLabel{kModularModel};
  }
  llvm::errs() << opName << " & " << frameworkName << "\n";
  return CompiledFrameworkLabel{kUnknown};
}

const char *CompiledFrameworkLabel::getAsString() const {
  switch (value) {
  case kUnknown:
    return "unknown";
  case kPyTorchModel:
    return "compiled PyTorch model";
  case kModularModel:
    return "compiled Modular model";
  }
  llvm::report_fatal_error("missing case");
}

const char *
CompiledFrameworkLabel::asLabelString(CompiledFrameworkLabel::Cases label) {
  switch (label) {
  case kUnknown:
    return nullptr;
  case kPyTorchModel:
    return "pytorch";
  case kModularModel:
    return "mof";
  }
  llvm::report_fatal_error("missing case");
}
