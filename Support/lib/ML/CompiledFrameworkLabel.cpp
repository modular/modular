//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/CompiledFrameworkLabel.h"

using namespace M;

const char *CompiledFrameworkLabel::getAsOpNameOrNull() const {
  switch (value) {
  case kTensorFlowModel:
    return "tfp.model";
  case kTFLiteModel:
    return "mgp.model";
  case kFauxModel:
    return "faux.testcase";
  default:
    return nullptr;
  }
}

const char *CompiledFrameworkLabel::getAsFrameworkNameOrNull() const {
  switch (value) {
  case kTFLiteModel:
    return "tfl";
  default:
    return nullptr;
  }
}

bool CompiledFrameworkLabel::isValidOpName(StringRef opName) {
  return opName == "tfp.model" || opName == "mgp.model" ||
         opName == "faux.testcase";
}

bool CompiledFrameworkLabel::isValidFrameworkName(StringRef frameworkName) {
  // TODO: "mgp" isn't really a framework
  return frameworkName == "tfl" || frameworkName == "mgp";
}

CompiledFrameworkLabel
CompiledFrameworkLabel::getLabelForOpName(StringRef opName,
                                          StringRef frameworkName) {
  if (opName == "tfp.model")
    return CompiledFrameworkLabel{kTensorFlowModel};
  if (opName == "faux.testcase")
    return CompiledFrameworkLabel{kFauxModel};
  if (opName == "mgp.model") {
    if (frameworkName == "tfl")
      return CompiledFrameworkLabel{kTFLiteModel};
  }
  return CompiledFrameworkLabel{kUnknown};
}

const char *CompiledFrameworkLabel::getAsString() const {
  switch (value) {
  case kUnknown:
    return "unknown";
  case kTensorFlowModel:
    return "compiled TensorFlow model";
  case kTFLiteModel:
    return "compiled TFLite model";
  case kFauxModel:
    return "compiled Faux model";
  }
  llvm::report_fatal_error("invalid CompiledFrameworkLabel value");
};
