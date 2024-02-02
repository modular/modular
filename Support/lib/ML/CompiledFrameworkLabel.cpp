//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/CompiledFrameworkLabel.h"

#include "llvm/Support/raw_ostream.h"

using namespace M;

const char *CompiledFrameworkLabel::getAsOpNameOrNull() const {
  switch (value) {
  case kUnknown:
    return nullptr;
  case kTensorFlowModel:
  case kTFLiteModel:
  case kONNXModel:
  case kPyTorchModel:
  case kModularModel:
    return "mgp.model";
  case kFauxModel:
    // TODO(#6190): Support mgp.model for faux.
    return "faux.testcase";
  }
  llvm::report_fatal_error("missing case");
}

const char *CompiledFrameworkLabel::getAsFrameworkNameOrNull() const {
  return asLabelString(value);
}

bool CompiledFrameworkLabel::isValidOpName(StringRef opName) {
  return opName == "mgp.model" ||
         // TODO(#6190)
         opName == "faux.testcase";
}

bool CompiledFrameworkLabel::isValidFrameworkName(StringRef frameworkName) {
  return frameworkName == "tfl" || frameworkName == "tf" ||
         // TODO(#6190): "mgp" isn't really a framework, replace with faux.
         frameworkName == "mgp" || frameworkName == "onnx" ||
         frameworkName == "pytorch" || frameworkName == "mof";
}

CompiledFrameworkLabel
CompiledFrameworkLabel::getLabelForOpName(StringRef opName,
                                          StringRef frameworkName) {
  if (opName == "faux.testcase")
    // TODO(#6190): Support mgp.model for faux.
    return CompiledFrameworkLabel{kFauxModel};
  if (opName == "mgp.model") {
    if (frameworkName == "tfl")
      return CompiledFrameworkLabel{kTFLiteModel};
    else if (frameworkName == "tf")
      return CompiledFrameworkLabel{kTensorFlowModel};
    else if (frameworkName == "onnx")
      return CompiledFrameworkLabel{kONNXModel};
    else if (frameworkName == "pytorch")
      return CompiledFrameworkLabel{kPyTorchModel};
    else if (frameworkName == "mof")
      return CompiledFrameworkLabel{kModularModel};
  }
  llvm::errs() << opName << " & " << frameworkName << "\n";
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
  case kONNXModel:
    return "compiled ONNX model";
  case kPyTorchModel:
    return "compiled PyTorch model";
  case kModularModel:
    return "compiled Modular model";
  }
  llvm::report_fatal_error("missing case");
};

const char *
CompiledFrameworkLabel::asLabelString(CompiledFrameworkLabel::Cases label) {
  switch (label) {
  case kUnknown:
    return nullptr;
  case kTensorFlowModel:
    return "tf";
  case kTFLiteModel:
    return "tfl";
  case kFauxModel:
    // TODO(#6190): Support mgp.model for faux.
    return nullptr;
  case kONNXModel:
    return "onnx";
  case kPyTorchModel:
    return "pytorch";
  case kModularModel:
    return "mof";
  }
  llvm::report_fatal_error("missing case");
}
