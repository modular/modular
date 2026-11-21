//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "LowerKGENToLLVMUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace M::KGEN;
using namespace mlir;

static LLVM::DenormalModeKind
toDialectDenormalModeKind(llvm::DenormalMode::DenormalModeKind kind) {
  switch (kind) {
  case llvm::DenormalMode::IEEE:
    return LLVM::DenormalModeKind::IEEE;
  case llvm::DenormalMode::PreserveSign:
    return LLVM::DenormalModeKind::PreserveSign;
  case llvm::DenormalMode::PositiveZero:
    return LLVM::DenormalModeKind::PositiveZero;
  case llvm::DenormalMode::Dynamic:
    return LLVM::DenormalModeKind::Dynamic;
  case llvm::DenormalMode::Invalid:
    return LLVM::DenormalModeKind::Invalid;
  }
  llvm_unreachable("unknown denormal mode kind");
}

ErrorOrSuccess M::KGEN::setDenormalFPEnv(LLVM::LLVMFuncOp func,
                                         NamedAttrList &attrs,
                                         llvm::StringRef suffix,
                                         Attribute value) {
  auto str = dyn_cast<StringAttr>(value);
  if (!str)
    return Error(llvm::Twine("'llvm.") + suffix + "' expects a string value");
  llvm::DenormalMode mode = llvm::parseDenormalFPAttribute(str.getValue());
  if (!mode.isValid())
    return Error(llvm::Twine("invalid 'llvm.") + suffix + "' value '" +
                 str.getValue() +
                 "'; expected an output mode and an optional input mode, "
                 "such as \"preserve-sign,preserve-sign\"");

  // LLVM assumes these modes when a key is absent. An invalid f32 mode falls
  // back to the default mode.
  LLVM::DenormalModeKind defaultOutput = LLVM::DenormalModeKind::IEEE;
  LLVM::DenormalModeKind defaultInput = LLVM::DenormalModeKind::IEEE;
  LLVM::DenormalModeKind floatOutput = LLVM::DenormalModeKind::Invalid;
  LLVM::DenormalModeKind floatInput = LLVM::DenormalModeKind::Invalid;
  StringAttr envName = func.getDenormalFpenvAttrName();
  if (auto env =
          dyn_cast_if_present<LLVM::DenormalFPEnvAttr>(attrs.get(envName))) {
    defaultOutput = env.getDefaultOutputMode();
    defaultInput = env.getDefaultInputMode();
    floatOutput = env.getFloatOutputMode();
    floatInput = env.getFloatInputMode();
  }
  if (suffix == "denormal_fp_math") {
    defaultOutput = toDialectDenormalModeKind(mode.Output);
    defaultInput = toDialectDenormalModeKind(mode.Input);
  } else {
    floatOutput = toDialectDenormalModeKind(mode.Output);
    floatInput = toDialectDenormalModeKind(mode.Input);
  }
  attrs.set(envName, LLVM::DenormalFPEnvAttr::get(func.getContext(),
                                                  defaultOutput, defaultInput,
                                                  floatOutput, floatInput));
  return M::success();
}
