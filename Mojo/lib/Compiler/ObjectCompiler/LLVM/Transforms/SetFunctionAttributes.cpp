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
#include "SetFunctionAttributes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// SetFunctionAttributes
//===----------------------------------------------------------------------===//

PreservedAnalyses SetFunctionAttributes::run(Module &module,
                                             ModuleAnalysisManager &MAM) {
  if (!denormalFP32Mode)
    return PreservedAnalyses::all();

  DenormalMode f32Mode(*denormalFP32Mode, *denormalFP32Mode);
  for (Function &func : module) {
    DenormalFPEnv existing = func.getDenormalFPEnv();
    if (existing.F32Mode == f32Mode)
      continue;
    DenormalFPEnv fpEnv(existing.DefaultMode, f32Mode);
    AttrBuilder attrs(func.getContext());
    attrs.addDenormalFPEnvAttr(fpEnv);
    func.addFnAttrs(attrs);
  }
  return PreservedAnalyses::none();
}
