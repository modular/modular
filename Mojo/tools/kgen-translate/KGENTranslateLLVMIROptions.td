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

include "KGENTranslateMLIROptions.td"

def KGENTranslateLLVMIRDescription : CommandDescription<
  "kgen-translate", "-mlir-to-llvmir", "Translates MLIR to LLVM IR.",
  "Parses MLIR that uses the KGEN and LLVM dialects and prints its "
  "translation to LLVM IR.",
  [UsageInfo<"options", "file", /*variadicInput=*/false>]>;

def mlir_to_llvmir : MLIRToLLVMIRFlag;

def no_implicit_module : Flag<["--", "-"], "no-implicit-module">,
  Group<TestingOptionGroup>,
  HelpText<"Requires the input to be a single top-level operation, rather "
           "than wrapping it in an implicit module.">,
  MarshallingInfoFlag<TranslateOpts<"noImplicitModule">>;
